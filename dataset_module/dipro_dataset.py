import os
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import ast
from mimic_data_extract.mimic3benchmark.constants import SELECTED_VAR, DEMOGRAPHIC, PROGRESSION_DISEASE, PROGRESSION_DICT, FILL_VALUE, SELECTED_REGIONS, LOS_bins, LOS_labels, IMAGE_SIZE
import warnings

warnings.filterwarnings("ignore")


###### define dataset class ######
class DiproDataset(Dataset):
    """
    Args:
        partition (str): train, val or test
        task_name (str): disease_progression, mortality, length_of_stay
        label_path (str): label path containing task specific labels and input cxr meta data
        demographic_path (str): demographic path
        ehr_time_series_path (str): ehr time series path
        bbox_csv (str): bbox csv path
        mimic_cxr_path (str): mimic-cxr-jpg path (e.g, ../mimic-cxr-jpg/2.0.0/files)
        cxr_meta_path (str): mimic-cxr meta path  (e.g, ../mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv)
    """
    def __init__(self,
                 partition,
                 task_name,
                 label_path,
                 demographic_path,
                 ehr_time_series_path,
                 bbox_csv,
                 mimic_cxr_path,
                 cxr_meta_path,
                 input_size=IMAGE_SIZE,
                 ):
        super().__init__()
        self.partition = partition
        self.task_name=task_name
        self.label_path=label_path
        self.ehr_time_series_path=ehr_time_series_path
        self.demographic_df=pd.read_csv(demographic_path)
        self.original_bbox=pd.read_csv(bbox_csv)
        self.mimic_cxr_path=mimic_cxr_path

        self.mimic_meta=pd.read_csv(cxr_meta_path)[['dicom_id','subject_id','study_id']]
        # set dicom_id as index
        self.mimic_meta.set_index('dicom_id',inplace=True)

        self.input_size=input_size
        
        
        if self.task_name=='disease_progression':
            # get the label csv file
            self.label_df = pd.read_csv(os.path.join(self.label_path, task_name, f'{self.partition}.csv'))

            # get image level progression labels
            for col in PROGRESSION_DISEASE:
                self.label_df[col] = self.label_df[col].map(PROGRESSION_DICT)
                self.label_df[col] = self.label_df[col].fillna(FILL_VALUE)

            # get anatomical region progression labels for the PAE module
            for organ in SELECTED_REGIONS:
                self.label_df[f'{organ}_progression']=self.label_df[f'{organ}_progression'].apply(ast.literal_eval)
                
            

        elif self.task_name in ['mortality','length_of_stay']:
            # get the label csv file
            self.label_df = pd.read_csv(os.path.join(self.label_path, 'mortality_los', f'{self.partition}.csv'))

            # get meta data for input CXRs
            self.label_df['cxr_list']=self.label_df['cxr_list'].apply(ast.literal_eval)

            # get anatomical region progression labels for the PAE module
            for organ in SELECTED_REGIONS:
                self.label_df[f'{organ}_progression']=self.label_df[f'{organ}_progression'].apply(ast.literal_eval)
            
            # get length of stay labels
            self.label_df['los_bin'] = pd.cut(
                self.label_df['los'],
                bins=LOS_bins,
                labels=LOS_labels,
                right=False,        
                include_lowest=True
            )
            self.label_df['los_bin'] = self.label_df['los_bin'].astype('int8')
            
        else:
            raise NotImplementedError
        
        # get demographic ehr
        self.demographic_df=pd.read_csv(demographic_path)
        self.demographic_df=self.demographic_df[self.demographic_df['stay_id'].isin(self.label_df['stay_id'])]
    
        # Data augmentation from ChexRelNet
        self.input_size=input_size
        self.transform= transforms.Compose([
        transforms.Resize(size=(input_size, input_size),
                          interpolation=transforms.functional.InterpolationMode.NEAREST),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])

    def get_cxr_regions(self, image, bbox_info):
        """
        Args:
            image: PIL.Image
            bbox_info: pd.DataFrame
        Returns:
            cxr_regions: torch.Tensor [num_region, channel, w, h]
        """
        cxr_regions = [] 
        
        for bbox_name in SELECTED_REGIONS:
            # get bbox
            bbox = ast.literal_eval(bbox_info[bbox_name].values[0])
            if bbox==[0,0,0,0]:
                cxr_regions.append(torch.zeros(3, self.input_size, self.input_size))
                continue

            # crop and augment each region    
            cropped = image.crop(bbox)
            if self.transform is not None:
                cropped = self.transform(cropped)
            cxr_regions.append(cropped)        
        
        return torch.stack(cxr_regions, dim=0) # [num_region, channel, w, h]
    
    def ensemble_cxr_regions(self, input_cxrs_paths):
        """
        Args:
            input_cxrs_paths: list
        Returns:
            cxr_regions: torch.Tensor [num_cxr, num_region, channel, w, h]
        """
        cxr_regions_ensemble=[]
        # get each cxr and get the bbox of each region then augment each region
        for cxr_dicom_id in input_cxrs_paths:

            # get subject_id and study_id
            subject_id=self.mimic_meta.loc[cxr_dicom_id,'subject_id']
            study_id=self.mimic_meta.loc[cxr_dicom_id,'study_id']
            cxr_path = os.path.join(self.mimic_cxr_path, f'p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{cxr_dicom_id}.jpg')
            cxr = Image.open(cxr_path).convert('RGB')

            bbox_info=self.original_bbox[self.original_bbox['cxr_id']==cxr_dicom_id]
            cxr=self.get_cxr_regions(cxr,bbox_info)
            cxr_regions_ensemble.append(cxr)
        cxr_regions_ensemble=torch.stack(cxr_regions_ensemble) 
        return cxr_regions_ensemble

    def parse_progression_dict(self,progression_dict):
        """
        Parse the anatomic region progression labels
        """
        result_ls=[]
        for disease in PROGRESSION_DISEASE:
            t_pro=PROGRESSION_DICT[progression_dict[disease]]
            result_ls.append(t_pro)
        return result_ls

    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, index):
        """
        returns: 
            ehr: torch.Tensor [num_timestep, num_ts_var]
            cxr_regions: torch.Tensor [num_cxr, num_region, channel, w, h]
            demograhic_ehr: torch.Tensor [num_demographic_var]
            y: torch.Tensor, disease labels  
            y_aux: torch.Tensor, disease progression task: [num_region, num_disease]; general ICU tsks: [num_cxr-1, num_region, num_disease]
            ehr_timestep: torch.Tensor, disease progression task: [time interval between two CXRs]; general ICU tsks: [48]
            interval_times: torch.Tensor, disease progression task: [2]; general ICU tsks: [num_cxr-1, 2]
        """


        # get the ehr time-series data for the current stay
        stay_id=self.label_df.loc[index,'stay_id']
        ehr_ts_data = pd.read_csv(os.path.join(self.ehr_time_series_path, 'New_episode'+str(int(stay_id))+'_timeseries.csv'))
        
        # get demographic ehr
        demograhic_ehr=self.demographic_df[self.demographic_df['stay_id']==stay_id][DEMOGRAPHIC].values[0]
        demograhic_ehr=torch.from_numpy(demograhic_ehr).float()
            
        
        if self.task_name=='disease_progression':
            cxr1_time=self.label_df.loc[index,'cxr1_time']
            cxr2_time=self.label_df.loc[index,'cxr2_time']

            # get ehr time-series data between the two CXRs taken time
            ehr=ehr_ts_data[(ehr_ts_data['regular_interval']-1<=cxr2_time)&(ehr_ts_data['regular_interval']+1>=cxr1_time)][SELECTED_VAR].values
            ehr = torch.from_numpy(ehr).float()

            # get the ehr timesteps for the MMF module
            ehr_timestep=torch.arange(int(cxr1_time),int(cxr1_time)+len(ehr), dtype=torch.float)
            
            # get the CXR pair taken times for the MMF module
            cxr_pairs_time=[[cxr1_time,cxr2_time]]
            interval_times=torch.tensor(cxr_pairs_time).float()

            # get the cxr pair
            cxr1_dicom_id=self.label_df.loc[index,'cxr1_dicom_id']
            cxr2_dicom_id=self.label_df.loc[index,'cxr2_dicom_id']
            input_cxrs_paths=[cxr1_dicom_id,cxr2_dicom_id]
            # get the anatomic regions for the cxr pair
            cxr_regions=self.ensemble_cxr_regions(input_cxrs_paths)

            # get labels
            progression = self.label_df.loc[index, PROGRESSION_DISEASE].values
            y = torch.tensor(list(progression)).long()

            # get the anatomical region disease progression labels for the PAE module
            organ_disease_ls=[]
            for organ in SELECTED_REGIONS:
                organ_disease_ls.append(self.parse_progression_dict(self.label_df.loc[index,f'{organ}_progression']))
            y_aux=torch.tensor([organ_disease_ls]).long()  # num_region * num_disease

        else:
            assert self.task_name in ['mortality','length_of_stay']

            # get 48 hours ehr
            ehr=ehr_ts_data[ehr_ts_data['regular_interval']<=48][SELECTED_VAR].values
            ehr = torch.from_numpy(ehr).float()

            # get the ehr timesteps for the MMF module
            ehr_timestep=torch.arange(0,49, dtype=torch.float)

            # get longitudinal CXRs
            cxr_list=self.label_df.loc[index,'cxr_list']
            input_cxrs_paths=[i[0] for i in cxr_list]
            # get the anatomic regions for the longitudinal CXRs
            cxr_regions=self.ensemble_cxr_regions(input_cxrs_paths)

            # get the CXRs taken times for the MMF module
            cxr_pairs_time=[[cxr_list[i][1],cxr_list[i+1][1]] for i in range(len(cxr_list)-1)]
            interval_times=torch.tensor(cxr_pairs_time).float()
            
            # get labels
            if self.task_name=='mortality':
                y = torch.tensor(self.label_df.loc[index, 'mortality_inhospital']).to(torch.float)
            else:
                assert self.task_name=='length_of_stay'
                y = torch.tensor(self.label_df.loc[index, 'los_bin']).long()

            # get the anatomical region disease progression labels for the PAE module
            organ_disease_ls=[]
            for pair in range(len(cxr_list)-1):
                pair_region_ls=[]
                for organ in SELECTED_REGIONS:
                    pair_region_ls.append(self.parse_progression_dict(self.label_df.loc[index,f'{organ}_progression'][pair]))
                organ_disease_ls.append(pair_region_ls)
                
            y_aux=torch.tensor(organ_disease_ls).long()  # num_pairs*num_region * num_disease [num_pairs=num_cxr-1]

        return ehr, cxr_regions, demograhic_ehr, y, y_aux, ehr_timestep, interval_times



class CollateFunc():
    def __init__(self, dim=0):
        self.dim = dim
        

    @staticmethod
    def pad_tensor(vec, pad, dim):
        if isinstance(vec,np.ndarray):
            vec = torch.from_numpy(vec)
        origin_shape = vec.shape
        # print(origin_shape)
        pad_size = list(origin_shape)
        pad_size[dim] = pad - vec.size(dim)
        padded_vec = torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
        mask = torch.cat([torch.ones(origin_shape[dim]), torch.zeros(pad - origin_shape[dim])], dim=-1)
        return padded_vec, mask.to(torch.bool)
    
    @staticmethod
    def pad_tensor_y_aux(vec, pad, dim):
        if isinstance(vec,np.ndarray):
            vec = torch.from_numpy(vec)
        origin_shape = vec.shape
        # print(origin_shape)
        pad_size = list(origin_shape)
        pad_size[dim] = pad - vec.size(dim)
        padded_vec = torch.cat([vec, torch.ones(*pad_size)*FILL_VALUE], dim=dim)
        mask = torch.cat([torch.ones(origin_shape[dim]), torch.zeros(pad - origin_shape[dim])], dim=-1)
        return padded_vec, mask.to(torch.bool)
    
    def __call__(self, batch):
        ehr_max_len = max(map(lambda x: x[0].shape[0], batch))
        padded_ehr_mask = list(map(lambda x: self.pad_tensor(x[0], pad=ehr_max_len, dim=0), batch))
        reshaped_ehr = torch.stack(list(map(lambda x: x[0], padded_ehr_mask)), dim=0)
        reshaped_ehr_mask = torch.stack(list(map(lambda x: x[1], padded_ehr_mask)), dim=0)

        region_max_len = max(map(lambda x: x[1].shape[0], batch))
        padded_region_mask = list(map(lambda x: self.pad_tensor(x[1], pad=region_max_len, dim=0), batch))
        reshaped_region = torch.stack(list(map(lambda x: x[0], padded_region_mask)), dim=0)
        reshaped_region_mask = torch.stack(list(map(lambda x: x[1], padded_region_mask)), dim=0)

        demograhic_ehr=torch.stack(list(map(lambda x: x[2], batch)), dim=0)
        y=torch.stack(list(map(lambda x: x[3], batch)), dim=0)

        y_aux_max_len = max(map(lambda x: x[4].shape[0], batch))
        padded_y_aux_mask = list(map(lambda x: self.pad_tensor_y_aux(x[4], pad=y_aux_max_len, dim=0), batch))
        reshaped_y_aux = torch.stack(list(map(lambda x: x[0], padded_y_aux_mask)), dim=0)
        reshaped_y_aux_mask = torch.stack(list(map(lambda x: x[1], padded_y_aux_mask)), dim=0)
        
        ehr_time_max_len = max(map(lambda x: x[5].shape[0], batch))
        padded_ehr_time_mask = list(map(lambda x: self.pad_tensor_y_aux(x[5], pad=ehr_time_max_len, dim=0), batch))
        reshaped_ehr_time = torch.stack(list(map(lambda x: x[0], padded_ehr_time_mask)), dim=0)

        cxr_time_max_len = max(map(lambda x: x[6].shape[0], batch))
        padded_cxr_time_mask = list(map(lambda x: self.pad_tensor_y_aux(x[6], pad=cxr_time_max_len, dim=0), batch))
        reshaped_cxr_time = torch.stack(list(map(lambda x: x[0], padded_cxr_time_mask)), dim=0)
        
        return reshaped_ehr, reshaped_ehr_mask, reshaped_region, reshaped_region_mask, demograhic_ehr, y, reshaped_y_aux, reshaped_y_aux_mask, reshaped_ehr_time, reshaped_cxr_time
  