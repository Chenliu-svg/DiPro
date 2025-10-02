"""
1. split the dataset basing on suject_id
2. build data and label csv file for each task
"""
import argparse
import os
import sys
from tqdm import tqdm
import pandas as pd
import warnings
from datetime import timedelta
from collections import defaultdict
import ast
from mimic3benchmark.constants import SELECTED_REGIONS,PROGRESSION_DISEASE
warnings.filterwarnings("ignore")



parser = argparse.ArgumentParser(description='get cxr for each icu stay.')
parser.add_argument('--all_stay_csv_path', type=str, required=True, help='the all_stay_csv path, containing the all the icu_stay samples needed.')
parser.add_argument('--cxr_stay', type=str, required=True,
                    help='aligned cxr and stay csv path')
parser.add_argument('--split_save_dir', type=str, help='the path of the aligned CXR csv') 
parser.add_argument('--golden_dir', type=str, required=True, help='the path of the golden labels')
parser.add_argument('--progression_task_label_dir', type=str, required=True, help='the path of the golden labels')
parser.add_argument('--all_comparison_path', type=str, required=True, help='path to the extracted comparison from Chest-ImaGenome')
args, _ = parser.parse_known_args()

def get_prevalence(df):
    value_counts = df.apply(lambda x: x.value_counts())
    value_counts = value_counts.reindex([0, 1], fill_value=0)
    ratio = value_counts.loc[0] / value_counts.loc[1]
    ratio=ratio.round(3)
    return ratio


def split_dataset_for_each_task(all_stay_csv_path,cxr_stay,golden_dir,save_dir,progression_task_label_dir):
    # split the dataset into train, val and test, get the predicted labels, the prevalence of each label
    import os
    from sklearn.model_selection import train_test_split

    cxr_stay=pd.read_csv(cxr_stay)
    subject_ids=list(cxr_stay.subject_id.unique())

    # get golden subject_id
    golden_labels=pd.read_csv(golden_dir, sep='\t')[['patient_id','current_image_id','previous_image_id','label_name','comparison']]
    golden_subject_ids=list(golden_labels.patient_id.unique())
    
    # print(f'total test subject ids: {len(subject_ids)*0.2}')
    # print(f'{len([subject_id for subject_id in subject_ids if subject_id in golden_subject_ids])}') # 39 not enough for test
    # breakpoint()
    # golden set should be in the test set
    fixed_test_subject_ids=[subject_id for subject_id in subject_ids if subject_id in golden_subject_ids]
    left_subject_ids=[subject_id for subject_id in subject_ids if subject_id not in fixed_test_subject_ids]
    # split the patient using suject id
 
    random_seed=0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # train_val,test=train_test_split(subject_ids, test_size=0.2,random_state=random_seed)
    left_test_ratios=(len(subject_ids)*0.2-len(fixed_test_subject_ids))/len(subject_ids)
    train_val,test=train_test_split(left_subject_ids, test_size=left_test_ratios, random_state=random_seed)
    test=test+fixed_test_subject_ids
    train,val=train_test_split(train_val, test_size=1/8, random_state=random_seed)
    train_df=pd.DataFrame(data={'subject_id':train})
    val_df=pd.DataFrame(data={'subject_id':val})
    test_df=pd.DataFrame(data={'subject_id':test})
    


    ### Build the dataset for disease progression
    # x:[cxr_i,cxr_(i+1)], ehr: [max(time of cxr_i-\delta,0),time of cxr_(i+1)+\delta], y: [disease progression labels of cxr_i to cxr_(i+1)]
    disease_list=['lung opacity','pleural effusion','atelectasis','enlarged cardiac silhouette','pulmonary edema','consolidation','pneumonia']
    disease_progression_dir=os.path.join(save_dir,'disease_progression')
    if not os.path.exists(disease_progression_dir):
        os.mkdir(disease_progression_dir)
    for disease in tqdm(disease_list, total=len(disease_list),desc='split each disease'):
        disase_save_dir=os.path.join(disease_progression_dir,disease)
        if not os.path.exists(disase_save_dir):
            os.mkdir(disase_save_dir)
        disease_df=pd.read_csv(os.path.join(progression_task_label_dir,f'{disease}_image_pairs.csv'))
        # drop progression column
        disease_df=disease_df.drop(columns=['progression'])
        disease_progression_train_df=pd.merge(train_df,disease_df,on='subject_id',how='left')
        disease_progression_val_df=pd.merge(val_df,disease_df,on='subject_id',how='left')
        disease_progression_test_df=pd.merge(test_df,disease_df,on='subject_id',how='left')
        # drop the nan
        disease_progression_train_df=disease_progression_train_df.dropna()
        disease_progression_val_df=disease_progression_val_df.dropna()
        disease_progression_test_df=disease_progression_test_df.dropna()
        # save the df
        disease_progression_train_df.to_csv(os.path.join(disase_save_dir,f'{disease}_train.csv'),index=False)
        disease_progression_val_df.to_csv(os.path.join(disase_save_dir,f'{disease}_val.csv'),index=False)
        disease_progression_test_df.to_csv(os.path.join(disase_save_dir,f'{disease}_test.csv'),index=False)
          
    ### Build the dataset for prediction of mortality_inhospital, los

    # cxr taken less then 48 hours 
    # x: [cxr_0,...,cxr_n], ehr: [0,time of cxr_(n+1)] y：[mortality_inhospital, los]
    all_stay=pd.read_csv(all_stay_csv_path)
    mortality_los_save_dir=os.path.join(save_dir,'mortality_los')
    if not os.path.exists(mortality_los_save_dir):
        os.mkdir(mortality_los_save_dir)
    # get the subset for task 3:
    mortality_los_cxr_stay=cxr_stay[(cxr_stay.CXRRelTime<48)]
    mortality_los_sample_dict={'subject_id':[],'stay_id':[],'cxr_list':[],'mortality_inhospital':[],'los':[]}
    mortality_los_stay_id_list=mortality_los_cxr_stay.stay_id.unique()
    for stay_id in tqdm(mortality_los_stay_id_list, total=len(mortality_los_stay_id_list),desc='split dataset for general ICU prediction'): 
        stay_df=mortality_los_cxr_stay[mortality_los_cxr_stay.stay_id==stay_id]
        subject_id=stay_df.subject_id.values[0]
        # sort the cxr based on the CxrRelTime
        stay_df=stay_df.sort_values(by='CXRRelTime')
        cxr_list=stay_df[['dicom_id','CXRRelTime']].values
        if len(cxr_list)<2:
                continue
        # turn into tuple
        cxr_list=[tuple(x) for x in cxr_list]
        mortality_los_sample_dict['subject_id'].append(subject_id)
        mortality_los_sample_dict['stay_id'].append(stay_id)
        mortality_los_sample_dict['cxr_list'].append(cxr_list)
        mortality_inhospital=all_stay[(all_stay.subject_id==subject_id)&(all_stay.stay_id==stay_id)]['mortality_inhospital'].values[0]
        los=all_stay[(all_stay.subject_id==subject_id)&(all_stay.stay_id==stay_id)]['los'].values[0]
        mortality_los_sample_dict['mortality_inhospital'].append(mortality_inhospital)
        mortality_los_sample_dict['los'].append(los)
    mortality_los_sample_df=pd.DataFrame(mortality_los_sample_dict)
    # assert no nan in the mortality_inhospital and los
    assert mortality_los_sample_df['mortality_inhospital'].isnull().sum()==0
    assert mortality_los_sample_df['los'].isnull().sum()==0
    # split into train, val and test
    mortality_los_train_df=pd.merge(train_df,mortality_los_sample_df,on='subject_id',how='left')
    mortality_los_val_df=pd.merge(val_df,mortality_los_sample_df,on='subject_id',how='left')
    mortality_los_test_df=pd.merge(test_df,mortality_los_sample_df,on='subject_id',how='left')
    # drop the nan
    mortality_los_train_df=mortality_los_train_df.dropna()
    mortality_los_val_df=mortality_los_val_df.dropna()
    mortality_los_test_df=mortality_los_test_df.dropna()
    # save the df
    mortality_los_train_df.to_csv(os.path.join(mortality_los_save_dir,'train.csv'),index=False)
    mortality_los_val_df.to_csv(os.path.join(mortality_los_save_dir,'val.csv'),index=False)
    mortality_los_test_df.to_csv(os.path.join(mortality_los_save_dir,'test.csv'),index=False)
    
def merge_progression_task(disease_csv_dir):

    df_list={'train':[],'val':[],'test':[]}
    for split in ['train','test','val']:
        for disease in PROGRESSION_DISEASE:
            #  subject_id,study_id,cxr1_dicom_id,cxr2_dicom_id,disease,corrected_progression,from_golden,stay_id,cxr1_time,cxr2_time
            df_temp=pd.read_csv(os.path.join(disease_csv_dir,'disease_progression',disease,f'{disease}_{split}.csv'))[['subject_id', 'study_id', 'cxr1_dicom_id', 'cxr2_dicom_id', 'stay_id',"cxr1_time","cxr2_time",'corrected_progression']]
            df_list[split].append(df_temp)
        
        merged_df = None
        for i, df in enumerate(df_list[split]):
            
            temp_df = df.copy()
            temp_df = temp_df.rename(columns={'corrected_progression': PROGRESSION_DISEASE[i]})
            if merged_df is None:
                merged_df = temp_df
            else:

                merged_df = pd.merge(
                    merged_df,
                    temp_df,
                    on=['subject_id', 'study_id', 'cxr1_dicom_id', 'cxr2_dicom_id', 'stay_id',"cxr1_time","cxr2_time"],
                    how='outer'
                )

        merged_df.to_csv(os.path.join(disease_csv_dir,'disease_progression',f'{split}.csv'),index=False)
    
def get_organ_progression_labels_dp(
    golden_dir,
    all_comparison_path,
    split_dir,
    save_dir
):

    all_comparison = pd.read_csv(all_comparison_path)
    all_comparison['disease_progression'] = all_comparison['disease_progression'].apply(ast.literal_eval)
    
    golden_labels = pd.read_csv(golden_dir, sep='\t')[
        ['previous_image_id', 'current_image_id', 'comparison', 'bbox', 'label_name']
    ]

    progression_dict = defaultdict(list)
    for _, row in all_comparison.iterrows():
        key = (row['cxr1_dicom_id'], row['cxr2_dicom_id'])
        progression_dict[key] = row['disease_progression']
    
    golden_dict = defaultdict(list)
    for _, row in golden_labels.iterrows():
        key = (row['previous_image_id'], row['current_image_id'])
        golden_dict[key].append((row['bbox'], row['label_name'], row['comparison']))
        
    os.makedirs(save_dir, exist_ok=True)
    
    for s in ['train', 'val', 'test']:
        input_path = os.path.join(
            split_dir,'disease_progression',f'{s}.csv'
        )
        output_path = os.path.join(save_dir,'disease_progression', f'{s}.csv')
        
        progression_task_df = pd.read_csv(input_path)
        
        for organ in SELECTED_REGIONS:
            progression_task_df[f'{organ}_progression'] = [
                {k: 'not mentioned' for k in PROGRESSION_DISEASE}
                for _ in range(len(progression_task_df))
            ]
        
        for idx, row in tqdm(progression_task_df.iterrows(), desc=f'Processing anatomical disease progression for {s} subset'):
            cxr1, cxr2 = row['cxr1_dicom_id'], row['cxr2_dicom_id']
            key = (cxr1, cxr2)
        
            if key in progression_dict:
                for organ_disease in progression_dict[key]:
                    organ, diseases, progression = organ_disease
                    if (organ in SELECTED_REGIONS) and (len(progression) == 1):
                        for disease in diseases:
                            if disease in PROGRESSION_DISEASE:
                              
                                new_dict = progression_task_df.at[idx, f'{organ}_progression'].copy()
                                new_dict[disease] = progression[0]
                                progression_task_df.at[idx, f'{organ}_progression'] = new_dict
            
            if key in golden_dict:
                for organ, disease, comparison in golden_dict[key]:
                    if (organ in SELECTED_REGIONS) and (disease in PROGRESSION_DISEASE):
                        new_dict = progression_task_df.at[idx, f'{organ}_progression'].copy()
                        new_dict[disease] = comparison
                        progression_task_df.at[idx, f'{organ}_progression'] = new_dict
        

        progression_task_df.to_csv(output_path, index=False)                     


def get_organ_progression_labels_mort_los(
    golden_dir,
    all_comparison_path,
    split_dir,
    save_dir
):
  
    all_comparison = pd.read_csv(all_comparison_path)
    all_comparison['disease_progression'] = all_comparison['disease_progression'].apply(ast.literal_eval)
    
    golden_labels = pd.read_csv(golden_dir, sep='\t')[
        ['previous_image_id', 'current_image_id', 'comparison', 'bbox', 'label_name']
    ]

    progression_dict = defaultdict(list)
    for _, row in all_comparison.iterrows():
        key = (row['cxr1_dicom_id'], row['cxr2_dicom_id'])
        progression_dict[key] = row['disease_progression']
    
    golden_dict = defaultdict(list)
    for _, row in golden_labels.iterrows():
        key = (row['previous_image_id'], row['current_image_id'])
        golden_dict[key].append((row['bbox'], row['label_name'], row['comparison']))
    

   
    os.makedirs(save_dir, exist_ok=True)
    
    for s in ['train', 'val', 'test']:
        input_path = os.path.join(split_dir, 'mortality_los',
            f'{s}.csv'
        )
        output_path = os.path.join(save_dir, 'mortality_los', f'{s}.csv')
        
        mort_los_df = pd.read_csv(input_path)
        mort_los_df['cxr_list'] = mort_los_df['cxr_list'].apply(ast.literal_eval)
        for organ in SELECTED_REGIONS:
            mort_los_df[f'{organ}_progression'] = [
                []
                for _ in range(len(mort_los_df))
            ]
        
        for idx, row in tqdm(mort_los_df.iterrows(), desc=f'Processing anatomical disease progression for {s} subset'):
            
            cxr_list= row['cxr_list']
            organ_disease_dict ={}
            # initial with 'not mentioned'
            for organ in SELECTED_REGIONS:
                organ_disease_dict[f'{organ}_progression'] = [{k: 'not mentioned' for k in PROGRESSION_DISEASE} for _ in range(len(cxr_list)-1)]
                    
            for i in range(len(cxr_list)-1):
            
                
                cxr1, cxr2 = cxr_list[i][0], cxr_list[i+1][0]
                key = (cxr1, cxr2)
            
                if key in progression_dict:
                    for organ_disease in progression_dict[key]:
                        organ, diseases, progression = organ_disease
                        if (organ in SELECTED_REGIONS) and (len(progression) == 1):
                            for disease in diseases:
                                if disease in PROGRESSION_DISEASE:
                                    organ_disease_dict[f'{organ}_progression'][i][disease] = progression[0]
                if key in golden_dict:
                    for organ, disease, comparison in golden_dict[key]:
                        if (organ in SELECTED_REGIONS) and (disease in PROGRESSION_DISEASE):
                            organ_disease_dict[f'{organ}_progression'][i][disease] = comparison
            
            for organ in SELECTED_REGIONS:
                mort_los_df.at[idx, f'{organ}_progression'] = organ_disease_dict[f'{organ}_progression']
                                    
        mort_los_df.to_csv(output_path, index=False)                     

  


if __name__ == '__main__':
    split_dataset_for_each_task(args.all_stay_csv_path,args.cxr_stay,args.golden_dir,args.split_save_dir,args.progression_task_label_dir)
    merge_progression_task(args.split_save_dir)
    print('getting anatomical disease progression labels for disease progression task')
    get_organ_progression_labels_dp(args.golden_dir, args.all_comparison_path, args.split_save_dir, args.split_save_dir)
    print('getting anatomical disease progression labels for mortality and length of stay task')
    get_organ_progression_labels_mort_los(args.golden_dir, args.all_comparison_path,args.split_save_dir, args.split_save_dir )

        