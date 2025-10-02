"""
1. match cxr(AP) and ehr in a icu stay, (path, taken time, )
2. check if the cxr has bbounding box
"""
import argparse
import os
import sys
from tqdm import tqdm
import pandas as pd
import warnings
from datetime import timedelta
import json
from mimic3benchmark.constants import FINDINGS, REGIONS

warnings.filterwarnings("ignore")



parser = argparse.ArgumentParser(description='get cxr for each icu stay.')
parser.add_argument('--all_stay_csv_path', type=str, required=True, help='the all_stay_csv path, containing the all the icu_stay samples needed.')
parser.add_argument('--mimic_cxr_meta', type=str, required=True,
                    help='mimic-cxr meta csv data path')
parser.add_argument('--chexpert_label', type=str,required=True, help='mimic-cxr chexpert labels data path')
parser.add_argument('--ImaGenome', type=str, required=True,
                     help='the ImaGenome dataset path')
parser.add_argument('--output_dir', type=str, help='the path of the aligned CXR csv') 

args, _ = parser.parse_known_args()

one_day=timedelta(days=1)


def align_cxr_with_stay(cxr_mata, chexpert_label, ImaGenome, all_icu_stay, output_dir):
    """
    input: cxr_mata, chexpert_label, ImaGenome, all_icu_stay, output_dir
    output: cxr_stay.csv with columns: subject_id, stay_id, study_id, dicom_id, CXRTime (CXR_study_time-icu_intime), ViewPosition,  chexpert labels)
    """
    # get cxr
    cxr_meta = pd.read_csv(cxr_mata,dtype={'StudyDate': str, 'StudyTime': str})

    # convert time to pd.datetime
    cxr_meta['StudyTime'] = cxr_meta['StudyTime'].apply(lambda x: f'{int(float(x)):06}' )
    cxr_meta['StudyDateTime'] = pd.to_datetime(cxr_meta['StudyDate'].astype(str) + ' ' + cxr_meta['StudyTime'].astype(str) ,format="%Y%m%d %H%M%S")

    # get the AP
    # print(len(cxr[(cxr['ViewPosition']=='AP')]))
    cxr_AP=cxr_meta[(cxr_meta['ViewPosition']=='AP')].dropna(subset=['ViewPosition'])
    
    
    # get the latest AP for each study case
    cxr_AP_sorted=cxr_AP.sort_values(['subject_id','study_id','StudyDateTime'],ascending=True)
    cxr_latest_AP=cxr_AP_sorted.groupby(['subject_id','study_id','ViewPosition']).nth(-1).reset_index()
    
    all_stay=pd.read_csv(all_icu_stay)
    # convert time
    all_stay.intime=pd.to_datetime(all_stay.intime)
    all_stay.outtime=pd.to_datetime(all_stay.outtime)
    # remove patients who stayed for less than 48 hours
    all_stay=all_stay[all_stay.los*24>48]
    # exclude the patient with extreme los using IQR
    q1 = all_stay.los.quantile(0.25)
    q3 = all_stay.los.quantile(0.75)
    iqr = q3 - q1
    all_stay = all_stay[(all_stay.los > (q1 - 1.5 * iqr)) & (all_stay.los < (q3 + 1.5 * iqr))]
    # print(all_stay.los.describe())

    # link cxr with icu stay
    AP_merged_icustays = cxr_latest_AP.merge(all_stay, how='inner', on='subject_id')
    AP_merged_icustays = AP_merged_icustays.loc[
    (AP_merged_icustays.StudyDateTime<=AP_merged_icustays.outtime)
    &(AP_merged_icustays.StudyDateTime>=AP_merged_icustays.intime)
    ]
    # get cxr taken in the first 48 hours in icu stay or in the last 24 hours in ED
    AP_merged_icustays['CXRRelTime']=AP_merged_icustays[['StudyDateTime','intime']].apply(lambda x:(x["StudyDateTime"]-x["intime"])/one_day*24, axis=1)
    
    # get the chexpert labels
    chexpert=pd.read_csv(chexpert_label)
    
    AP_icustays_merged_chex=AP_merged_icustays.merge(chexpert,how="left",on=['subject_id','study_id'])

    # 0: uncertain, 1: negative, 2: positive 3: not mentioned
    AP_icustays_merged_chex=AP_icustays_merged_chex.fillna(3)
    AP_icustays_merged_chex[FINDINGS]=AP_icustays_merged_chex[FINDINGS].replace(1,2).replace(0,1).replace(-1,0)
    # replace chexpert labels to [0,1]: 0: not known, uncertain, negative  1:positive
    AP_icustays_merged_chex[FINDINGS]=AP_icustays_merged_chex[FINDINGS].replace(1,0).replace(3,0).replace(2,1)
    for col in FINDINGS:
        assert all(AP_icustays_merged_chex[col].isin([0, 1])), f"Column '{col}' should only contain 0 and 1."

    # check whether having bbox in ImaGenome
    # 0a0a1d17-5fb4ffcc-a8ca1541-8c44d583-fd9c6388_SceneGraph.json
    ImaGenome_cxrs=[i.split('_')[0] for i in os.listdir(ImaGenome) if i.endswith('_SceneGraph.json')]

    # print('original cxrs condidered.')
    # print(len(AP_icustays_merged_chex))

    AP_icustays_merged_chex=AP_icustays_merged_chex[AP_icustays_merged_chex['dicom_id'].isin(ImaGenome_cxrs)]
    # print('after exclude the cxrs without bbox')
    # print(len(AP_icustays_merged_chex))
    
    # each icu stay at least has two cxrs
    AP_icustays_merged_chex=AP_icustays_merged_chex.groupby(['subject_id','stay_id']).filter(lambda x: len(x)>=2)
    # print('after exclude the stays without two cxrs')
    # print(len(AP_icustays_merged_chex))

    # get the number of cxrs for each stay
    AP_icustays_merged_chex['num_cxrs']=AP_icustays_merged_chex.groupby(['subject_id','stay_id'])['dicom_id'].transform('count')
    # get the distribution of the number of cxrs
    stay_num_cxrs=AP_icustays_merged_chex.groupby(['subject_id','stay_id'])['num_cxrs'].first()
    # print(stay_num_cxrs.describe())

 
    
    # exclude the stays with more than 14 cxrs
    AP_icustays_merged_chex=AP_icustays_merged_chex[AP_icustays_merged_chex['num_cxrs']<=14]
    # print('after exclude the stays with more than 14 cxrs')
    stay_num_cxrs=AP_icustays_merged_chex.groupby(['subject_id','stay_id'])['num_cxrs'].first()
    # print(stay_num_cxrs.describe())

    AP_icustays_merged_chex=AP_icustays_merged_chex.sort_values(['subject_id','stay_id','CXRRelTime'],ascending=True)
    AP_icustays_merged_chex['neighbor_time_diff'] = AP_icustays_merged_chex.groupby(['subject_id','stay_id'])['CXRRelTime'].diff()
    AP_icustays_merged_chex['neighbor_time_diff']=AP_icustays_merged_chex['neighbor_time_diff'].fillna(12)
    
    AP_icustays_merged_chex=AP_icustays_merged_chex[(AP_icustays_merged_chex['neighbor_time_diff']>=8)]
    # get icu stays with neighbor time diff >=96
    exclude_stays=AP_icustays_merged_chex[(AP_icustays_merged_chex['neighbor_time_diff']>=96)].stay_id.unique()
    AP_icustays_merged_chex=AP_icustays_merged_chex[~AP_icustays_merged_chex['stay_id'].isin(exclude_stays)]
    # print('after exclude the stays with neighbor time diff >=96')


    # get the necessary columns
    AP_icustays_merged_chex=AP_icustays_merged_chex[['subject_id','stay_id','study_id','dicom_id','CXRRelTime','num_cxrs']+FINDINGS]
    AP_icustays_merged_chex.to_csv(os.path.join(output_dir,'cxr_stay.csv'),index=False)
    # print('cxr_stay.csv has been saved in ',os.path.join(output_dir,'cxr_stay.csv'))

    AP_icustays_merged_chex=AP_icustays_merged_chex[AP_icustays_merged_chex['num_cxrs']>3]
    unique_stays=len(AP_icustays_merged_chex.stay_id.unique())
    # print('unique stays:',unique_stays)
    # print('cxrs:',len(AP_icustays_merged_chex))
    # print('smaples',len(AP_icustays_merged_chex)-unique_stays*2)
    
    

    # get CXRRelTime<48 and cxr_num>1
    AP_icustays_merged_chex=AP_icustays_merged_chex[(AP_icustays_merged_chex['CXRRelTime']<48)&(AP_icustays_merged_chex['num_cxrs']>1)]
    # print(len(AP_icustays_merged_chex.stay_id.unique()))  # 3763



def get_bbox_for_cxrs(cxr_stay_path, scene_graph_json_dir, output_dir):
    
    cxr_stay = pd.read_csv(cxr_stay_path)
    cxr_dicom_ids=cxr_stay.dicom_id.unique()

    cxr_region_bbox_dict={'cxr_id':[],**{k:[] for k in REGIONS}}
    region_missing={k:0 for k in REGIONS}
    for cxr_id in tqdm(cxr_dicom_ids,total=len(cxr_dicom_ids),desc='get bbox for cxrs'):
    
        cxr_region_bbox_dict['cxr_id'].append(cxr_id)
        scene_graph_json_path=os.path.join(scene_graph_json_dir, cxr_id + '_SceneGraph.json')
        if os.path.exists(scene_graph_json_path):
            scene_graph_json=json.load(open(scene_graph_json_path))
        else:
            # print(f'no scene graph for {cxr_id}')
            continue
        objects=scene_graph_json['objects']
        objects_bbox_name=[i['bbox_name'] for i in objects ]
        

        for obj in objects:
            if obj['bbox_name'] in REGIONS:
                cxr_region_bbox_dict[obj['bbox_name']].append([obj['original_x1'], obj['original_y1'], obj['original_x2'], obj['original_y2']])
        # check if there is any organ missing
        for organ in REGIONS:
            if organ not in objects_bbox_name:
                # print(f'no {organ} for {cxr_id}')
                # add [0,0,0,0]
                cxr_region_bbox_dict[organ].append([0,0,0,0])
                region_missing[organ]+=1
    # print(region_missing)
    # print(len(cxr_dicom_ids))
    # breakpoint()
    # lth={n:len(k) for n,k in cxr_region_bbox_dict.items()}
    # save to csv
    cxr_region_bbox_df=pd.DataFrame(data=cxr_region_bbox_dict)
    cxr_region_bbox_df.to_csv(os.path.join(output_dir, 'cxr_bbox.csv'), index=False)
    


def correct_scenegraph(scene_graph_json_path):
   
    # 3e953c67-28b78460-4f783437-379841d3-a79ec530_SceneGraph.json
    json_3e953c67=json.load(open(os.path.join(scene_graph_json_path, '3e953c67-28b78460-4f783437-379841d3-a79ec530_SceneGraph.json')))
    error_object= {
            "object_id": "3e953c67-28b78460-4f783437-379841d3-a79ec530_left mid lung zone",
            "x1": 129,
            "y1": 65,
            "x2": 186,
            "y2": 89,
            "width": 57,
            "height": 24,
            "bbox_name": "left mid lung zone",
            "synsets": [
                "CL380306"
            ],
            "name": "Left mid lung zone",
            "original_x1": 1500,
            "original_y1": 886,
            "original_x2": 2278,
            "original_y2": 1214,
            "original_width": 778,
            "original_height": 328
        }
    # delete the error object
    json_3e953c67['objects'].remove(error_object)
    filename='3e953c67-28b78460-4f783437-379841d3-a79ec530_SceneGraph.json'
    json.dump(json_3e953c67, open(os.path.join(scene_graph_json_path, filename), 'w'))

    #8bd72ea4-f6e5470f-3ddbc81e-a1da43d8-b6e39876_SceneGraph.json
    json_8bd72ea4=json.load(open(os.path.join(scene_graph_json_path, '8bd72ea4-f6e5470f-3ddbc81e-a1da43d8-b6e39876_SceneGraph.json')))
    error_object= {
            "object_id": "8bd72ea4-f6e5470f-3ddbc81e-a1da43d8-b6e39876_left lower lung zone",
            "x1": 76,
            "y1": 43,
            "x2": 107,
            "y2": 103,
            "width": 31,
            "height": 60,
            "bbox_name": "left lower lung zone",
            "synsets": [
                "C0934571"
            ],
            "name": "Left lower lung zone",
            "original_x1": 777,
            "original_y1": 586,
            "original_x2": 1199,
            "original_y2": 1404,
            "original_width": 422,
            "original_height": 818
    }
    # delete the error object
    json_8bd72ea4['objects'].remove(error_object)
    filename='8bd72ea4-f6e5470f-3ddbc81e-a1da43d8-b6e39876_SceneGraph.json'
    json.dump(json_8bd72ea4, open(os.path.join(scene_graph_json_path, filename), 'w'))
    
    # 38bbebbd-b67908dc-387567bf-e3f965b7-77a18536_SceneGraph.json
    json_38bbebbd=json.load(open(os.path.join(scene_graph_json_path, '38bbebbd-b67908dc-387567bf-e3f965b7-77a18536_SceneGraph.json')))
    error_object= {
            "object_id": "38bbebbd-b67908dc-387567bf-e3f965b7-77a18536_right apical zone",
            "x1": 103,
            "y1": 39,
            "x2": 150,
            "y2": 63,
            "width": 47,
            "height": 24,
            "bbox_name": "right apical zone",
            "synsets": [
                "C0929167"
            ],
            "name": "Apical zone of right lung",
            "original_x1": 1146,
            "original_y1": 532,
            "original_x2": 1787,
            "original_y2": 859,
            "original_width": 641,
            "original_height": 327
        }
    
    #  right object
    right_object={
            "object_id": "38bbebbd-b67908dc-387567bf-e3f965b7-77a18536_right apical zone",
            "x1": 103,
            "y1": 39,
            "x2": 150,
            "y2": 63,
            "width": 47,
            "height": 24,
            "bbox_name": "left apical zone",
            "synsets": [
                "C0929167"
            ],
            "name": "Apical zone of left lung",
            "original_x1": 1146,
            "original_y1": 532,
            "original_x2": 1787,
            "original_y2": 859,
            "original_width": 641,
            "original_height": 327
        }
    # replace with the right object
    json_38bbebbd['objects'][json_38bbebbd['objects'].index(error_object)]=right_object

    error_object_2= {
            "object_id": "38bbebbd-b67908dc-387567bf-e3f965b7-77a18536_right upper lung zone",
            "x1": 104,
            "y1": 38,
            "x2": 163,
            "y2": 84,
            "width": 59,
            "height": 46,
            "bbox_name": "right upper lung zone",
            "synsets": [
                "C0934570"
            ],
            "name": "Right upper lung zone",
            "original_x1": 1159,
            "original_y1": 518,
            "original_x2": 1964,
            "original_y2": 1146,
            "original_width": 805,
            "original_height": 628
        }
    right_object_2={
            "object_id": "38bbebbd-b67908dc-387567bf-e3f965b7-77a18536_left upper lung zone",
            "x1": 104,
            "y1": 38,
            "x2": 163,
            "y2": 84,
            "width": 59,
            "height": 46,
            "bbox_name": "left upper lung zone",
            "synsets": [
                "C0934570"
            ],
            "name": "Left upper lung zone",
            "original_x1": 1159,
            "original_y1": 518,
            "original_x2": 1964,
            "original_y2": 1146,
            "original_width": 805,
            "original_height": 628
        }

    json_38bbebbd['objects'][json_38bbebbd['objects'].index(error_object_2)]=right_object_2
    filename='38bbebbd-b67908dc-387567bf-e3f965b7-77a18536_SceneGraph.json'
    json.dump(json_38bbebbd, open(os.path.join(scene_graph_json_path, filename), 'w'))
    

    # 038c0310-98178aef-3414521f-1d34c309-8b94b78e_SceneGraph.json
    json_038c0310=json.load(open(os.path.join(scene_graph_json_path, '038c0310-98178aef-3414521f-1d34c309-8b94b78e_SceneGraph.json')))
    error_object={
            "object_id": "038c0310-98178aef-3414521f-1d34c309-8b94b78e_left lower lung zone",
            "x1": 47,
            "y1": 146,
            "x2": 127,
            "y2": 197,
            "width": 80,
            "height": 51,
            "bbox_name": "left lower lung zone",
            "synsets": [
                "C0934571"
            ],
            "name": "Left lower lung zone",
            "original_x1": 641,
            "original_y1": 1732,
            "original_x2": 1732,
            "original_y2": 2428,
            "original_width": 1091,
            "original_height": 696
        }
    right_object={
            "object_id": "038c0310-98178aef-3414521f-1d34c309-8b94b78e_right lower lung zone",
            "x1": 47,
            "y1": 146,
            "x2": 127,
            "y2": 197,
            "width": 80,
            "height": 51,
            "bbox_name": "right lower lung zone",
            "synsets": [
                "C0934571"
            ],
            "name": "Right lower lung zone",
            "original_x1": 641,
            "original_y1": 1732,
            "original_x2": 1732,
            "original_y2": 2428,
            "original_width": 1091,
            "original_height": 696
        }
    
    json_038c0310['objects'][json_038c0310['objects'].index(error_object)]=right_object

    error_object_2={
            "object_id": "038c0310-98178aef-3414521f-1d34c309-8b94b78e_left mid lung zone",
            "x1": 57,
            "y1": 115,
            "x2": 128,
            "y2": 147,
            "width": 71,
            "height": 32,
            "bbox_name": "left mid lung zone",
            "synsets": [
                "CL380306"
            ],
            "name": "Left mid lung zone",
            "original_x1": 777,
            "original_y1": 1309,
            "original_x2": 1746,
            "original_y2": 1746,
            "original_width": 969,
            "original_height": 437
        }
    right_object_2={
            "object_id": "038c0310-98178aef-3414521f-1d34c309-8b94b78e_right mid lung zone",
            "x1": 57,
            "y1": 115,
            "x2": 128,
            "y2": 147,
            "width": 71,
            "height": 32,
            "bbox_name": "right mid lung zone",
            "synsets": [
                "CL380306"
            ],
            "name": "Right mid lung zone",
            "original_x1": 777,
            "original_y1": 1309,
            "original_x2": 1746,
            "original_y2": 1746,
            "original_width": 969,
            "original_height": 437
        }
    json_038c0310['objects'][json_038c0310['objects'].index(error_object_2)]=right_object_2

    filename='038c0310-98178aef-3414521f-1d34c309-8b94b78e_SceneGraph.json'
    json.dump(json_038c0310, open(os.path.join(scene_graph_json_path, filename), 'w'))
    




if __name__=='__main__':
    align_cxr_with_stay(args.mimic_cxr_meta, args.chexpert_label, args.ImaGenome, args.all_stay_csv_path, args.output_dir)
    cxr_stay_path=os.path.join(args.output_dir,'cxr_stay.csv')
    correct_scenegraph(args.ImaGenome)
    get_bbox_for_cxrs(cxr_stay_path, args.ImaGenome, args.output_dir)
    

