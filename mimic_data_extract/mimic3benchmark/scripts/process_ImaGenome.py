"""
Get all paired cxr disease progression labels
"""
import argparse
import os
import sys
from tqdm import tqdm
import pandas as pd
import warnings
from datetime import timedelta
import json
import ast
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='get disease progression labels.')
parser.add_argument('--scene_graph', type=str, required=True, help='the all_stay_csv path, containing the all the icu_stay samples needed.')
parser.add_argument('--golden_dir', type=str, required=True, help='the path of the golden labels')
parser.add_argument('--save_path', type=str, help='the path to save files')
parser.add_argument('--cxr_stay', type=str, required=True,
                    help='aligned cxr and stay csv path')
parser.add_argument('--order_map_path', type=str, default=os.path.join(os.path.dirname(__file__),'../resources/all_comparison_order_map.csv'), help='the path of the all comparison order mapping for reproducebility')

args, _ = parser.parse_known_args()

# read json if path exists
def readJSON(filepath):
    try:
        with open(filepath) as f:
            data = json.load(f)
            return data
    except Exception as e:
        print('File does not exist',filepath)
        return None

def get_disease_progression_labels(scene_graph,save_path):
    # /silver_dataset/scene_graph/scene_graph/0a0a1d17-5fb4ffcc-a8ca1541-8c44d583-fd9c6388_SceneGraph.json
    tab_dict={'subject_id':[],'study_id':[],'cxr1_dicom_id':[],'cxr2_dicom_id':[],'disease_progression':[]}
    graph_files =[i for i in os.listdir(scene_graph) if i.endswith('_SceneGraph.json')]
    # graph_files=graph_files[:20] 
    for graph in tqdm(graph_files,total=len(graph_files),desc='get disease progression labels from scene graph'):
        data=readJSON(os.path.join(scene_graph,graph))
        # breakpoint()
        comparisons = data['relationships']
        if len(comparisons)==0:
            continue
        tab_dict['subject_id'].append(data['patient_id'])
        tab_dict['study_id'].append(data['study_id']) 

        # The 'relationship_id' uniquely identifies each comparison relationship between the object ('subject_id') on the current exam and the object ('object_id' for the same anatomical location) from the previous exam. 
        tab_dict['cxr2_dicom_id'].append(comparisons[0]['subject_id'].split('_')[0])
        tab_dict['cxr1_dicom_id'].append(comparisons[0]['object_id'].split('_')[0])
        disease_progression=[]
        for comp in comparisons:
            compare = comp['relationship_names']
            compare = [x for x in compare if x in ['comparison|yes|worsened'
                                                   ,'comparison|yes|improved','comparison|yes|no change']]
            # compare = ';;'.join(sorted([x.split('|')[2] for x in compare]))
            compare =[x.split('|')[2] for x in compare]
            if len(compare)>0:
                # get turple (bbox_name,[attributes],[compare])
                turple = [comp['bbox_name'],[i.split("|")[-1] for i in comp['attributes'] if i.split('|')[0] in ['anatomicalfinding','disease']],compare]
                disease_progression.append(turple)
            
        tab_dict['disease_progression'].append(disease_progression)
    # breakpoint()
    tab=pd.DataFrame(tab_dict)
    tab.to_csv(os.path.join(save_path,'all_comparison.csv'),index=False)

def get_disease_and_progression(row):
    disease_progression = row['disease_progression']
    # remove progressions with ambiguous labels
    
    disease_progression = [i for i in disease_progression if len(i[2])==1]
    # get disease and their progression labels
    disease_dict={}
    for organ in disease_progression:
        for disease in organ[1]:
            if disease not in disease_dict:
                disease_dict[disease]=[]
            disease_dict[disease].append(organ[2][0])
    
    # remove diseases with ambiguous progression labels
    disease_dict = {k:v[0] for k,v in disease_dict.items() if len(set(v))==1} 
    
    # # add subject_id and study_id and cxr1_dicom_id and cxr2_dicom_id
    # disease_dict['subject_id']=row['subject_id']
    # disease_dict['study_id']=row['study_id']
    # disease_dict['cxr1_dicom_id']=row['cxr1_dicom_id']
    # disease_dict['cxr2_dicom_id']=row['cxr2_dicom_id']
    return disease_dict

def get_neighbor_cxr_labels(all_comparison,save_path):
    
    neighbor_progression_dict={'subject_id':[],'study_id':[],'cxr1_dicom_id':[],'cxr2_dicom_id':[],'disease':[],'progression':[]}
    all_comparison=pd.read_csv(all_comparison)
    # turn the string to list
    all_comparison['disease_progression']=all_comparison['disease_progression'].apply(lambda x: ast.literal_eval(x))
    
    # get disease and their progression labels
    all_comparison['disease_dict'] = all_comparison.apply(get_disease_and_progression,axis=1)
    
    # drop the rows with empty disease_dict
    all_comparison = all_comparison[all_comparison['disease_dict'].apply(lambda x: len(x)>0)]
    
    # get the disease and progression labels
    for _,row in tqdm(all_comparison.iterrows(),total=len(all_comparison),desc='get progression labels for CXR pairs'):
        disease_dict = row['disease_dict']
        for disease,progression in disease_dict.items():
            neighbor_progression_dict['subject_id'].append(row['subject_id'])
            neighbor_progression_dict['study_id'].append(row['study_id'])
            neighbor_progression_dict['cxr1_dicom_id'].append(row['cxr1_dicom_id'])
            neighbor_progression_dict['cxr2_dicom_id'].append(row['cxr2_dicom_id'])
            neighbor_progression_dict['disease'].append(disease)
            neighbor_progression_dict['progression'].append(progression)
 
    neighbor_progression_df=pd.DataFrame(neighbor_progression_dict)
    neighbor_progression_df.to_csv(os.path.join(save_path,'neighbor_progression.csv'),index=False)

def correct_using_golden_labels(golden_dir,neighbor_progression):
    # correct and also using the golden labels to evaluate
    # get [patient_id,current_image_id,previous_image_id,label_name,comparison] from the golden labels
    golden_labels=pd.read_csv(golden_dir, sep='\t')[['patient_id','current_image_id','previous_image_id','label_name','comparison']]
    # rename to ['subject_id','cxr2_dicom_id','cxr1_dicom_id','disease','progression']
    golden_labels.columns=['subject_id','cxr2_dicom_id','cxr1_dicom_id','disease','progression']
    # drop dupilcates
    golden_labels=golden_labels.drop_duplicates()
    golden_cxr_pairs = set(golden_labels[['cxr2_dicom_id', 'cxr1_dicom_id']].apply(tuple, axis=1))

    neighbor_progression_origin=pd.read_csv(neighbor_progression)
    neighbor_progression=neighbor_progression_origin[neighbor_progression_origin[['cxr2_dicom_id','cxr1_dicom_id']].apply(lambda x: tuple(x) in golden_cxr_pairs,axis=1)]
    # merge the golden_labels with neighbor_progression
    neighbor_progression_with_golden=neighbor_progression.merge(golden_labels,on=['subject_id','cxr1_dicom_id','cxr2_dicom_id','disease'],how='outer',suffixes=('_neighbor', '_golden'))
    # fill na with 'empty'
    neighbor_progression_with_golden=neighbor_progression_with_golden.fillna('empty')

    
    # correct the neighbor_progression using golden_labels
    neighbor_progression_origin['corrected_progression']=neighbor_progression_origin.apply(lambda x: _get_golden_labels(x,golden_labels,golden_cxr_pairs),axis=1)
    # drop empty in the corrected_progression
    neighbor_progression_origin=neighbor_progression_origin[neighbor_progression_origin['corrected_progression']!='empty']
    # mark rows from golden_labels
    neighbor_progression_origin['from_golden']=neighbor_progression_origin[['cxr2_dicom_id','cxr1_dicom_id']].apply(lambda x: 1 if tuple(x) in golden_cxr_pairs else 0,axis=1)
    assert 'empty' not in neighbor_progression_origin['corrected_progression'].values
    neighbor_progression_origin.to_csv(os.path.join(args.save_path,'golden_corrected_neighbor_progression.csv'),index=False)


def _get_golden_labels(row,golden_df,golden_cxr_pairs):
    cxr2_dicom_id = row['cxr2_dicom_id']
    cxr1_dicom_id = row['cxr1_dicom_id']
    if (cxr2_dicom_id, cxr1_dicom_id) not in golden_cxr_pairs:
        return row['progression']
    # get the golden labels
    golden_labels=golden_df[(golden_df['cxr2_dicom_id']==cxr2_dicom_id)&(golden_df['cxr1_dicom_id']==cxr1_dicom_id)]
    if len(golden_labels)==0:
        return 'empty'
    else:
        return golden_labels['progression'].values[0]


def pair_progression_statistic(neighbor_progression,cxr_stay,save_dir):
    neighbor_progression=pd.read_csv(neighbor_progression)

    # disease, progression, count
    # print(neighbor_progression.groupby(['disease','progression']).size().reset_index(name='count'))
    # # disease, count
    # print(neighbor_progression.groupby(['disease']).size().reset_index(name='count'))

    cxr_stay=pd.read_csv(cxr_stay)
    cxr_stay_sorted = cxr_stay.sort_values(by=['subject_id', 'stay_id', 'CXRRelTime'])

    image_pairs = []

    for (subject_id, stay_id), group in cxr_stay_sorted.groupby(['subject_id', 'stay_id']):
        dicom_ids = group['dicom_id'].tolist()
        
        # generate image pairs
        for i in range(len(dicom_ids) - 1):
            image_pairs.append({
                'subject_id': subject_id,
                'stay_id': stay_id,
                'cxr1_dicom_id': dicom_ids[i],
                'cxr2_dicom_id': dicom_ids[i + 1],
                'cxr1_time': group.iloc[i]['CXRRelTime'],
                'cxr2_time': group.iloc[i + 1]['CXRRelTime']
            })  
    image_pairs_in_stay = pd.DataFrame(image_pairs)
    # neighbor pairs should be in the image_pairs
    image_pairs_in_stay_dicom_id = set(image_pairs_in_stay[['cxr2_dicom_id', 'cxr1_dicom_id']].apply(tuple, axis=1))
    neighbor_progression=neighbor_progression[neighbor_progression[['cxr2_dicom_id','cxr1_dicom_id']].apply(lambda x: tuple(x) in image_pairs_in_stay_dicom_id,axis=1)]
    
    # # disease, progression, count and print to file
    # print(neighbor_progression.groupby(['disease','progression']).size().reset_index(name='count'),file=open(os.path.join(args.save_path,'pair_progression_statistic.txt'),'w'))
    # # disease, count and sort by count
    # print(neighbor_progression.groupby(['disease']).size().reset_index(name='count').sort_values(by='count',ascending=False),file=open(os.path.join(args.save_path,'pair_progression_statistic.txt'),'a'))

    


    # get image pairs and progression for each disease
    disease_list=['lung opacity','pleural effusion','atelectasis','enlarged cardiac silhouette','pulmonary edema/hazy opacity','consolidation','pneumonia']
    # merge the neighbor_progression with image_pairs_in_stay
    neighbor_progression=neighbor_progression.merge(image_pairs_in_stay,on=['subject_id','cxr1_dicom_id','cxr2_dicom_id'],how='left')
    save_path=os.path.join(save_dir,'disease_progressions')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for disease in disease_list:
        # get the image pairs for each disease
        disease_image_pairs=neighbor_progression[neighbor_progression['disease']==disease]
        disease_image_pairs.to_csv(os.path.join(save_path,f'{disease.split("/")[0]}_image_pairs.csv'),index=False)

#  for reproducability (To get the same splits as the paper)
def apply_order_map(new_csv, order_map_path, keys=['subject_id','study_id','cxr1_dicom_id','cxr2_dicom_id']):
    new_df = pd.read_csv(new_csv)
    order_map = pd.read_csv(order_map_path)
    merged = new_df.merge(order_map, on=keys, how='left')
    merged = merged.sort_values(by='order_index').drop(columns=['order_index'])
    merged.to_csv(new_csv, index=False)


if __name__ == '__main__':
    get_disease_progression_labels(args.scene_graph,args.save_path)
    all_comparison_path=os.path.join(args.save_path,'all_comparison.csv')
    apply_order_map(all_comparison_path, args.order_map_path)
    get_neighbor_cxr_labels(all_comparison_path,args.save_path)
    correct_using_golden_labels(args.golden_dir,os.path.join(args.save_path,'neighbor_progression.csv'))
    pair_progression_statistic(os.path.join(args.save_path,'golden_corrected_neighbor_progression.csv'),args.cxr_stay,args.save_path)
    
    
    
