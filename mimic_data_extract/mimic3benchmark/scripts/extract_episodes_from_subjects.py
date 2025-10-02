from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import sys
from tqdm import tqdm

from mimic3benchmark.subject import read_stays, read_diagnoses, read_events, get_events_for_stay,\
    add_hours_elpased_to_events
from mimic3benchmark.subject import convert_events_to_timeseries, get_first_valid_from_timeseries
from mimic3benchmark.preprocessing import read_itemid_to_variable_map, read_hw_itemid_to_variable_map, map_itemids_to_variables, clean_events
from mimic3benchmark.preprocessing import assemble_episodic_data

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
parser.add_argument('--variable_map_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/itemid_to_variable_map.csv'),
                    help='CSV containing ITEMID-to-variable map.')
parser.add_argument('--reference_range_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/variable_ranges.csv'),
                    help='CSV containing reference ranges for variables.')
args, _ = parser.parse_known_args()

# var_map： 'variable', 'mimic_label'，‘ITEMID’
# var_map = var_map[(var_map.STATUS == 'ready')]
var_map = read_itemid_to_variable_map(args.variable_map_file)
variables = var_map.variable.unique()

hw_map=read_hw_itemid_to_variable_map(args.variable_map_file)
hw_variables=hw_map.variable.unique()


for subject_dir in tqdm(os.listdir(args.subjects_root_path), desc='Iterating over subjects'):
    
    dn = os.path.join(args.subjects_root_path, subject_dir)
    
    try:
        subject_id = int(subject_dir)
        if not os.path.isdir(dn):
            raise Exception
    except:
        continue

    try:
        # reading tables of this subject
        stays = read_stays(os.path.join(args.subjects_root_path, subject_dir))
        
        diagnoses = read_diagnoses(os.path.join(args.subjects_root_path, subject_dir))
        events_origin = read_events(os.path.join(args.subjects_root_path, subject_dir))

    except:
        sys.stderr.write('Error reading from disk for subject: {}\n'.format(subject_id))
        continue

    episodic_data = assemble_episodic_data(stays, diagnoses)

    # cleaning and converting to time series

    # Subject_id,hadm_id,stay_id,charttime,itemid,value,valuenum, 'variable', 'mimic_label'
    events = map_itemids_to_variables(events_origin, var_map)

    hw_events=map_itemids_to_variables(events_origin,hw_map)
    # print(hw_events.columns)
    # exit(0)

    events = clean_events(events)
    hw_events = clean_events(hw_events)

    if events.shape[0] == 0:
        # no valid events for this subject
        continue
    # else:
    #     print(f'events for subject_id {events.shape[0]}')

    timeseries = convert_events_to_timeseries(events, variables=variables)
    hw_timeseries = convert_events_to_timeseries(hw_events, variables=hw_variables)

    # extracting separate episodes
    # import pdb; pdb.set_trace()      

    for i in range(stays.shape[0]):
        stay_id = stays.stay_id.iloc[i]
        intime = stays.intime.iloc[i]
        outtime = stays.outtime.iloc[i]

        episode = get_events_for_stay(timeseries, stay_id, intime, outtime)
        hw_episode = get_events_for_stay(hw_timeseries, stay_id, intime, outtime)

        if episode.shape[0] == 0:
            # no data for this episode
            continue

        episode = add_hours_elpased_to_events(episode, intime).set_index('HOURS').sort_index(axis=0)
        hw_episode = add_hours_elpased_to_events(hw_episode, intime).set_index('HOURS').sort_index(axis=0)
        if stay_id in episodic_data.index:
            episodic_data.loc[stay_id, 'Weight'] = get_first_valid_from_timeseries(hw_episode, 'Weight')
            episodic_data.loc[stay_id, 'Height'] = get_first_valid_from_timeseries(hw_episode, 'Height')
        # episodic_data.loc[episodic_data.index == stay_id].to_csv(os.path.join(args.subjects_root_path, subject_dir,
        #                                                                       'episode{}.csv'.format(i+1)),
        #                                                          index_label='Icustay')
        episodic_data.loc[episodic_data.index == stay_id].to_csv(os.path.join(args.subjects_root_path, subject_dir,
                                                                              'New_episode{}.csv'.format(stay_id)),
                                                                 index_label='Icustay')
        
        columns = list(episode.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x == "Hours" else x))
        episode = episode[columns_sorted]
        episode.to_csv(os.path.join(args.subjects_root_path, subject_dir, 'New_episode{}_timeseries.csv'.format(stay_id)),
                       index_label='Hours')
        # episode.to_csv(os.path.join(args.subjects_root_path, subject_dir, 'episode{}_timeseries.csv'.format(i+1)),
        #                index_label='Hours')
    # print(dn)
    # exit(0)