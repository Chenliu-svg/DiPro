import argparse
import os
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from mimic3benchmark.constants import SELECTED_VAR, GCS

parser = argparse.ArgumentParser(description='Regular every episode.')
parser.add_argument('--all_stay_csv_path', type=str, required=True, help='the all_stay_csv path, containing the all the icu_stay samples needed.')
parser.add_argument('--subject_episode_dir', type=str, required=True,
                    help='the directory containing the episodes of each subject, including the timeseries and metadata of the stay.')
parser.add_argument('--output_dir', type=str,required=True, help='the output directory to store the regular episodes.')
parser.add_argument('--OUTLIER_FILE', type=str,default='./mimic3benchmark/resources/variable_ranges.csv',
                     help='the file containing the outlier information of each variable.')
parser.add_argument('--time_interval', type=int, default=1, help='the time interval to regular the episode in hour.') 
parser.add_argument('--admission_path', type=str, default=None, help='the admission csv path from mimic-iv')
parser.add_argument('--meta_save_dir', type=str, default=None, help='the directory to save the meta data')
args, _ = parser.parse_known_args()


try:
    os.makedirs(args.output_dir)
except:
    pass

outlier_file=pd.read_csv(args.OUTLIER_FILE)
outlier_file['LEVEL2']=outlier_file['LEVEL2'].str.lower()

def get_mean_std(all_stay_csv_path, subject_episode_dir, output_dir, variables_list):
    all_stay=pd.read_csv(all_stay_csv_path)
    mean_std={var : {'total_sum':0, 'total_squared_sum':0, 'total_count':0, 'total_outlier_count':0} for var in variables_list}

    for index, row in tqdm(all_stay.iterrows(), total=all_stay.shape[0], desc='Getting mean and std'):
    
        
        subject_id=row['subject_id']
        stay_id=row['stay_id']
        # /10000032/New_episode39553978_timeseries.csv
        episode_path=os.path.join(subject_episode_dir, str(subject_id), 'New_episode'+str(stay_id)+'_timeseries.csv')
        episode=pd.read_csv(episode_path)
        
    
          
        if not os.path.exists(episode_path):
            # print("episode not found", subject_id, stay_id)
            continue
        episode=pd.read_csv(episode_path)
        for var in variables_list:
            # change the GCS to the corresponding values
            if var in GCS:
                episode[var]=episode[var].map(GCS[var]['values'])
            
            # if var not in outlier_file['LEVEL2'].values:
            #     print(var)
            #     exit(0)
            # change the extreme values to upper outlier or the lower outlier
            lower_outlier=outlier_file[outlier_file['LEVEL2']==var.lower()]['OUTLIER LOW'].values[0]
            upper_outlier=outlier_file[outlier_file['LEVEL2']==var.lower()]['OUTLIER HIGH'].values[0]
            total_outlier_count=episode[(episode[var]<lower_outlier) | (episode[var]>upper_outlier)].shape[0]
            mean_std[var]['total_outlier_count']+=total_outlier_count

            episode.loc[episode[var]<lower_outlier, var]=lower_outlier
            episode.loc[episode[var]>upper_outlier, var]=upper_outlier
            
            mean_std[var]['total_sum']+=episode[var].sum()
            mean_std[var]['total_squared_sum']+=(episode[var]**2).sum()
            mean_std[var]['total_count']+=episode[var].count()
    
    for var in variables_list:
        mean_std[var]['outlier_rate']=mean_std[var]['total_outlier_count']/mean_std[var]['total_count']
        mean_std[var]['mean']=mean_std[var]['total_sum']/mean_std[var]['total_count']
        mean_std[var]['std']=((mean_std[var]['total_squared_sum']/mean_std[var]['total_count'])-mean_std[var]['mean']**2)**0.5

    # write the mean and std to the output file in csv format
    mean_std_df=pd.DataFrame(mean_std).T
    mean_std_df.to_csv(os.path.join(output_dir, 'mean_std.csv'))


def regular_episode(all_stay_csv_path, subject_episode_dir, output_dir, variables_list,mean_std,time_interval):
    all_stay=pd.read_csv(all_stay_csv_path)
    
    ###########test################
    # get the first 10 raws for test
    # all_stay=all_stay.head(2)
    ###############################
    for _, row in tqdm(all_stay.iterrows(), total=all_stay.shape[0], desc='Regularizing ehr episodes'):
        subject_id=row['subject_id']
        stay_id=row['stay_id']
        episode_path=os.path.join(subject_episode_dir, str(subject_id), 'New_episode'+str(stay_id)+'_timeseries.csv')
        if not os.path.exists(episode_path):
            continue
        episode=pd.read_csv(episode_path)[['Hours']+SELECTED_VAR]
        
        max_regular_interval=int(episode['Hours'].max()//time_interval)
        for var in variables_list:
            # change the GCS to the corresponding values
            if var in GCS:
                episode[var]=episode[var].map(GCS[var]['values'])
            
            # if var not in outlier_file['LEVEL2'].values:
            #     print(var)
            #     exit(0)
            # change the extreme values to upper outlier or the lower outlier
            lower_outlier=outlier_file[outlier_file['LEVEL2']==var.lower()]['OUTLIER LOW'].values[0]
            upper_outlier=outlier_file[outlier_file['LEVEL2']==var.lower()]['OUTLIER HIGH'].values[0]
            episode.loc[episode[var]<lower_outlier, var]=lower_outlier
            episode.loc[episode[var]>upper_outlier, var]=upper_outlier

    
        # make sure the regular interval is continuous
        interval_df=pd.DataFrame({'regular_interval':range(-1,max_regular_interval+1)})
        # episode_latest=pd.merge(interval_df, episode_latest, on='regular_interval', how='left')
        # get the mask of the missing data

        for var in variables_list:
            var_df=episode[['Hours',var]]
            var_df['regular_interval']=var_df['Hours']//time_interval
            
            var_df_soted=var_df.sort_values(['regular_interval','Hours'],ascending=True)
            # drop nan
            var_df_soted.dropna(subset=[var], inplace=True)
            var_df_latest=var_df_soted.groupby('regular_interval').nth(-1)
            var_df_latest.drop(columns='Hours', inplace=True)
            # merge with the interval_df
            var_df_latest=pd.merge(interval_df, var_df_latest, on='regular_interval', how='left')

            # mask
            var_df_latest[f'{var}_mask'] = var_df_latest[var].notna().astype(int)

            # impute the data
            normal_impute_value=outlier_file[outlier_file['LEVEL2']==var.lower()]['IMPUTE'].values[0]
            # if no previous value available, use the normal value
            if pd.isnull(var_df_latest[var].iloc[0]):
                # breakpoint()
                var_df_latest[var].iloc[0] = normal_impute_value
            # breakpoint()
            var_df_latest[var].fillna(method='ffill', inplace=True)
            # update the interval_df
            interval_df=var_df_latest
        
        # normalize the data
        for var in variables_list:
            # breakpoint()
            mean=mean_std[mean_std['variables']==var]['mean'].values[0]
            std=mean_std.loc[mean_std['variables']==var]['std'].values[0]
            interval_df[var]=(interval_df[var]-mean)/std
        
        # delete the first line
        interval_df.drop(index=0, inplace=True)

        
        ############## test ################
        # save the original data to compare
        # episode.to_csv(os.path.join(output_dir, 'Original_episode'+str(stay_id)+'_timeseriesl.csv'), index=False)
        ###################################
        # save the data
        interval_df.to_csv(os.path.join(output_dir, 'New_episode'+str(stay_id)+'_timeseries.csv'), index=False)

def get_demographic_ehr(all_stay, admission_path,episode_path,output_dir):
    all_stay=pd.read_csv(all_stay)
    admission=pd.read_csv(admission_path)
    
    all_stay=all_stay.merge(admission,on=['hadm_id','subject_id'],how='left')

    # all_stay=all_stay[:10] # for debug
    # 10000032/New_episode39553978.csv
    for _, row in tqdm(all_stay.iterrows(), total=len(all_stay), desc='Getting demographic information'):
        subject_id = row['subject_id']
        episode_id = row['stay_id']
        source_path = os.path.join(episode_path, str(subject_id), "New_episode"+str(episode_id)+".csv")
        episode=pd.read_csv(source_path)
        if len(episode)==0:
            print(f'no episode for {subject_id}, {episode_id}')
            # write the episode_id to a file
            with open(os.path.join(output_dir,'episode_not_found.txt'),'a') as f:
                f.write(f'{subject_id},{episode_id}\n')
            continue
        height=episode['Height'].values[0]
        weight=episode['Weight'].values[0]
        # add the demographic information to the all_stay
        all_stay.loc[(all_stay['subject_id']==subject_id)&(all_stay['stay_id']==episode_id),'height']=height
        all_stay.loc[(all_stay['subject_id']==subject_id)&(all_stay['stay_id']==episode_id),'weight']=weight

    demographic=all_stay[['subject_id','stay_id','age','gender','race','language','marital_status','height','weight']]
    demographic.to_csv(os.path.join(output_dir,'demographic.csv'),index=False)


def process_demographic(demographic_path, ouput_dir):
    demographic=pd.read_csv(demographic_path)
    # print(demographic.describe())
    # continuous variables do outlier detection using IQR and remove the outliers, and impute using mean, and do normalization

    cont_var=['age','height','weight']
    cont_vat_dict={k:{'mean':0,'std':0,'1Q':0,'3Q':0} for k in cont_var}
    
    for var in cont_var:
        cont_vat_dict[var]['1Q']=demographic[var].quantile(0.25)
        cont_vat_dict[var]['3Q']=demographic[var].quantile(0.75)
        lower_outlier=cont_vat_dict[var]['1Q']-1.5*(cont_vat_dict[var]['3Q']-cont_vat_dict[var]['1Q'])
        upper_outlier=cont_vat_dict[var]['3Q']+1.5*(cont_vat_dict[var]['3Q']-cont_vat_dict[var]['1Q'])
        clear_data=demographic[(demographic[var]>=lower_outlier)&(demographic[var]<=upper_outlier)][var]
        cont_vat_dict[var]['mean']=clear_data.mean()
        cont_vat_dict[var]['std']=clear_data.std()
    
    # print(cont_vat_dict)

    # impute missing values with mean
    for var in cont_var:
        demographic[var].fillna(cont_vat_dict[var]['mean'],inplace=True)

    # normalize continuous variables
    for var in cont_var:
        demographic[var]=(demographic[var]-cont_vat_dict[var]['mean'])/cont_vat_dict[var]['std']
   
    # categorical variables imput using mode and do one-hot encoding 
    ethnicity_mapping = {
    'WHITE': 'WHITE',
    'WHITE - OTHER EUROPEAN': 'WHITE',
    'WHITE - RUSSIAN': 'WHITE',
    'WHITE - EASTERN EUROPEAN': 'WHITE',
    'WHITE - BRAZILIAN': 'WHITE',
    'BLACK/AFRICAN AMERICAN': 'BLACK/AFRICAN AMERICAN',
    'BLACK/CAPE VERDEAN': 'BLACK/AFRICAN AMERICAN',
    'BLACK/CARIBBEAN ISLAND': 'BLACK/AFRICAN AMERICAN',
    'BLACK/AFRICAN': 'BLACK/AFRICAN AMERICAN',
    'HISPANIC/LATINO - PUERTO RICAN': 'HISPANIC/LATINO',
    'HISPANIC OR LATINO': 'HISPANIC/LATINO',
    'HISPANIC/LATINO - DOMINICAN': 'HISPANIC/LATINO',
    'HISPANIC/LATINO - GUATEMALAN': 'HISPANIC/LATINO',
    'HISPANIC/LATINO - SALVADORAN': 'HISPANIC/LATINO',
    'HISPANIC/LATINO - MEXICAN': 'HISPANIC/LATINO',
    'HISPANIC/LATINO - CUBAN': 'HISPANIC/LATINO',
    'HISPANIC/LATINO - COLUMBIAN': 'HISPANIC/LATINO',
    'HISPANIC/LATINO - HONDURAN': 'HISPANIC/LATINO',
    'HISPANIC/LATINO - CENTRAL AMERICAN': 'HISPANIC/LATINO',
    'ASIAN': 'ASIAN',
    'ASIAN - CHINESE': 'ASIAN',
    'ASIAN - SOUTH EAST ASIAN': 'ASIAN',
    'ASIAN - ASIAN INDIAN': 'ASIAN',
    'ASIAN - KOREAN': 'ASIAN',
    'OTHER': 'OTHER',
    'AMERICAN INDIAN/ALASKA NATIVE': 'OTHER',
    'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'OTHER',
    'SOUTH AMERICAN': 'OTHER',
    'MULTIPLE RACE/ETHNICITY': 'OTHER',
    'UNKNOWN': 'UNKNOWN/UNABLE TO OBTAIN/PATIENT DECLINED TO ANSWER',
    'UNABLE TO OBTAIN': 'UNKNOWN/UNABLE TO OBTAIN/PATIENT DECLINED TO ANSWER',
    'PATIENT DECLINED TO ANSWER': 'UNKNOWN/UNABLE TO OBTAIN/PATIENT DECLINED TO ANSWER'
    }

    #map fine-grained ethnicity to coarse-grained
    demographic['race_group'] = demographic['race'].map(ethnicity_mapping)
    # drop race
    demographic.drop('race',axis=1,inplace=True)

    cat_var=['gender','race_group','language','marital_status']
    cat_var_dict={k:{'mode':'unknown'} for k in cat_var}
    # get the mode for each categorical variable
    for var in cat_var:
        cat_var_dict[var]['mode']=demographic[var].mode()[0]

    # print(cat_var_dict)
    # imput and encode categorical variables
    # imput using mode
    for var in cat_var:
        demographic[var].fillna(cat_var_dict[var]['mode'],inplace=True)

    # do one-hot encoding
    for var in cat_var:
        dummy=pd.get_dummies(demographic[var],prefix=var)
        # turn bool into int
        dummy=dummy.astype(int)
        demographic=pd.concat([demographic,dummy],axis=1)
        del demographic[var] 
       
    # save the processed data
    save_path=os.path.join(ouput_dir,'demographic_processed.csv')
    demographic.to_csv(save_path,index=None)

    
      
            
if __name__=='__main__':
    get_mean_std(args.all_stay_csv_path, args.subject_episode_dir, args.meta_save_dir, SELECTED_VAR)

    mean_std_path=os.path.join(args.meta_save_dir,'mean_std.csv')
    # ,total_sum,total_squared_sum,total_count,total_outlier_count,outlier_rate,mean,std

    # rename the first column to variables
    mean_std=pd.read_csv(mean_std_path)
    mean_std=mean_std.rename(columns={'Unnamed: 0': 'variables'})
    mean_std.to_csv(mean_std_path,index=None)

    regular_episode(args.all_stay_csv_path, args.subject_episode_dir, args.output_dir, SELECTED_VAR, mean_std, args.time_interval)

    # get demographic
    get_demographic_ehr(args.all_stay_csv_path, args.admission_path, args.subject_episode_dir, args.meta_save_dir)
    demographic_path=os.path.join(args.meta_save_dir,'demographic.csv')
    process_demographic(demographic_path, args.meta_save_dir)
    