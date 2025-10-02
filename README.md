# Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment

paper, NeurIPS-2025

Abstract

Method

## Set up environment


```shell
conda create -n dipro python=3.8 
conda activate dipro 
# CUDA 11.1 
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html 
pip install -r requirements.txt 
pip install taming-transformers-rom1504

```

## Data PReparation

1. set the config file

```shell
bash data_preparation.sh
```
START:
        stay_ids: 76943
        hadm_ids: 69639
        subject_ids: 53569
REMOVE ICU TRANSFERS:
        stay_ids: 71334
        hadm_ids: 65112
        subject_ids: 50764
REMOVE MULTIPLE STAYS PER ADMIT:
        stay_ids: 59818
        hadm_ids: 59818
        subject_ids: 47477
REMOVE PATIENTS AGE < 18:
        stay_ids: 59818
        hadm_ids: 59818
        subject_ids: 47477


## Experiments

```bash
bash experiments.sh
```

[] 测试模型代码客服先
[] 测试patient=10/30
[] 3个seed平均值的代码
[] 写readme
[] 开源！！！


## Citation

If you found this repository useful, please consider cite our paper:

> 

## Acknowledgements

We would like to acknowledge the following open-source projects that were used in our work:

- 
- 


/data1/liuc/NIPS_2025/files/mimiciv/Processed_EHR_DDL/10003400/New_episode34577403_timeseries.csv
/data1/liuc/DiPro/processed_data/ehr_subjects_new/10003400/New_episode34577403_timeseries.csv
  1%|▋                                                                                             | 474/59818 [00:01<03:57, 249.43it/s]/data1/liuc/NIPS_2025/files/mimiciv/Processed_EHR_DDL/10089085/New_episode36182571_timeseries.csv
/data1/liuc/DiPro/processed_data/ehr_subjects_new/10089085/New_episode36182571_timeseries.csv
  1%|█▎                                                                                            | 810/59818 [00:03<03:51, 255.40it/s]/data1/liuc/NIPS_2025/files/mimiciv/Processed_EHR_DDL/10145540/New_episode36107231_timeseries.csv
/data1/liuc/DiPro/processed_data/ehr_subjects_new/10145540/New_episode36107231_timeseries.csv
  1%|█▎                                                                                            | 862/59818 [00:03<03:52, 253.32it/s]/data1/liuc/NIPS_2025/files/mimiciv/Processed_EHR_DDL/10151556/New_episode39814818_timeseries.csv
/data1/liuc/DiPro/processed_data/ehr_subjects_new/10151556/New_episode39814818_timeseries.csv