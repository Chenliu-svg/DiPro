# Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment

paper, NeurIPS-2025 Spotlight

## Abstract

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

## Experiments

```bash
bash experiments.sh
```


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