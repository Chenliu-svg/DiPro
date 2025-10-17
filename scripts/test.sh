source scripts/config.sh

task_name=$1
Run_name=$3
LOGDIR=./logs/$Run_name/$task_name


gpu=$4, # Please do not delete the comma
batch_size=4

# specify data paths
label_path=$processed_data_dir/split
demographic_path=$processed_data_dir/demographic_processed.csv
ehr_time_series_path=$processed_data_dir/ehr_preprocessed
bbox_csv=$processed_data_dir/cxr_bbox.csv
mimic_cxr_path=$mimic_cxr_dir/files
cxr_meta_path=$mimic_cxr_dir/mimic-cxr-2.0.0-metadata.csv

# set model-related hyperparameters based on task_name 
case $task_name in
  "length_of_stay")
    d_model=128
    num_transformer_layers=2
    ;;
  
  "mortality")
    d_model=256
    num_transformer_layers=4
    ;;
  
  "disease_progression")
    d_model=256
    num_transformer_layers=2
    ;;
  
  *)
    echo "Error: Unknown task name '$task_name'"
    exit 1
    ;;
esac


for  seed in 23 29 66; do
    ckpt_path=$2/$task_name/seed_$seed.ckpt
    python main.py \
        --base configs/DiPro.yaml \
        -t False \
        --seed $seed \
        --gpu $gpu\
        --name seed_$seed \
        --logdir $LOGDIR \
        model.params.d_model=$d_model \
        model.params.ckpt_path=$ckpt_path \
        model.params.task_name=$task_name \
        model.params.label_path=$label_path \
        model.params.num_transformer_layers=$num_transformer_layers \
        data.params.test_mode=True \
        data.params.batch_size=$batch_size \
        data.params.test.params.task_name=$task_name \
        data.params.test.params.demographic_path=$demographic_path \
        data.params.test.params.ehr_time_series_path=$ehr_time_series_path \
        data.params.test.params.bbox_csv=$bbox_csv \
        data.params.test.params.mimic_cxr_path=$mimic_cxr_path \
        data.params.test.params.cxr_meta_path=$cxr_meta_path \
        data.params.test.params.label_path=$label_path
done

python get_results.py \
    --logdir $LOGDIR \
    --task_name $task_name

