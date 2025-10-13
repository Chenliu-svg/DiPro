source scripts/config.sh

RUN_NAME="test_run"
task_name='length_of_stay'
LOGDIR=./logs/$RUN_NAME/$task_name

gpu=0,
batch_size=4

# specify data paths
label_path=$processed_data_dir/split
demographic_path=$processed_data_dir/demographic_processed.csv
ehr_time_series_path=$processed_data_dir/ehr_preprocessed
bbox_csv=$processed_data_dir/cxr_bbox.csv
# mimic_cxr_path=$mimic_cxr_dir/files
mimic_cxr_path=$mimic_cxr_dir/
cxr_meta_path=$mimic_cxr_dir/mimic-cxr-2.0.0-metadata.csv


lambda_consistency=0.1
lambda_CE_loss=10.0 
lambda_static_order=0.5
lambda_dynamic_order=0.1
lambda_orthogonal=0.001



base_learning_rate=8.0e-06
d_model=128
dropout_rate=0.1
group_num=8
num_transformer_layers=2

for  seed in 23 29 66; do
    python main.py \
        --base configs/DiPro.yaml \
        -t \
        --seed $seed \
        --gpu $gpu\
        --name $seed \
        --logdir $LOGDIR \
        model.params.task_name=$task_name \
        model.base_learning_rate=$base_learning_rate \
        model.params.d_model=$d_model \
        model.params.dropout_rate=$dropout_rate \
        model.params.group_num=$group_num \
        model.params.lambda_CE_loss=$lambda_CE_loss \
        model.params.lambda_consistency=$lambda_consistency \
        model.params.lambda_orthogonal=$lambda_orthogonal \
        model.params.lambda_static_order=$lambda_static_order \
        model.params.lambda_dynamic_order=$lambda_dynamic_order \
        model.params.num_transformer_layers=$num_transformer_layers \
        data.params.train.params.task_name=$task_name \
        data.params.validation.params.task_name=$task_name \
        data.params.test.params.task_name=$task_name \
        data.params.batch_size=$batch_size \
        data.params.train.params.demographic_path=$demographic_path \
        data.params.validation.params.demographic_path=$demographic_path \
        data.params.test.params.demographic_path=$demographic_path \
        data.params.train.params.ehr_time_series_path=$ehr_time_series_path \
        data.params.validation.params.ehr_time_series_path=$ehr_time_series_path \
        data.params.test.params.ehr_time_series_path=$ehr_time_series_path \
        data.params.train.params.bbox_csv=$bbox_csv \
        data.params.validation.params.bbox_csv=$bbox_csv \
        data.params.test.params.bbox_csv=$bbox_csv \
        data.params.train.params.mimic_cxr_path=$mimic_cxr_path \
        data.params.validation.params.mimic_cxr_path=$mimic_cxr_path \
        data.params.test.params.mimic_cxr_path=$mimic_cxr_path \
        data.params.train.params.cxr_meta_path=$cxr_meta_path \
        data.params.validation.params.cxr_meta_path=$cxr_meta_path \
        data.params.test.params.cxr_meta_path=$cxr_meta_path \
        model.params.label_path=$label_path \
        data.params.train.params.label_path=$label_path \
        data.params.validation.params.label_path=$label_path \
        data.params.test.params.label_path=$label_path \
        lightning.callbacks.EarlyStopping.params.patience=10 \
        lightning.trainer.accumulate_grad_batches=4

done
