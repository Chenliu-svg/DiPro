source scripts/data_source.sh

cd ./mimic_data_extract

echo "MIMIC-IV CSV: $mimic_iv_csv_dir"
echo "MIMIC-CXR: $mimic_cxr_dir"
echo "Processed: $processed_data_dir"
echo "Scene Graph: $chest_imagenome_scene_graph"
echo "Golden Comparison: $chest_imagenome_golden_comparison"

mimic_cxr_meta=$mimic_cxr_dir/mimic-cxr-2.0.0-metadata.csv
mimic_cxr_chexpert=$mimic_cxr_dir/mimic-cxr-2.0.0-chexpert.csv

mimic_iv_subjects_dir=$processed_data_dir/original_ehr 
ehr_preprocessed_dir=$processed_data_dir/ehr_preprocessed

admission_path=$mimic_iv_csv_dir/hosp/admissions.csv

echo "extracting ehr..."
python -m mimic3benchmark.scripts.extract_subjects_iv \
    $mimic_iv_csv_dir \
    $mimic_iv_subjects_dir

python -m mimic3benchmark.scripts.extract_episodes_from_subjects $mimic_iv_subjects_dir

echo "preprocessing ehr..."
python -m mimic3benchmark.scripts.ehr_preprocess \
    --all_stay_csv_path  $mimic_iv_subjects_dir/all_stays.csv \
    --subject_episode_dir $mimic_iv_subjects_dir \
    --output_dir $ehr_preprocessed_dir \
    --meta_save_dir $processed_data_dir \
    --admission_path $admission_path


# # get CXRs and anatomical bbox
echo "get CXRs and anatomical bbox"
python -m mimic3benchmark.scripts.get_cxr \
    --all_stay_csv_path   $mimic_iv_subjects_dir/all_stays.csv \
    --mimic_cxr_meta $mimic_cxr_meta \
    --chexpert_label $mimic_cxr_chexpert \
    --ImaGenome $chest_imagenome_scene_graph \
    --output_dir $processed_data_dir


# # # Process Chest-ImaGenome
echo "Processing Chest-ImaGenome"
python -m mimic3benchmark.scripts.process_ImaGenome \
    --scene_graph $chest_imagenome_scene_graph \
    --save_path $processed_data_dir \
    --golden_dir $chest_imagenome_golden_comparison \
    --cxr_stay $processed_data_dir/cxr_stay.csv


echo "splitting the dataset...."
python -m mimic3benchmark.scripts.split_train_and_test \
    --progression_task_label_dir $processed_data_dir/disease_progressions \
    --golden_dir  $chest_imagenome_golden_comparison \
    --all_stay_csv_path $mimic_iv_subjects_dir/all_stays.csv \
    --cxr_stay  $processed_data_dir/cxr_stay.csv \
    --split_save_dir   $processed_data_dir/split \
    --all_comparison_path $processed_data_dir/all_comparison.csv
    