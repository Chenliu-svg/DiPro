
## Set the following paths before running preprocessing or training scripts:

# Directory containing the MIMIC-IV structured EHR CSV files
# Please unzip the csv files in the mimic-iv
mimic_iv_csv_dir=/Directory/containing/the/MIMIC-IV/structured/EHR/data/CSV/files/ 
# e.g., ~/physionet.org/files/mimiciv/2.0

# Directory containing MIMIC-CXR JPEG images
mimic_cxr_dir=/Directory/containing/MIMIC-CXR-JPG/images/ 
# e.g., ~/physionet.org/files/mimic-cxr-jpg/2.0.0

# Directory containing Chest ImaGenome scene graph data
# Please unzip the scene_graph in the chest-imagenome silver_dataset first
chest_imagenome_scene_graph=/Directory/containing/Chest-ImaGenome/scene_graph/
# e.g., ~/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/scene_graph

# Path to Chest ImaGenome gold standard object comparison file (with coordinates)
chest_imagenome_golden_comparison=/Directory/to/Chest-ImaGenome/gold_object_comparison_with_coordinates.txt
# e.g., ~/physionet.org/files/chest-imagenome/1.0.0/gold_dataset/gold_object_comparison_with_coordinates.txt

# Directory to store all preprocessed multimodal data for training/testing
processed_data_dir=/Directory/for/saving/the/processed/data/ 
# e.g., ~/DiPro/processed_data