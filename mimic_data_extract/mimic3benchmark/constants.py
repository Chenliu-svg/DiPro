# Chexpert labels
FINDINGS=['Atelectasis',
 'Cardiomegaly',
 'Consolidation',
 'Edema',
 'Enlarged Cardiomediastinum',
 'Lung Lesion',
 'Lung Opacity',
 'Pleural Effusion',
 'Pleural Other',
 'Pneumonia',
 'Pneumothorax']

# Anatomical regions in Chest ImaGenome
REGIONS=['aortic arch', 'cardiac silhouette', 'left apical zone',
       'left costophrenic angle', 'left hilar structures',
       'left lower lung zone', 'left lung', 'left mid lung zone',
       'left upper lung zone', 'mediastinum', 'right apical zone',
       'right costophrenic angle', 'right hilar structures',
       'right lower lung zone', 'right lung', 'right mid lung zone',
       'right upper lung zone', 'trachea', 'upper mediastinum']

# EHR variables considered
SELECTED_VAR=['Alanine aminotransferase', 'Albumin', 'Alkaline phosphate',
       'Anion gap', 'Asparate aminotransferase', 'Bicarbonate',
       'Bilirubin', 'Blood urea nitrogen',  'Chloride', 'Creatinine',
       'Diastolic blood pressure', 'Fraction inspired oxygen',
       'Glascow coma scale eye opening',
       'Glascow coma scale motor response',
       'Glascow coma scale verbal response', 'Glucose', 'Heart Rate',
       'Height', 'Hematocrit', 'Hemoglobin', 'Magnesium',
       'Mean blood pressure', 'Oxygen saturation',
       'Partial pressure of carbon dioxide',
       'Partial thromboplastin time', 'Platelets',
       'Positive end-expiratory pressure', 'Potassium',
       'Prothrombin time', 'Respiratory rate', 'Sodium',
       'Systolic blood pressure', 'Temperature', 'Troponin-T',
       'Urine output', 'Weight', 'White blood cell count', 'pH']

GCS={"Glascow coma scale verbal response": {
        "possible_values": ["1 No Response", "1.0 ET/Trach", "2 Incomp sounds", "3 Inapprop words", "4 Confused", "5 Oriented", "Confused", "Inappropriate Words", "Incomprehensible sounds", "No Response", "No Response-ETT", "Oriented"],
        "values": {
            "No Response-ETT": 1,
            "No Response": 1,
            "1 No Response": 1,
            "1.0 ET/Trach": 1,
            "2 Incomp sounds": 2,
            "Incomprehensible sounds": 2,
            "3 Inapprop words": 3,
            "Inappropriate Words": 3,
            "4 Confused": 4,
            "Confused": 4,
            "5 Oriented": 5,
            "Oriented": 5
        }
    },
    "Glascow coma scale eye opening": {
        "possible_values": ["1 No Response", "2 To pain", "3 To speech", "4 Spontaneously", "None", "Spontaneously", "To Pain", "To Speech"],
        "values": {
            "None": 0,
            "1 No Response": 1,
            "2 To pain": 2, 
            "To Pain": 2,
            "3 To speech": 3, 
            "To Speech": 3,
            "4 Spontaneously": 4,
            "Spontaneously": 4
        }
    },
    "Glascow coma scale motor response": {
        "possible_values": ["1 No Response", "2 Abnorm extensn", "3 Abnorm flexion", "4 Flex-withdraws", "5 Localizes Pain", "6 Obeys Commands", "Abnormal Flexion", "Abnormal extension", "Flex-withdraws", "Localizes Pain", "No response", "Obeys Commands"],
        "values": {
            "1 No Response": 1,
            "No response": 1,
            "2 Abnorm extensn": 2,
            "Abnormal extension": 2,
            "3 Abnorm flexion": 3,
            "Abnormal Flexion": 3,
            "4 Flex-withdraws": 4,
            "Flex-withdraws": 4,
            "5 Localizes Pain": 5,
            "Localizes Pain": 5,
            "6 Obeys Commands": 6,
            "Obeys Commands": 6
        }
    }
}

# diseases considered in the disease progression task
PROGRESSION_DISEASE=[ "atelectasis",'enlarged cardiac silhouette',"consolidation",'pulmonary edema','lung opacity','pleural effusion','pneumonia']

# Anatomical regions considered
SELECTED_REGIONS=[ "left lung",
    "right lung",
    "left lower lung zone",
    "right lower lung zone",
    "left hilar structures",
    "right hilar structures",
    "left costophrenic angle",
    "right costophrenic angle",
    "cardiac silhouette",
    "mediastinum"]


# progression lables
PROGRESSION_DICT={'worsened':0, 'improved':1, 'no change':2, 'not mentioned':3}
FILL_VALUE=3


# mask indicators of EHR time-series variables
SELECTED_VAR_MASK=[i+'_mask' for i in SELECTED_VAR]

# EHR demographics, here we consider age, height, weight, gender, race, language, marital status
# We discretize  gender, race, language and marital status
DEMOGRAPHIC=[ 'age', 'height', 'weight', 'gender_F',
       'gender_M', 'race_group_ASIAN', 'race_group_BLACK/AFRICAN AMERICAN',
       'race_group_HISPANIC/LATINO', 'race_group_OTHER',
       'race_group_UNKNOWN/UNABLE TO OBTAIN/PATIENT DECLINED TO ANSWER',
       'race_group_WHITE', 'language_?', 'language_ENGLISH',
       'marital_status_DIVORCED', 'marital_status_MARRIED',
       'marital_status_SINGLE', 'marital_status_WIDOWED']


# length of stay labels definition
LOS_bins = [2, 3, 4, 6, 11]
LOS_labels = [0, 1, 2, 3]  

# output size for each task
TASK_OUTPUT_SIZE={'length_of_stay':4,'disease_progression':3,'mortality':1}

# training monitor for each task
TASK_MONITOR={'length_of_stay':'val/accuracy','disease_progression':'val/f1_macro','mortality':'val/pr_auc'}

# disease progression is multi-label, mortality is binary classification, length of stay is multi-class classification
TASK_NUM_CLASSES={'length_of_stay':1,'disease_progression':7,'mortality':1}

IMAGE_CHANNEL=3
IMAGE_SIZE=224

VARIABLE_DIM=38 # contianing the Variable and its mask
NUM_DEMOGRAPHIC_VAR=17 # containing the demographic variables and the dicretized variables