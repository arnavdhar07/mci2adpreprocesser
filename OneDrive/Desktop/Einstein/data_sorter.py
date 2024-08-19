import pandas as pd
import numpy as np
import os
from pathlib import Path
import shutil

def copy_files(src, dest):
    print(src)
    print(dest)
    shutil.copytree(src, dest, dirs_exist_ok=True)

os.chdir('/gs/gsfs0/users/shonandi/Projects/')
patient_list = pd.read_csv('/gs/gsfs0/users/shonandi/Projects/ADNI/AD_CN_TRAIN_VAL_12m_7_22_2024.csv')
dataset_dir = Path('/gs/gsfs0/users/shonandi/Projects/AD_CN_TRAIN_VAL_data_collection/ADNI/')
dataset_list = os.listdir(dataset_dir)

tagged_list = patient_list[['Subject','Group']]

patient_list_CN = patient_list[patient_list['Group']=='CN']
patient_list_AD = patient_list[patient_list['Group']=='AD']

#Store the list of Image IDs for CN and AD to sort the data from the collection directory
subject_ids_CN = patient_list_CN['Subject']
#subject_ids_CN.to_csv('Subject_IDs_CN_list.csv')
subject_ids_AD = patient_list_AD['Subject']
#subject_ids_AD.to_csv('Subject_IDs_AD_list.csv')

subject_ids_CN.to_list()
subject_ids_AD.to_list()


# match files and copy them to new folder
matched_files_AD = [file for file in subject_ids_AD if file in dataset_list]
print("Matched AD patients...\n", matched_files_AD)
for file in matched_files_AD:
    copy_files(os.path.join(dataset_dir,file), os.path.join('base_AD_TRAIN_VAL_data/', os.path.basename(file)))
    
matched_files_CN = [file for file in subject_ids_CN if file in dataset_list]
print("Matched CN patients...\n",matched_files_CN)
for file in matched_files_CN:
    copy_files(os.path.join(dataset_dir,file), os.path.join('base_CN_TRAIN_VAL_data/', os.path.basename(file)))

print("Done")