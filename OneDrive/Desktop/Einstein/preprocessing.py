#imports
import numpy as np
import os
from fsl.wrappers.flirt import flirt

# import matplotlib.pyplot as plt
import nibabel as nb
from deepbrain import Extractor


def skull_stripping(imgpath):
    '''Strips the skull from the brain and returns the skull-stripped image with its binary mask'''
    img = nb.load(imgpath).get_fdata()
    ext = Extractor()
    prob = ext.run(img) 

    # mask can be obtained as:
    mask = prob > 0.5
    print("\n\nSkull-stripping done..\n")
    print("Shape of img is {}, and mask is {}".format(np.shape(img),np.shape(mask)))
    stripped_img = img * mask
    return stripped_img, mask

def normalize(img):
    '''Normalize the intensity [0,255]'''
    print("\n\nIntensity normalization done..\n")
    return (255* (img - np.min(img))/(np.max(img)-np.min(img)))
   
def numpytonifti(np_arr):
    '''Converts the numpy array to Nifti image'''
    affine = np.eye(4)
    nii_img = nb.Nifti1Image(np_arr, affine)
    return nii_img

def extract_subfolder_names(folder_path):
  print("calling this...")
  subfolder_list = []
  for root, dirs, files in os.walk(folder_path):
    # print(root,'\n')
    if root == folder_path:  # Only process the root directory (level 1)
        for dir_name in dirs:
            # print("\n\n\n\nDIR NAME=", dir_name)
            subfolder_list.append(dir_name)
        break
    # for file in files:
    #     nii_file_list.append(file)
    # print("Subfolder list=", subfolder_list)
  return subfolder_list

def register_mni(stripped_img_nii, folder_path, output_path):
    reg = 1

    ref = f"/public/apps/fsl/6.0.5_cpu/data/standard/MNI152_T1_1mm_brain.nii.gz"
    out = output_path + f"{folder_path}_reg_brain_{reg}.nii.gz"
    mat = output_path + '/mat/' + f"{folder_path}_reg_brain_{reg}.mat"
    print(mat)
    if not os.path.exists(mat):
        flirt(stripped_img_nii,ref,out=out,omat=mat,v=True)
        print("Registered {} to MNI space".format(folder_path))

        
def main(folder_name, output_path):
    # Load the images and apply skull-stripping, then save the files

    # CN
    cn_files_list = []
    subfolder_names_list = extract_subfolder_names(folder_name)
    # print('\n\n Subfolder list from function =', subfolder_names_list)

    # Track which subfolder is being processed
    subfolder_index = -1  # Initialize to -1 to start with 0 after increment
    
    for root, dirs, files in os.walk(folder_name):
        # Only increment the subfolder index if there are files in the current folder
        if files:
            subfolder_index += 1  # Move to the next subfolder
        
        # Ensure subfolder index doesn't exceed the length of subfolder_names_list
        if subfolder_index < len(subfolder_names_list):
            for i, file in enumerate(files):
                print(f"Processing file in subfolder: {subfolder_names_list[subfolder_index]}")
                
                file_path = os.path.abspath(os.path.join(root, file))
                print(file_path)

                # Apply skull stripping
                stripped_img, stripped_img_mask = skull_stripping(file_path)
                stripped_img_nii = numpytonifti(stripped_img)

                # Register with MNI, using the correct subfolder name
                register_mni(stripped_img_nii, subfolder_names_list[subfolder_index], output_path)
        else:
            print(f"Warning: Subfolder index {subfolder_index} exceeds subfolder list length.")

    
if __name__=='__main__':
    print("CUDA VISIBLE DEVICES = ",os.environ.get("CUDA_VISIBLE_DEVICES"))
    # test_data = Path('/gs/gsfs0/users/shonandi/Projects/ADNI/002_S_0413/')
    datafolder_CN = f"/gs/gsfs0/users/shonandi/Projects/ADNI/CN_TRAIN_VAL_data"
    # output_CN = f"/gs/gsfs0/users/shonandi/Projects/ADNI/Preprocessed/CN_List/"
    output_AD = f"/gs/gsfs0/users/shonandi/Projects/ADNI/Preprocessed/AD_List/"

    # main(datafolder_CN, output_CN)
    main(datafolder_CN, output_AD)
    
    print('Complete')