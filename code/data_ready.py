# https://youtu.be/oB35sV1npVI
"""
Use this code to get your BRATS 2020 dataset ready for semantic segmentation. 
Code can be divided into a few parts....

#Combine 
#Changing mask pixel values (labels) from 4 to 3 (as the original labels are 0, 1, 2, 4)
#Visualize


https://pypi.org/project/nibabel/

All BraTS multimodal scans are available as NIfTI files (.nii.gz) -> commonly used medical imaging format to store brain imagin data obtained using MRI and describe different MRI settings

T1: T1-weighted, native image, sagittal or axial 2D acquisitions, with 1–6 mm slice thickness.
T1c: T1-weighted, contrast-enhanced (Gadolinium) image, with 3D acquisition and 1 mm isotropic voxel size for most patients.
T2: T2-weighted image, axial 2D acquisition, with 2–6 mm slice thickness.
FLAIR: T2-weighted FLAIR image, axial, coronal, or sagittal 2D acquisitions, 2–6 mm slice thickness.

#Note: Segmented file name in Folder 355 has a weird name. Rename it to match others.
"""


import numpy as np
import nibabel as nib
import glob
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


def crop_img(img):
    temp_sz = img.shape[0] - 320
    szs = int(temp_sz/2)
    img = img[szs:img.shape[0] - szs, :, :]
    if(img.shape[1] != 480 or img.shape[2] != 480):
        temp_sz = img.shape[1] - 480
        szs = int(temp_sz/2)
        img = img[:, szs:img.shape[1] - szs, szs:img.shape[2] - szs]
        
    return img

##########################
#This part of the code to get an initial understanding of the dataset.
#################################
#PART 1: Load sample images and visualize
#Includes, dividing each image by its max to scale them to [0,1]
#Converting mask from float to uint8
#Changing mask pixel values (labels) from 4 to 3 (as the original labels are 0, 1, 2, 4)
#Visualize
###########################################
#View a few images

#Note: Segmented file name in Folder 355 has a weird name. Rename it to match others.

TRAIN_DATASET_PATH = 'training_data2/'
# # #VALIDATION_DATASET_PATH = 'BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'

# test_image_t1=nib.load(TRAIN_DATASET_PATH+'imgs/sub-A038_ses-01_acq-highres_T1w.nii.gz').get_fdata()
# print(test_image_t1.max())
# #Scalers are applied to 1D so let us reshape and then reshape back to original shape. 
# test_image_t1=scaler.fit_transform(test_image_t1.reshape(-1, test_image_t1.shape[-1])).reshape(test_image_t1.shape)

# test_mask=nib.load(TRAIN_DATASET_PATH+'masks/sub-A038_ses-01_segmented.nii.gz').get_fdata()
# print(test_mask.max())
# test_mask=test_mask.astype(np.uint8)

# # print(np.unique(test_mask))  #0, 1, 2, 4 (Need to reencode to 0, 1, 2, 3)
# # print(np.unique(test_mask,return_counts=True)) 

# # # prueba = np.resize(test_mask, (256,256,256))
# print(test_image_t1.shape)
# print(test_mask.shape)




# if(test_image_t1.shape[0] != 320):
#     test_image_t1 = crop_img(test_image_t1)
        
# if(test_mask.shape[0] != 320):
#     test_mask = crop_img(test_mask)



# import cv2

# width = 256
# height = 256
# img_stack_sm_2 = np.zeros((width, height,len(test_image_flair)))

# for idx in range(len(test_image_flair)):
#     img = test_image_flair[:, :, idx]   
#     img_sm = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
#     img_stack_sm_2[:, :, idx] = img_sm



# # print(np.unique(img_stack_sm,return_counts=True)) 

# # print(np.where(img_stack_sm_2==1))

# import random

# plt.figure(figsize=(18, 14))

# plt.subplot(231)
# plt.imshow(test_image_t1[:,:,160], cmap='gray')
# plt.title('Image flair')
# plt.subplot(235)
# plt.imshow(test_mask[:,:,160])
# plt.title('Mask')
# plt.show()

####################################################################
#####################################
#End of understanding the dataset. Now get it organized.
#####################################

#Now let us apply the same as above to all the images...
#Merge channels, crop, patchify, save
#GET DATA READY =  GENERATORS OR OTHERWISE

#Keras datagenerator does ntot support 3d

# # # images lists harley
t1_list = sorted(glob.glob(TRAIN_DATASET_PATH+'imgs/*.nii.gz'))
mask_list = sorted(glob.glob(TRAIN_DATASET_PATH+'masks/*.nii.gz'))


#Each volume generates 18 64x64x64x4 sub-volumes. 
#Total 369 volumes = 6642 sub volumes

for img in range(len(t1_list)):   #Using t1_list as all lists are of same size
    print("Now preparing image and masks number: ", img)
      
    temp_image_t1=nib.load(t1_list[img]).get_fdata()
    temp_image_t1=scaler.fit_transform(temp_image_t1.reshape(-1, temp_image_t1.shape[-1])).reshape(temp_image_t1.shape)
        
    temp_mask=nib.load(mask_list[img]).get_fdata()
    temp_mask=temp_mask.astype(np.uint8)
    #print(np.unique(temp_mask))
    
    if(temp_image_t1.shape[0] != 320):
        temp_image_t1 = crop_img(temp_image_t1)
        
    if(temp_mask.shape[0] != 320):
        temp_mask = crop_img(temp_mask)
    
    np.save(TRAIN_DATASET_PATH+'imgs_npy/imgs/image_'+str(img)+'.npy', temp_image_t1)
    np.save(TRAIN_DATASET_PATH+'imgs_npy/masks/mask_'+str(img)+'.npy', temp_mask)  
   
     
################################################################
#Repeat the same from above for validation data folder OR
#Split training data into train and validation

"""
Code for splitting folder into train, test, and val.
Once the new folders are created rename them and arrange in the format below to be used
for semantic segmentation using data generators. 

pip install split-folders
"""
import splitfolders  # or import split_folders

input_folder = TRAIN_DATASET_PATH+'imgs_npy/'
output_folder = TRAIN_DATASET_PATH+'input_data/'
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None) # default values
########################################
