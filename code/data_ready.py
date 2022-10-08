import splitfolders  # or import split_folders
from scipy.ndimage import zoom
from patchify import patchify, unpatchify
import numpy as np
import nibabel as nib
import glob
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


def crop_img(img):
    temp_sz = img.shape[0] - 256
    szs = int(temp_sz/2)
    img = img[szs:img.shape[0] - szs, :, :]
    if (img.shape[1] != 256 or img.shape[2] != 256):
        temp_sz = img.shape[1] - 256
        szs = int(temp_sz/2)
        img = img[:, szs:img.shape[1] - szs, szs:img.shape[2] - szs]

    return img


TRAIN_DATASET_PATH = 'training_data2/'
# # #VALIDATION_DATASET_PATH = 'BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'

# test_image_t1 = nib.load(TRAIN_DATASET_PATH+'imgs/sub-A003_ses-01_acq-highres_T1w.nii.gz').get_fdata()
# test_mask=nib.load(TRAIN_DATASET_PATH+'masks/sub-A003_ses-01_segmented.nii.gz').get_fdata()
# # print(test_mask.max())
# print(test_image_t1.max())
# # Scalers are applied to 1D so reshape and then reshape back to original shape.
# test_image_t1 = scaler.fit_transform(test_image_t1.reshape(-1, test_image_t1.shape[-1])).reshape(test_image_t1.shape)


# x = crop_img(test_image_t1)
# # x = zoom(test_image_t1, (1, 1.143, 1.143))

# print(x.shape)


# z = test_image_t1.swapaxes(-2,-1)[...,:,::-1]


# # test_image = patchify(x, (64, 64, 64), step=64)

# # test_image = np.reshape(
# #     test_image, (-1, test_image.shape[3], test_image.shape[4], test_image.shape[5]))

# # print(test_image.shape)

# plt.figure(figsize=(18, 14))

# plt.subplot(231)
# plt.imshow(test_image_t1[:,:,130], cmap='gray')
# plt.subplot(232)
# plt.imshow(test_mask[:,:,130], cmap='gray')
# plt.subplot(235)
# plt.imshow(x[:,:,18], cmap='gray')
# plt.show()

# plt.imshow(test_image_t1[ :, :, 160],cmap='gray')
# plt.show()

# test_image = unpatchify(test_image, test_image_t1.shape)

# print(test_image_t1.shape)


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
# End of understanding the dataset. Now get it organized.
#####################################

# Now let us apply the same as above to all the images...
# Merge channels, crop, patchify, save
# GET DATA READY =  GENERATORS OR OTHERWISE

# Keras datagenerator does ntot support 3d

# # # images lists harley
t1_list = sorted(glob.glob(TRAIN_DATASET_PATH+'imgs/*.nii.gz'))
mask_list = sorted(glob.glob(TRAIN_DATASET_PATH+'masks/*.nii.gz'))


# Each volume generates 18 64x64x64x4 sub-volumes.
# Total 369 volumes = 6642 sub volumes

for img in tqdm(range(len(t1_list))):  # Using t1_list as all lists are of same size
    print("Now preparing image and masks number: ", img)

    temp_image_t1 = nib.load(t1_list[img]).get_fdata()
    temp_image_t1 = scaler.fit_transform(temp_image_t1.reshape(-1, temp_image_t1.shape[-1])).reshape(temp_image_t1.shape)

    temp_mask = nib.load(mask_list[img]).get_fdata()
    # temp_mask=temp_mask.astype(np.uint8)
    # print(np.unique(temp_mask))

    temp_image_t1 = crop_img(temp_image_t1)
    # temp_image_t1 = zoom(temp_image_t1, (1, 1.143, 1.143))


    # temp_image_t1 = patchify(temp_image_t1, (64, 64, 64), step=64)

    # temp_image_t1 = np.reshape(temp_image_t1, (-1, temp_image_t1.shape[3], temp_image_t1.shape[4], temp_image_t1.shape[5]))
    
    
    temp_mask = crop_img(temp_mask)
    # temp_mask = zoom(temp_mask, (1, 1.143, 1.143))


    # temp_mask = patchify(temp_mask, (64, 64, 64), step=64)

    # temp_mask = np.reshape(temp_mask, (-1, temp_mask.shape[3], temp_mask.shape[4], temp_mask.shape[5]))


    np.save(TRAIN_DATASET_PATH+'imgs_npy/imgs/image_'+ \
            str(img)+'.npy', temp_image_t1)
    np.save(TRAIN_DATASET_PATH+'imgs_npy/masks/mask_'+str(img)+'.npy', temp_mask)


################################################################
# Repeat the same from above for validation data folder OR
# Split training data into train and validation

"""
Code for splitting folder into train, test, and val.
Once the new folders are created rename them and arrange in the format below to be used
for semantic segmentation using data generators. 

pip install split-folders
"""

input_folder = TRAIN_DATASET_PATH+'imgs_npy/'
output_folder = TRAIN_DATASET_PATH+'input_data/'
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None)  # default values
########################################
