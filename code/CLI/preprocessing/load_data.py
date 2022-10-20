from tqdm import tqdm
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import glob
import numpy as np
import os
import splitfolders

from crop import crop_img

def load_input_data(dataset_path):

    scaler = MinMaxScaler()

    # os.path.join(os.getcwd()+'/*.nii.gz')

    t1ce_list = sorted(glob.glob(dataset_path+'/imgs/*.ni*'))
    mask_list = sorted(glob.glob(dataset_path+'/masks/*.ni*'))


    print('Preparing images for training...')

    for img in tqdm(range(len(t1ce_list))): 
    
    
        temp_image_t1ce=nib.load(t1ce_list[img]).get_fdata()

        
        temp_image_t1 = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)


        temp_image_t1 = crop_img(temp_image_t1ce)
        
    
            
        temp_mask = nib.load(mask_list[img]).get_fdata()

        temp_mask = crop_img(temp_mask)
        
        os.mkdir(os.getcwd()+'/imgs_preprocessed/imgs_npy')
        os.mkdir(os.getcwd()+'/imgs_preprocessed/masks_npy')

        
        np.save(os.getcwd()+'/imgs_preprocessed/imgs_npy/image_'+str(img)+'.npy', temp_image_t1)
        np.save(os.getcwd()+'/imgs_preprocessed/masks_npy/mask_'+str(img)+'.npy', temp_mask)  

    

    print('Dividing images into train and validation folders...')
    
    splitfolders.ratio(os.getcwd()+'/imgs_preprocessed/', output=os.getcwd()+'/input_data/', seed=80 , ratio=(.8, .2), group_prefix=None)
    
    train_imgs = os.getcwd()+'/input_data/train/imgs_npy/'
    train_masks = os.getcwd()+'/input_data/train/masks_npy/'
    
    val_imgs = os.getcwd()+'/input_data/val/imgs_npy/'
    val_masks = os.getcwd()+'/input_data/val/masks_npy/'
    
    return train_imgs, train_masks, val_imgs, val_masks
   