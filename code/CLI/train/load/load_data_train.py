from tqdm import tqdm
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import glob
import numpy as np
import os

from train.load import prepare_data_train as prep
from train.load import divide_data


def load_input_data(dataset_path, output_model):
    
    print("Loading data...")

    scaler = MinMaxScaler()
    
    if not os.path.exists(os.getcwd()+'/imgs_preprocessed'):
        os.mkdir(os.getcwd()+'/imgs_preprocessed')
            
        os.mkdir(os.getcwd()+'/imgs_preprocessed/imgs_npy')
        os.mkdir(os.getcwd()+'/imgs_preprocessed/masks_npy')
        
    if(str(dataset_path).__contains__('.nii')):
        print('Dataset needs to be a folder with .nii images, not a single .nii image')
        exit()


    imgs_list = sorted(glob.glob(dataset_path+'/imgs/*.ni*'))
    mask_list = sorted(glob.glob(dataset_path+'/masks/*.ni*'))


    print('Preparing images for training...')

    for img in tqdm(range(len(imgs_list))): 
    
    
        temp_image_t1ce=nib.load(imgs_list[img]).get_fdata()
        
        temp_image_t1 = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
        
        imgs = prep.prepare_data(temp_image_t1)
    
            
        temp_mask = nib.load(mask_list[img]).get_fdata()
        
        mask = prep.prepare_data(temp_mask)
        
        

        
        np.save(os.getcwd()+'/imgs_preprocessed/imgs_npy/image_'+str(img)+'.npy', imgs)
        np.save(os.getcwd()+'/imgs_preprocessed/masks_npy/mask_'+str(img)+'.npy', mask)  

    

    divide_data.divide(output_model)
   