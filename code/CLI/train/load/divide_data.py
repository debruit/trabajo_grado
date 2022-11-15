## =========================================================================
## @author Juan Sebastián Ruiz Bulla (ruizju@javeriana.edu.co)
## @author David Alejandro Castillo Chíquiza (castillo_da@javeriana.edu.co)
## @author Oscar David Falla Pulido (falla_o@javeriana.edu.co)
## =========================================================================

import splitfolders
import os
from train.train_m import fit 

def divide(output_model):
    print('Dividing images into train and validation folders...')
    
    splitfolders.ratio(os.getcwd()+'/imgs_preprocessed/', output=os.getcwd()+'/input_data/' , ratio=(.8, .2), group_prefix=None)
    
    train_imgs = os.getcwd()+'/input_data/train/imgs_npy/'
    train_masks = os.getcwd()+'/input_data/train/masks_npy/'
    
    val_imgs = os.getcwd()+'/input_data/val/imgs_npy/'
    val_masks = os.getcwd()+'/input_data/val/masks_npy/'
    
    fit.fit_model(train_imgs, train_masks, val_imgs, val_masks, output_model)