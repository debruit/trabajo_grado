## =========================================================================
## @author Juan Sebastián Ruiz Bulla (ruizju@javeriana.edu.co)
## @author David Alejandro Castillo Chíquiza (castillo_da@javeriana.edu.co)
## @author Oscar David Falla Pulido (falla_o@javeriana.edu.co)
## =========================================================================

import nibabel as nib
import os    
from segment.load import prepare_data_seg
import glob
from tqdm import tqdm

def load_data(model, input_path, output_path):
    
    print("Loading data...")
    
    if not os.path.exists(os.getcwd()+'/segmentations'):
        os.mkdir(os.getcwd()+'/segmentations')
    
    if(str(input_path).__contains__('.nii')):
        info_img = nib.load(input_path)
        img_data = info_img.get_fdata()
        
        out_format = str(input_path).split('/')[-1].split('.')[0] + '_seg.nii.gz' 
        
        if(str(output_path).__contains__('.nii')):
            output = 'segmentations/' + out_format
        else:
            output = output_path +'/'+ out_format
            
        print("Image loaded")
        
        prepare_data_seg.prepare_data(model, info_img, img_data, output)
    else:
        list_imgs = sorted(glob.glob(input_path + '/*.nii*'))
        for img in tqdm(range(len(list_imgs))):
            info_img = nib.load(list_imgs[img])
            img_data = info_img.get_fdata()
            
            out_format = str(list_imgs[img]).split('/')[-1].split('.')[0] + '_seg.nii.gz' 
        
            if(str(output_path).__contains__('.nii')):
                output = 'segmentations/' + out_format
            else:
                output = output_path +'/'+ out_format
            
            print("Image loaded")
            
            prepare_data_seg.prepare_data(model, info_img, img_data, output)