## =========================================================================
## @author Juan Sebastián Ruiz Bulla (ruizju@javeriana.edu.co)
## @author David Alejandro Castillo Chíquiza (castillo_da@javeriana.edu.co)
## @author Oscar David Falla Pulido (falla_o@javeriana.edu.co)
## =========================================================================

import nibabel as nib
import numpy as np

def convert_img(info_img, pred, output_path):
    
    print("Reconstructing image...")    
    
    img_data = info_img.get_fdata()
    
    
    prediction = pred[0,...,0]
    
    temp_pred = np.zeros_like(img_data)
    
    temp_sz_y = temp_pred.shape[1] - 96
    szs_y = int(temp_sz_y/2)
    temp_pred[85:245, szs_y+25:(temp_pred.shape[1] - szs_y) + 25, 70:198] = prediction
    
    pred_img = nib.Nifti1Image(temp_pred, info_img.affine, info_img.header)
    
    nib.save(pred_img, output_path)
    
    print('Segmented image saved at: ', output_path)