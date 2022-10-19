import nibabel as nib

def convert_img(mask, pred):
    mask_nii = nib.load(mask)
    mask = mask_nii.get_fdata()
    
    prediction = pred[0,...,0]
    
    temp_pred = mask.copy()
    temp_sz_y = temp_pred.shape[1] - 96
    szs_y = int(temp_sz_y/2)
    temp_pred[85:245, szs_y+25:(temp_pred.shape[1] - szs_y) + 25, 70:198] = prediction
    
    pred_img = nib.Nifti1Image(temp_pred, mask_nii.affine, mask_nii.header)
    
    return pred_img
    
    # nib.save(pred_img, 'segmented_image.nii.gz')