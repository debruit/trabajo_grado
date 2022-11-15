## =========================================================================
## @author Juan Sebastián Ruiz Bulla (ruizju@javeriana.edu.co)
## @author David Alejandro Castillo Chíquiza (castillo_da@javeriana.edu.co)
## @author Oscar David Falla Pulido (falla_o@javeriana.edu.co)
## =========================================================================

import numpy as np
from segment.predict import segment
from preprocessing import crop
    

def prepare_data(model, info_img, input_img, output_path):
    
    print("Preparing data...")
    
    temp_img = crop.crop_img(input_img)

    temp_img_input = np.expand_dims(temp_img, axis=0)
    
    segment.seg_img(model, temp_img_input, info_img, output_path)