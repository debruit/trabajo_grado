import numpy as np
from predict import segment
from preprocessing import crop
    

def prepare_data(model, info_img, input_img, output_path):
    temp_img = crop.crop_img(input_img)

    temp_img_input = np.expand_dims(temp_img, axis=0)
    
    segment.seg_img(model, temp_img_input, info_img, output_path)