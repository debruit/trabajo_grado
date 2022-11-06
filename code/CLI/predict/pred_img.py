from keras.models import load_model
import nibabel as nib
import numpy as np
from preprocessing import load_data_seg
from postprocessing import reconstruction

def predict(model_path, input_path, output_path):
    my_model = load_model(model_path, compile=False)
    if not my_model:
        print('Error loading model or model not found')
        exit()
        
        
    load_data_seg.load_data(my_model,input_path, output_path)


# print(temp_prediction_argmax.shape)
# print(test_mask_argmax.shape)
# print(np.unique(temp_prediction_argmax))

