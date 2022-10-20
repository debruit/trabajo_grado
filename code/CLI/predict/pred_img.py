from keras.models import load_model
import nibabel as nib
import numpy as np
from preprocessing import crop
from postprocessing import reconstruction

def predict(model_path, input_path, output_path):
    my_model = load_model(model_path, compile=False)

    input_img = nib.load(input_path).get_fdata()
    
    temp_img = crop.crop_img(input_img)

    temp_img_input = np.expand_dims(temp_img, axis=0)
    temp_prediction = my_model.predict(temp_img_input)
    
    reconstruction.convert_img(input_path, temp_prediction, output_path)


# print(temp_prediction_argmax.shape)
# print(test_mask_argmax.shape)
# print(np.unique(temp_prediction_argmax))

