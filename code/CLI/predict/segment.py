from postprocessing import reconstruction    


def seg_img(model, temp_img_input, info_img, output_path):   
    temp_prediction = model.predict(temp_img_input)
    
    reconstruction.convert_img(info_img, temp_prediction, output_path)