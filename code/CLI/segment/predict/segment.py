## =========================================================================
## @author Juan Sebastián Ruiz Bulla (ruizju@javeriana.edu.co)
## @author David Alejandro Castillo Chíquiza (castillo_da@javeriana.edu.co)
## @author Oscar David Falla Pulido (falla_o@javeriana.edu.co)
## =========================================================================

from segment.postprocessing import reconstruction    


def seg_img(model, temp_img_input, info_img, output_path):   
    
    print("Segmenting image...")
    
    temp_prediction = model.predict(temp_img_input)
    
    reconstruction.convert_img(info_img, temp_prediction, output_path)