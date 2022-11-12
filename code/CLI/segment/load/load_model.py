from keras.models import load_model
from segment.load import load_data_seg

def load_model_data(model_path, input_path, output_path):
    print('Loading model...')
    
    my_model = load_model(model_path, compile=False)
    if not my_model:
        print('Error loading model or model not found')
        exit()
        
    print('Model loaded')
        
    load_data_seg.load_data(my_model,input_path, output_path)
