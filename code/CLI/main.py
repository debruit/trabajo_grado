import argparse
from preprocessing import load_data_train
from predict import pred_img

parser = argparse.ArgumentParser(description='Initialize the system either \
        to segment adipose tissue on the input image(s) (MRI) or \
        to train a new model for ROI segmentation on the input dataset (MRI)')

parser.add_argument('-s','--system', help='Initialize the system (segment or train)', default='None')

parser.add_argument('-i','--input', help='Input image(s) path', default='None')
parser.add_argument('-o','--output', help='Output image(s) path', default='segmented_image.nii.gz')
parser.add_argument('-m','--model', help='Model path for prediction', default='model.h5')

parser.add_argument('-d', '--dataset', help='Dataset path for training', default='None')
parser.add_argument('-om', '--output-model', help='Output model path', default='model.h5')

args = parser.parse_args()

if(args.system == 'segment'):
    if(args.input == 'None'):
        print('Please provide an input image(s) path')
        exit()
    pred_img.predict(args.model, args.input, args.output)
    
elif(args.system == 'train'):
    if(args.dataset == 'None'):
        print('Please provide a dataset path for training the model')
        exit()
    train_imgs, train_masks, val_imgs, val_masks = load_data_train.load_input_data(args.dataset, args.output_model)

else:
    print('Need to specify -s flag to either segment or train')
    exit()

# print(args)