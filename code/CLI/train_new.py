import argparse
from preprocessing import load_data
from train import fit 

parser = argparse.ArgumentParser(description='Train a new model for ROI segmentation on the input dataset (MRI)')

# parser.add_argument('-m','--model', help='Model path for prediction', default='best_model.h5')
parser.add_argument('-d', '--dataset', help='Dataset path for training', required=True)
parser.add_argument('-om', '--output-model', help='Output model path', default='model.h5')


args = parser.parse_args()


train_imgs, train_masks, val_imgs, val_masks = load_data.load_input_data(args.dataset)


fit.fit_model(train_imgs, train_masks, val_imgs, val_masks, args.output_model)
