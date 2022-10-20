import argparse
import sys
sys.path.append("..")
from predict import pred_img

parser = argparse.ArgumentParser(description='Segment adipose tissue on the input image (MRI)')

parser.add_argument('-i','--input', help='Input image path', required=True)
parser.add_argument('-o','--output', help='Output image path', default='segmented_image.nii.gz')
parser.add_argument('-m','--model', help='Model path for prediction', default='best_model.h5')

args = parser.parse_args()

pred_img.predict(args.model, args.input, args.output)

# print(args)