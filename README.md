# Undergraduate thesis

Code implementation for the undergraduate thesis: Segmentación Semi-Automática de Tejido Adiposo en el Espacio Parafaríngeo por Medio de Aprendizaje Profundo.

## Default model

To run the code for adipose tissue segmentation, the model needs to be downloaded and placed into the same folder as main.py. The default model for adipose tissue segmentation can be found in the following link: https://drive.google.com/file/d/1zo54kOOyDXvUoWW3wFGMs2lzMJ_7qNNX/view?usp=share_link

## Usage

usage: main.py [-h] [-s SYSTEM] [-i INPUT] [-o OUTPUT] [-m MODEL] [-d DATASET] [-om OUTPUT_MODEL]

Initialize the system either to segment adipose tissue on the input image(s) (MRI) or to train a new model for ROI segmentation on the input dataset (MRI)

optional arguments:
  -h, --help            show this help message and exit
  -s SYSTEM, --system SYSTEM
                        Initialize the system (segment or train)
  -i INPUT, --input INPUT
                        Input image(s) path
  -o OUTPUT, --output OUTPUT
                        Output image(s) path
  -m MODEL, --model MODEL
                        Model path for prediction
  -d DATASET, --dataset DATASET
                        Dataset path for training
  -om OUTPUT_MODEL, --output-model OUTPUT_MODEL
                        Output model path