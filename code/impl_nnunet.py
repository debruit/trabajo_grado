import tensorflow as tf
import torch

print("Torch: ",torch.cuda.is_available())

      
print("TF: ",tf.test.is_built_with_cuda())