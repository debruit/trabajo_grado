# https://youtu.be/ScdCQqLtnis
"""
@author: Sreenivas Bhattiprolu

Code to train batches of cropped BraTS 2020 images using 3D U-net.

Please get the data ready and define custom data gnerator using the other
files in this directory.

Images are expected to be 128x128x128x3 npy data (3 corresponds to the 3 channels for 
                                                  test_image_flair, test_image_t1ce, test_image_t2)
Change the U-net input shape based on your input dataset shape (e.g. if you decide to only se 2 channels or all 4 channels)

Masks are expected to be 128x128x128x3 npy data (4 corresponds to the 4 classes / labels)


You can change input image sizes to customize for your computing resources.
"""


from gc import callbacks
import os
import numpy as np
from datagen import imageLoader
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import glob
import random


# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession


# def fix_gpu():
#     config = ConfigProto()
#     config.gpu_options.allow_growth = True
#     session = InteractiveSession(config=config)


# fix_gpu()



####################################################
train_img_dir = "training_data2/input_data/train/imgs/"
train_mask_dir = "training_data2/input_data/train/masks/"

# img_list = os.listdir(train_img_dir)
# msk_list = os.listdir(train_mask_dir)

# num_images = len(os.listdir(train_img_dir))

# img_num = 0
# test_img = np.load(train_img_dir+img_list[img_num])
# test_mask = np.load(train_mask_dir+msk_list[img_num])
# plt.figure(figsize=(12, 8))

# plt.subplot(221)
# plt.imshow(test_img[:,:,138], cmap='gray')
# plt.subplot(224)
# plt.imshow(test_mask[:,:,138])
# plt.title('Mask')
# plt.show()


##############################################################
#Define the image generators for training and validation


val_img_dir = "training_data2/input_data/val/imgs/"
val_mask_dir = "training_data2/input_data/val/masks/"

train_img_list=os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list=os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)
##################################

########################################################################
batch_size = 1

train_img_datagen = imageLoader(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = imageLoader(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)

# #Verify generator.... In python 3 next() is renamed as __next__()
# img, msk = train_img_datagen.__next__()

# img_num = 0
# test_img=img[img_num]
# test_mask=msk[img_num]

# plt.figure(figsize=(12, 8))

# plt.subplot(221)
# plt.imshow(test_img[:,:,138], cmap='gray')
# plt.title('Image flair')
# plt.subplot(224)
# plt.imshow(test_mask[:,:,138])
# plt.title('Mask')
# plt.show()


###########################################################################
#Define loss, metrics and optimizer to be used for training

import segmentation_models_3D as sm
dice_loss = sm.losses.DiceLoss()

total_loss = [ 'binary_crossentropy', dice_loss ]

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5),
             tf.keras.callbacks.ModelCheckpoint('best_model.h5',verbose=1,save_best_only=True)
             ]

LR = 0.0001
optim = keras.optimizers.Adam(LR)
#######################################################################
#Fit the model 

steps_per_epoch = len(train_img_list)//batch_size
val_steps_per_epoch = len(val_img_list)//batch_size


from  simple_3d_unet import simple_unet_model

model = simple_unet_model(IMG_HEIGHT=256, 
                          IMG_WIDTH=256, 
                          IMG_DEPTH=256, 
                          IMG_CHANNELS=1)

model.compile(optimizer = optim, loss=total_loss, metrics=metrics)
# print(model.summary())

# print(model.input_shape)
# print(model.output_shape)

history=model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=100,
          verbose=1,
          validation_data=val_img_datagen,
          validation_steps=val_steps_per_epoch,
          callbacks=[callbacks]
          )

model.save('model_tesis.hdf5')
##################################################################


# #plot the training and validation IoU and loss at each epoch
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# plt.plot(epochs, acc, 'y', label='Training accuracy')
# plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
# #################################################
# from keras.models import load_model

# #Load model for prediction or continue training

# #For continuing training....
# #The following gives an error: Unknown loss function: dice_loss_plus_1focal_loss
# #This is because the model does not save loss function and metrics. So to compile and 
# #continue training we need to provide these as custom_objects.
# my_model = load_model('saved_models/brats_3d_100epochs_simple_unet_weighted_dice.hdf5')

# #So let us add the loss as custom object... but the following throws another error...
# #Unknown metric function: iou_score
# my_model = load_model('saved_models/brats_3d_100epochs_simple_unet_weighted_dice.hdf5', 
#                       custom_objects={'dice_loss_plus_1focal_loss': total_loss})

# #Now, let us add the iou_score function we used during our initial training
# my_model = load_model('saved_models/brats_3d_100epochs_simple_unet_weighted_dice.hdf5', 
#                       custom_objects={'dice_loss_plus_1focal_loss': total_loss,
#                                       'iou_score':sm.metrics.IOUScore(threshold=0.5)})

# #Now all set to continue the training process. 
# history2=my_model.fit(train_img_datagen,
#           steps_per_epoch=steps_per_epoch,
#           epochs=1,
#           verbose=1,
#           validation_data=val_img_datagen,
#           validation_steps=val_steps_per_epoch,
#           )
# #################################################

# #For predictions you do not need to compile the model, so ...
# my_model = load_model('saved_models/brats_3d_100epochs_simple_unet_weighted_dice.hdf5', 
#                       compile=False)


# #Verify IoU on a batch of images from the test dataset
# #Using built in keras function for IoU
# #Only works on TF > 2.0
# from keras.metrics import MeanIoU

# batch_size=8 #Check IoU for a batch of images
# test_img_datagen = imageLoader(val_img_dir, val_img_list, 
#                                 val_mask_dir, val_mask_list, batch_size)

# #Verify generator.... In python 3 next() is renamed as __next__()
# test_image_batch, test_mask_batch = test_img_datagen.__next__()

# test_pred_batch = my_model.predict(test_image_batch)

# n_classes = 4
# IOU_keras = MeanIoU(num_classes=n_classes)
# print("Mean IoU =", IOU_keras.result().numpy())

# #############################################
# #Predict on a few test images, one at a time
# #Try images: 
# img_num = 82

# test_img = np.load("BraTS2020_TrainingData/input_data_128/val/images/image_"+str(img_num)+".npy")

# test_mask = np.load("BraTS2020_TrainingData/input_data_128/val/masks/mask_"+str(img_num)+".npy")

# test_img_input = np.expand_dims(test_img, axis=0)
# test_prediction = my_model.predict(test_img_input)



# #Plot individual slices from test predictions for verification
# from matplotlib import pyplot as plt
# import random

# n_slice = 55
# plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img[:,:,n_slice,1], cmap='gray')
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(test_mask[:,:,n_slice])
# plt.subplot(233)
# plt.title('Prediction on test image')
# plt.imshow(test_prediction[:,:, n_slice])
# plt.show()

############################################################

