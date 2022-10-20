import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import segmentation_models_3D as sm

import unet
from preprocessing import generator


def fit_model(imgs_folder, mask_folder, imgs_val_folder, mask_val_folder, output_model):

    train_img_folder = imgs_folder
    train_mask_folder = mask_folder

    val_img_dir = imgs_val_folder
    val_mask_dir = mask_val_folder

    train_img_arr = sorted(os.listdir(train_img_folder))
    train_mask_arr = sorted(os.listdir(train_mask_folder))

    val_img_list = sorted(os.listdir(val_img_dir))
    val_mask_list = sorted(os.listdir(val_mask_dir))


    batch_size = 2

    train_generator = generator.image_gen(train_img_folder, train_img_arr, train_mask_folder, train_mask_arr, batch_size)
    validation_generator = generator.image_gen(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size)


    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 34.5])) 

    metrics = [sm.metrics.FScore(),'accuracy']

    LR = 0.00001
    optim = tf.keras.optimizers.Adam(LR)
    #######################################################################
    #Fit the model 

    steps_per_epoch = len(train_img_arr)//batch_size
    val_steps_per_epoch = len(val_img_list)//batch_size


    model = unet.unet(IMG_HEIGHT=160, IMG_WIDTH=96, IMG_DEPTH=128, IMG_CHANNELS=1)



    callback = [tf.keras.callbacks.ModelCheckpoint(output_model,verbose=1,save_best_only=True)]


    model.compile(optimizer = optim, loss=dice_loss, metrics=metrics)




    ini = datetime.now()
    time_ini = (ini.hour, ini.minute)


    history=model.fit(train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=500,
            verbose=1,
            validation_data=validation_generator,
            validation_steps=val_steps_per_epoch,
            callbacks = callback
            )

    fin = datetime.now()
    time_fin = (fin.hour, fin.minute)

    print(time_ini,time_fin)