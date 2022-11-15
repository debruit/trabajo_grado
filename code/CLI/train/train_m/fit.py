## =========================================================================
## @author Juan Sebastián Ruiz Bulla (ruizju@javeriana.edu.co)
## @author David Alejandro Castillo Chíquiza (castillo_da@javeriana.edu.co)
## @author Oscar David Falla Pulido (falla_o@javeriana.edu.co)
## =========================================================================

import os
import matplotlib.pyplot as plt 
import tensorflow as tf
from datetime import datetime
import segmentation_models_3D as sm

from train.train_m import unet
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

    train_generator = generator.imageLoader(
        train_img_folder, train_img_arr, train_mask_folder, train_mask_arr, batch_size)
    validation_generator = generator.imageLoader(
        val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size)

    dice_loss = sm.losses.DiceLoss()

    metrics = [sm.metrics.FScore(), 'accuracy']

    LR = 0.00001
    optim = tf.keras.optimizers.Adam(LR)
    #######################################################################
    # Fit the model

    steps_per_epoch = len(train_img_arr)//batch_size
    val_steps_per_epoch = len(val_img_list)//batch_size

    print('Loading Unet...')

    model = unet.unet(IMG_HEIGHT=160, IMG_WIDTH=96,
                      IMG_DEPTH=128, IMG_CHANNELS=1)

    callback = [tf.keras.callbacks.ModelCheckpoint(output_model, verbose=1, save_best_only=True),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100), ]

    model.compile(optimizer=optim, loss=dice_loss, metrics=metrics)

    print('Fitting model...')

    ini = datetime.now()
    time_ini = (ini.hour, ini.minute)

    history = model.fit(train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=500,
                        verbose=1,
                        validation_data=validation_generator,
                        validation_steps=val_steps_per_epoch,
                        callbacks=callback
                        )

    fin = datetime.now()
    time_fin = (fin.hour, fin.minute)

    time_model = tuple(map(lambda i, j: i - j, time_fin, time_ini))

    print('Training time: ', time_model)

    print('Model saved in: ', output_model)

    print('Plotting results...')

    loss = history.history['loss']


    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    dice = history.history['f1-score']
    val_dice = history.history['val_f1-score']

    plt.plot(epochs, dice, 'y', label='Training dice')
    plt.plot(epochs, val_dice, 'r', label='Validation dice')
    plt.title('Training and validation dice coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.legend()
    plt.show()
