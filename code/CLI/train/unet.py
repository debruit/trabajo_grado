
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, Dropout, BatchNormalization

kernel_initializer =  'he_uniform' 
# regularizer = tf.keras.regularizers.L1(0.01)


################################################################
def unet(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    
    s = inputs

    #Contraction path
    c1 = Conv3D(32, (3, 3, 3), activation='sigmoid', kernel_initializer=kernel_initializer, padding='same', input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH,IMG_CHANNELS))(s)
    c1 = Dropout(0.4)(c1)
    c1 = Conv3D(32, (3, 3, 3), activation='sigmoid', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)
    
    c2 = Conv3D(64, (3, 3, 3), activation='sigmoid', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = Dropout(0.4)(c2)
    c2 = Conv3D(64, (3, 3, 3), activation='sigmoid', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)
     
    c3 = Conv3D(128, (3, 3, 3), activation='sigmoid', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = Dropout(0.4)(c3)
    c3 = Conv3D(128, (3, 3, 3), activation='sigmoid', kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)
     
    c4 = Conv3D(256, (3, 3, 3), activation='sigmoid', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = Dropout(0.6)(c4)
    c4 = Conv3D(256, (3, 3, 3), activation='sigmoid', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
     
    c5 = Conv3D(512, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = Dropout(0.6)(c5)
    c5 = Conv3D(512, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
    # Expansive path 
    u6 = Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = Dropout(0.6)(c6)
    c6 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
     
    u7 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = Dropout(0.4)(c7)
    c7 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
     
    u8 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = Dropout(0.4)(c8)
    c8 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
     
    u9 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = Dropout(0.4)(c9)
    c9 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
     
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

#Test if everything is working ok. 
model = unet(160, 96, 128, 1)

print(model.summary())
print(model.input_shape)
print(model.output_shape)