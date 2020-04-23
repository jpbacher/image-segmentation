from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate)


def Unet(img_col, img_row, batch_size):

    inputs = Input(shape=(img_col, img_row, 3), batch_size=batch_size)
    conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool1)
    conv2 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), padding='same', activation='relu')(pool2)
    conv3 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool3)
    conv4 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), padding='same', activation='relu')(pool4)

    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5),
                      conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), padding='same', activation='relu')(up6)
    conv6 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv6)

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6),
                       conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), padding='same', activation='relu')(up7)
    conv7 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv7)

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7),
                       conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), padding='same', activation='relu')(up8)
    conv8 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv8)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8),
                       conv1])
    conv9 = Conv2D(64, (3, 3), padding='same', activation='relu')(up9)
    conv9 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv9)

    conv10 = Conv2D(2, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model
