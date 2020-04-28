from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Activation, BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, concatenate
)


def unet(input_size=(256, 256, 3), num_classes=1, pretrained_weights=None):

    inputs = Input(shape=input_size)

    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    bnorm1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1)
    bnorm1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bnorm1)

    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool1)
    bnorm2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv2)
    bnorm2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bnorm2)

    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool2)
    bnorm3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(bnorm3)
    bnorm3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bnorm3)

    conv4 = Conv2D(256, (3, 3), padding='same', activation='relu')(pool3)
    bnorm4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), padding='same', activation='relu')(bnorm4)
    bnorm4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bnorm4)

    center = Conv2D(512, (3, 3), padding='same', activation='relu')(pool4)
    bnorm5 = BatchNormalization()(center)
    center = Conv2D(512, (3, 3), padding='same', activation='relu')(bnorm5)
    bnorm5 = BatchNormalization()(center)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bnorm5),
                      bnorm4], axis=3)
    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu')(up6)
    bnorm6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu')(bnorm6)
    bnorm6 = BatchNormalization()(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bnorm6),
                       bnorm3], axis=3)
    conv7 = Conv2D(128, (3, 3), padding='same', activation='relu')(up7)
    bnorm7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (3, 3), padding='same', activation='relu')(bnorm7)
    bnorm7 = BatchNormalization()(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(bnorm7),
                       conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), padding='same', activation='relu')(up8)
    bnorm8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (3, 3), padding='same', activation='relu')(bnorm8)
    bnorm8 = BatchNormalization()(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(bnorm8),
                       conv1])
    conv9 = Conv2D(32, (3, 3), padding='same', activation='relu')(up9)
    bnorm9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3, 3), padding='same', activation='relu')(bnorm9)
    bnorm9 = BatchNormalization()(conv9)

    convlast = Conv2D(num_classes, (1, 1), activation='sigmoid')(bnorm9)

    model = Model(inputs=[inputs], outputs=[convlast])

    if pretrained_weights:
        model.load_weights(filepath=pretrained_weights)

    return model


def unet_large(input_size=(256, 256, 3), num_classes=1, pretrained_weights=None):
    inputs = Input(shape=input_size)

    down1 = Conv2D(64, (3, 3), padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)

    convlast = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=[inputs], outputs=[convlast])

    if pretrained_weights:
        model.load_weights(filepath=pretrained_weights)

    return model