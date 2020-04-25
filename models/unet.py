from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate


def unet(input_size=(256, 256, 3), num_classes=1, pretrained_weights=None):

    inputs = Input(shape=input_size, batch_size=batch_size)
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
    conv7 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv7)
    bnorm7 = BatchNormalization()(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(bnorm7),
                       conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), padding='same', activation='relu')(up8)
    bnorm8 = BatchNormalization(conv8)
    conv8 = Conv2D(64, (3, 3), padding='same', activation='relu')(bnorm8)
    bnorm8 = BatchNormalization()(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(bnorm8),
                       conv1])
    conv9 = Conv2D(32, (3, 3), padding='same', activation='relu')(up9)
    bnorm9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv9)
    bnorm9 = BatchNormalization()(conv9)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(bnorm9)

    model = Model(inputs=[inputs], outputs=[classify])

    if pretrained_weights:
        model.load_weights(filepath=pretrained_weights)

    return model
