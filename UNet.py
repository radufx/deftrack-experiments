import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.losses import *

class UNet(Model):
    def convBlock(self, input, filters, kernel, kernel_init='he_normal', act='relu', transpose=False):
        if transpose == False:
            conv = Conv2D(filters, kernel, padding = 'same', kernel_initializer = kernel_init)(input)
        else:
            conv = Conv2DTranspose(filters, kernel, padding = 'same', kernel_initializer = kernel_init)(input)

        conv = Activation(act)(conv)
        return conv

    def __init__(self, inputs):
        conv1 = self.convBlock(inputs, 64, 3)
        conv1 = self.convBlock(conv1, 64, 3)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.convBlock(pool1, 128, 3)
        conv2 = self.convBlock(conv2, 128, 3)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        #drop2 = Dropout(drop_rate)(pool2)

        conv3 = self.convBlock(pool2, 256, 3)
        conv3 = self.convBlock(conv3, 256, 3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        #drop3 = Dropout(drop_rate)(pool3)

        conv4 = self.convBlock(pool3, 512, 3)
        conv4 = self.convBlock(conv4, 512, 3)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        #drop4 = Dropout(drop_rate)(pool4)

        conv5 = self.convBlock(pool4, 1024, 3)
        conv5 = self.convBlock(conv5, 1024, 3)

        ## Expansion phase
        up6 = (Conv2DTranspose(512, kernel_size=2, strides=2, kernel_initializer='he_normal')(conv5))
        merge6 = concatenate([conv4,up6])
        conv6 = self.convBlock(merge6, 512, 3)
        conv6 = self.convBlock(conv6, 512, 3)
        #conv6 = Dropout(drop_rate)(conv6)

        up7 = (Conv2DTranspose(256, kernel_size=2, strides=2, kernel_initializer='he_normal')(conv6))
        merge7 = concatenate([conv3,up7])
        conv7 = self.convBlock(merge7, 256, 3)
        conv7 = self.convBlock(conv7, 256, 3)
        #conv7 = Dropout(drop_rate)(conv7)

        up8 = (Conv2DTranspose(128, kernel_size=2, strides=2, kernel_initializer='he_normal')(conv7))
        merge8 = concatenate([conv2,up8])
        conv8 = self.convBlock(merge8, 128, 3)
        conv8 = self.convBlock(conv8, 128, 3)
        #conv8 = Dropout(drop_rate)(conv8)

        up9 = (Conv2DTranspose(64, kernel_size=2, strides=2, kernel_initializer='he_normal')(conv8))
        merge9 = concatenate([conv1,up9])
        conv9 = self.convBlock(merge9, 64, 3)
        conv9 = self.convBlock(conv9, 64, 3)

        # Output layer
        outputs = self.convBlock(conv9, 1, 1, act='sigmoid')

        super().__init__(inputs, outputs)
    
    