from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import *
from keras.layers.core import *
from keras.losses import *
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D

class SegNet(Model):
    def convBlock(self, input, size, kernel = 3):
        conv1 = Convolution2D(size, (kernel, kernel), padding="same")(input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)
        conv2 = Convolution2D(size, (kernel, kernel), padding="same")(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)

        return conv2
    
    def convBlockExtended(self, input, size, kernel):
        conv1 = self.convBlock(input, size, kernel)
        conv2 = Convolution2D(size, (kernel, kernel), padding="same")(conv1)
        conv2 = BatchNormalization()(conv1)
        conv2 = Activation("relu")(conv1)

        return conv2
    
    def downConvBlock(self, input, initial_size, final_size, kernel):
        conv1 = Convolution2D(initial_size, (kernel, kernel), padding="same")(input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)

        conv2 = Convolution2D(final_size, (kernel, kernel), padding="same")(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)

        return conv2
    
    def downConvBlockExtended(self, input, initial_size, final_size, kernel):
        conv1 = Convolution2D(initial_size, (kernel, kernel), padding="same")(input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)

        return self.downConvBlock(conv1, initial_size, final_size, kernel)
    

    def __init__(self, input_shape, n_labels = 2, kernel = 3, pool_size=(2, 2)):
        inputs = Input(input_shape)
        
        # encoder
        conv1 = self.convBlock(inputs, 64, kernel)
        pool1, mask1 = MaxPoolingWithArgmax2D(pool_size)(conv1)

        conv2 = self.convBlock(pool1, 128, kernel)
        pool2, mask2 = MaxPoolingWithArgmax2D(pool_size)(conv2)

        conv3 = self.convBlockExtended(pool2, 256, kernel)
        pool3, mask3 = MaxPoolingWithArgmax2D(pool_size)(conv3)

        conv4 = self.convBlockExtended(pool3, 512, kernel)
        pool4, mask4 = MaxPoolingWithArgmax2D(pool_size)(conv4)

        conv5 = self.convBlockExtended(pool4, 512, kernel)
        pool5, mask5 = MaxPoolingWithArgmax2D(pool_size)(conv5)

        print("Build encoder done..")


        # decoder
        unpool1 = MaxUnpooling2D(pool_size)([pool5, mask5])

        conv6 = self.downConvBlockExtended(unpool1, 512, 512, kernel)
        unpool2 = MaxUnpooling2D(pool_size)([conv6, mask4])

        conv7 = self.downConvBlockExtended(unpool2, 512, 256, kernel)
        unpool3 = MaxUnpooling2D(pool_size)([conv7, mask3])

        conv8 = self.downConvBlockExtended(unpool3, 256, 128, kernel)
        unpool4 = MaxUnpooling2D(pool_size)([conv8, mask2])

        conv9 = self.downConvBlock(unpool4, 128, 64, kernel)
        unpool5 = MaxUnpooling2D(pool_size)([conv9, mask1])


        conv10 = Convolution2D(64, (kernel, kernel), padding="same")(unpool5)
        conv10 = BatchNormalization()(conv10)
        conv10 = Activation("relu")(conv10)

        conv11 = Convolution2D(1, (1, 1), padding="valid")(conv10)
        conv11 = BatchNormalization()(conv11)

        outputs = Activation('sigmoid')(conv11)
        print("Build decoder done..")

        super().__init__(inputs, outputs)
    
    