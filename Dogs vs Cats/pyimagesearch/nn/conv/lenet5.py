from tensorflow.keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class Lenet5:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last"
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channel_first":
            inputShape = (depth, height, width)

        # define the first (and only) CONV => RELU layer
        model.add(Conv2D(6, (5, 5), padding = 'valid',
            input_shape=inputShape))
        model.add(Activation("tanh"))
        model.add(BatchNormalization())
        model.add(AveragePooling2D(pool_size=(2,2)))
        
        model.add(Conv2D(16, (5,5), padding = 'valid'))
        model.add(Activation("tanh"))
        model.add(BatchNormalization())
        model.add(AveragePooling2D(pool_size=(2,2)))
        
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Activation("tanh"))
        model.add(BatchNormalization())
        model.add(Dense(84))
        model.add(Activation("tanh"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecher
        return model

