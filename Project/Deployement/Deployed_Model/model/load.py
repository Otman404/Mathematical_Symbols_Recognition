import numpy as np
import keras.models
from scipy.misc import imread, imresize,imshow
import tensorflow as tf
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import Adam

def init():
    classes = 82
    LR = 1e-3
    EPOCHS = 12
    img_rows, img_cols = 45, 45
    input_shape = (img_rows, img_cols, 3)



    model = Sequential()

    model.add(Conv2D(32, (2, 2), input_shape = input_shape)) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
    
    model.add(Conv2D(32, (2, 2))) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
    
    model.add(Conv2D(64, (2, 2))) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
    
    model.add(Flatten()) 
    model.add(Dense(64)) 
    model.add(Activation('relu')) 
    model.add(Dropout(0.5)) 
    model.add(Dense(classes)) 
    model.add(Activation('softmax'))
    
    #load woeights into new model
    model.load_weights("weights.h5")
    print("Loaded Model from disk")

    #compile and evaluate loaded model
    opt = Adam(lr=LR, decay=LR / EPOCHS)

    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    #print('loss:', loss)
    #print('accuracy:', accuracy)
    graph = tf.get_default_graph()

    return model, graph