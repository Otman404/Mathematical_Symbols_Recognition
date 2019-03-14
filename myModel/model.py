#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


# In[18]:


class VGGNet:
	@staticmethod
	def build(width, height, depth, classes, activFct="softmax"): #finalAct='softmax' for single-label classification || finalAct='sigmoid' for multi-label classification 
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
 
		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
        # CONV => RELU => POOL
		model.add(Conv2D(32, (3, 3),
			input_shape=inputShape))  #32 nbr of layers
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
        # (CONV => RELU) * 2 => POOL
		model.add(Conv2D(32, (3, 3)))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
 
		# (CONV => RELU) * 2 => POOL
		model.add(Conv2D(64, (3, 3)))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
        # first (and only) set of FC => RELU layers
		model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))
		
 
		# use a *softmax* activation for single-label classification
		# and *sigmoid* activation for multi-label classification
		model.add(Dense(classes))
		model.add(Activation(activFct))
 
		# return the constructed network architecture
		return model


# In[ ]:




