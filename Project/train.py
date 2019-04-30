#%%
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from keras import backend as K
from myModel.model import VGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

#%%
# construct the argument parse and parse the arguments (for command line)
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", # --dataset : The path to our dataset. add required=True if you want
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", # --model : The path to our output serialized Keras model.
	help="path to output model")
ap.add_argument("-l", "--labelbin", # --labelbin : The path to our output multi-label binarizer object.
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png", # --plot : The path to our output plot of training loss and accuracy.
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())


#%%
#Initialize parametres

EPOCHS = 12
BS = 100 #Batch size
LR = 1e-3 #Learning rate 0.001
img_dim = (45,45,3)
train_data_dir = 'splited_dataset/train'
test_data_dir = 'splited_dataset/test'
labels = []
#Nbr of training images
train_samples_nbr  = sum(len(files) for _, _, files in os.walk(r'splited_dataset/train'))
#Nbr of testing images
test_samples_nbr  = sum(len(files) for _, _, files in os.walk(r'splited_dataset/test'))

#%%
# Infos about our Dataset
nbr_of_pictures = []

labels = os.listdir("data/extracted_images")
for _, _, files in os.walk(r'data/extracted_images'):
    nbr_of_pictures.append(len(files))

nbr_of_pictures=nbr_of_pictures[1:]
#print nbr of pictures in every class
print("Number of samples in every class ...")
for i in range(82):  # 82 : Nbr of classes
    print(labels[i]," : ",nbr_of_pictures[i])
#%%
# Checking image data format

if K.image_data_format() == 'channels_first':
    input_shape = (img_dim[2], img_dim[0], img_dim[1])
else:
    input_shape = (img_dim[0], img_dim[1], img_dim[2])

#%%



#%%

print(len(labels)," Classes : ",labels)
labels = np.array(labels)
#%%

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

#%%
# Building the model 
model = VGGNet.build(
	width=img_dim[1], height=img_dim[0],
	depth=img_dim[2], classes=82,
    activFct="softmax") #for multi-class classification
model.summary()
print('Number of layers of our model : ',len(model.layers))
#%%
# Compiling the model 

opt = Adam(lr=LR, decay=LR / EPOCHS)
#opt = RMSprop(lr=LR, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy']) 

#%%
# data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.0,
    zoom_range=0.0,
    featurewise_center=False,# set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0.0,  # randomly rotate images in the range (deg 0 to 180)
    width_shift_range=0.0,  # randomly shift images horizontally
    height_shift_range=0.0,  # randomly shift images vertically
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False
    )

# data augmentation for testing
test_datagen = ImageDataGenerator(rescale=1. / 255)


#%%

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_dim[0], img_dim[1]),
    batch_size=BS,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_dim[0], img_dim[1]),
    batch_size=BS,
    class_mode='categorical')

#%%
# Training
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples_nbr // BS,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=test_samples_nbr // BS)

#%%
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True);

#%%
# save the model to disk
print("Saving the model...")
#model.save(args["model"])
model.save("model.model")
model.save_weights("weights.h5")
#save the multi-label binarizer to disk
print("Saving Labels...")
# f = open(args["labelbin"], "wb")
f = open("labels.pickle", "wb")
f.write(pickle.dumps(mlb))
f.close()


#%%
#probabilities = model.predict_generator(validation_generator,2000)

# Evaluating the model / Get Validation accuracy on sample from validation set
scores = model.evaluate_generator(validation_generator,test_samples_nbr//BS,verbose=1) 
print("Accuracy = ", scores[1])

#%%


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('model_accuary_plot.png')
plt.show()

#%%
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('model_loss_plot.png')
plt.show()

#%%
# Train - Val plot
fig1, ax_acc = plt.subplots()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model - Accuracy')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.savefig("train_val_plot.png")