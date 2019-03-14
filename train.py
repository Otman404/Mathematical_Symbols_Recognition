
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
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, # --dataset : The path to our dataset.
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True, # --model : The path to our output serialized Keras model.
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True, # --labelbin : The path to our output multi-label binarizer object.
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png", # --plot : The path to our output plot of training loss and accuracy.
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())


#%%
img_dim = (28,28,3)
EPOCHS = 12
train_data_dir = 'splited_dataset/train'
test_data_dir = 'splited_dataset/test'
BS = 128
LR = 1e-3
labels = []
train_samples_nbr = file_count = sum(len(files) for _, _, files in os.walk(r'splited_dataset/train'))
test_samples_nbr = file_count = sum(len(files) for _, _, files in os.walk(r'splited_dataset/test'))

#%%
if K.image_data_format() == 'channels_first':
    input_shape = (img_dim[2], img_dim[0], img_dim[1])
else:
    input_shape = (img_dim[0], img_dim[1], img_dim[2])


#%%
# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
#imagePaths = sorted(list(paths.list_images(args["dataset"])))
imagePaths = sorted(list(paths.list_images(train_data_dir)))
#imagePaths = sorted(list(paths.list_images("data/extracted_images")))
random.seed(42)
random.shuffle(imagePaths)


#%%
l = label = [ item for item in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, item)) ]
labels.append(l)
print("Classes : ",labels[0]) #labels


#%%
# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
 
# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))


#%%
print("[INFO] compiling model...")
model = VGGNet.build(
	width=img_dim[1], height=img_dim[0],
	depth=img_dim[2], classes=82,
    activFct="softmax") #for multi-class classification

#%%
#opt = Adam(lr=LR, decay=LR / EPOCHS)
opt = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy']) #binary_crossentrpy 99% acc 


#%%
# data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

#%%
# data augmentation for testing
test_datagen = ImageDataGenerator(rescale=1. / 255)

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

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples_nbr // BS,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=test_samples_nbr // BS)

model.summary();
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True);

#%%
# save the model to disk
print("[INFO] serializing network...")
#model.save(args["model"])
model.save("trained_model.model")
model.save_weights("weights.h5")
#save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
# f = open(args["labelbin"], "wb")
f = open("mlb.pickle", "wb")
f.write(pickle.dumps(mlb))
f.close()


#%%
probabilities = model.predict_generator(validation_generator,2000)

scores = model.evaluate_generator(validation_generator,test_samples_nbr) #1514 testing images
print("Accuracy = ", scores[1])

#%%


fig1, ax_acc = plt.subplots()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model - Accuracy')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.savefig("plotting.png")
