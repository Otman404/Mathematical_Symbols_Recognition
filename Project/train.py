
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
import pydot
import numpy as np
import argparse
import random
import pickle
import cv2
import os

#%%
# construct the argument parse and parse the arguments
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
img_dim = (45,45,3)
EPOCHS = 12
train_data_dir = 'splited_dataset/train'
test_data_dir = 'splited_dataset/test'
BS = 100
LR = 1e-3
labels = []
train_samples_nbr = file_count = sum(len(files) for _, _, files in os.walk(r'splited_dataset/train'))
test_samples_nbr = file_count = sum(len(files) for _, _, files in os.walk(r'splited_dataset/test'))

#%%
nbr_of_pictures = []

labels = os.listdir("data/extracted_images")

for _, _, files in os.walk(r'data/extracted_images'):
    nbr_of_pictures.append(len(files))

nbr_of_pictures=nbr_of_pictures[1:]

# 82 nbrOfClasses
print("Number of samples of every class ...")
for i in range(82):  
    print(labels[i]," : ",nbr_of_pictures[i])
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
opt = Adam(lr=LR, decay=LR / EPOCHS)
#opt = RMSprop(lr=LR, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy']) #binary_crossentropy training 99% acc 

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
plot_model(model, to_file='model_plot_final.png', show_shapes=True, show_layer_names=True);

#%%
# save the model to disk
print("[INFO] serializing network...")
#model.save(args["model"])
model.save("trained_model_final.model")
model.save_weights("weights_final.h5")
#save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
# f = open(args["labelbin"], "wb")
f = open("mlb.pickle", "wb")
f.write(pickle.dumps(mlb))
f.close()


#%%
probabilities = model.predict_generator(validation_generator,2000)

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
plt.savefig('model_accuary_final.png')
plt.show()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('model_loss_final.png')
plt.show()
