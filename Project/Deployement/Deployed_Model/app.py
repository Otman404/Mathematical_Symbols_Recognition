from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import base64
import cv2
from keras.preprocessing.image import img_to_array
import pickle
from keras.models import load_model

import sys 
import os
sys.path.append(os.path.abspath("./model"))
from load import *

app = Flask(__name__)
global model, graph
model, graph = init()
    
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['GET','POST'])
def predict():
    # get data from drawing canvas and save as image
    parseImage(request.get_data())

    # read parsed image back in 8-bit, black and white mode (L)
#     x = imread('output.png', mode='RGB')
#     x = np.invert(x)
#     x = imresize(x,(45,45))

    image = cv2.imread('output.png')
    image = cv2.resize(image, (45, 45))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    mlb = pickle.loads(open("mlb.pickle", "rb").read())
    # reshape image data for use in neural network
#     x = x.reshape(1,45,45,1)
    with graph.as_default():
        out = model.predict(image)[0]
        print(out)
        # print(np.argmax(out, axis=1))
        print (' '.join(mlb.classes_[out.argmax(axis=-1)]))

        # response = np.array_str(np.argmax(out, axis=1))
        response = ' '.join(mlb.classes_[out.argmax(axis=-1)])
        return response 
    
def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)