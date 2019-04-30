## Project: `Recognition and Classification of Handwritten Math Symbols`

# Project Overview

The current project is a deep learning-based web application that aims to classify mathematical symbols using Convolutional Neural Network. The user will write a character and as an output, the application should predict and classify the input from the 82 classes available in the dataset. These classes represent the different math symbols, predefined functions, digits and Latin alphanumeric symbols.

# Model

- 3 Convolutional Layers
- Activation function : Relu
- 3 Pooling Layers
- 12 epochs
- Batch size : 128
- Optimizer : Adam
- Accuracy : 91.5%


# Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [scikit-learn](http://scikit-learn.org/0.17/install.html)
- [TensorFlow](http://tensorflow.org)
- [matplotlib](https://matplotlib.org/)
- [Keras](https://keras.io/)
- [opencv_python](https://pypi.org/project/opencv-python/)

to install them, run the following command:
```bash
cd Project
pip install -r requirements.txt
```

You will also need to have [Jupyter Notebook](https://jupyter.org/) installed to run and execute a notebook.



If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included.


### Code
- [`myModel/model.ipynb`](https://github.com/Otman404/Mathematical_Expressions_Recognition/blob/master/Project/myModel/model.ipynb) : Model architecture.
- [`train.ipynb`](https://github.com/Otman404/Mathematical_Expressions_Recognition/blob/master/Project/train.ipynb) : Training the model and getting the serialized model + weights.
- [`classifier.ipynb`](https://github.com/Otman404/Mathematical_Expressions_Recognition/blob/master/Project/classifier.ipynb) : Feeding the model a new image, gets resized to 45x45 px to get a prediction.
Better use it like this : 
```bash
python classifier.py --labelbin labels.pickle --model model.model --image your_image.png
```
Or use  directly the [`deployed`](https://github.com/Otman404/deployed_math_symbols_classification_flask) version.

### Data

While no data is directly provided with the project, you will be required to download and use the [Handwritten Math Symbols dataset](https://www.kaggle.com/xainano/handwrittenmathsymbols).

### Deployment
You will find the deployment in [this repository](https://github.com/Otman404/deployed_math_symbols_classification_flask)
