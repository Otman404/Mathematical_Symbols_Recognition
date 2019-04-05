# Deep Learning
## Project: `Recognition and Classification of Handwritten Math Symbols`

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [scikit-learn](http://scikit-learn.org/0.17/install.html)
- [TensorFlow](http://tensorflow.org)
- [matplotlib](https://matplotlib.org/)
- [Keras](https://keras.io/)
- [opencv_python](https://pypi.org/project/opencv-python/)

to install them, run the following command:
```bash
pip install -r requirements.txt
```

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html).



If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included.


### Code
- [`myModel/model.ipynb`](https://github.com/Otman404/Mathematical_Expressions_Recognition/blob/master/Project/myModel/model.ipynb) : Model architecture.
- [`train.ipynb`](https://github.com/Otman404/Mathematical_Expressions_Recognition/blob/master/Project/train.ipynb) : Training the model and getting the serialized model + weights.
- [`classifier.ipynb`](https://github.com/Otman404/Mathematical_Expressions_Recognition/blob/master/Project/classifier.ipynb) : Feeding the model a new image, gets resized to 45x45 px to get a prediction.
Better use it like this : 
```bash
python classifier.py --labelbin mlb.pickle --model model.model --image your_image.png
```

### Data

While no data is directly provided with the project, you will be required to download and use the [Handwritten Math Symbols](https://www.kaggle.com/xainano/handwrittenmathsymbols).