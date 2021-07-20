## To run the training:

We are working with python3.7. Other Python version should work normally.

Just run the train.py script:
```
python3 train.py
```
After training finishes, the model will be saved at `model.model` file, its weights are at `weights_new.h5` file and the labels are saved at the file `labels.pickle`.

To infer an image:
```
python3 classifier.py --labelbin labels.pickle --model model.model --image <path/to/image>
```

