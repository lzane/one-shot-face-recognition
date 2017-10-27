import numpy as np 
from myface.classes.train import Face_train
from myface.utils import utils

MODEL_PATH = '../model/'

# create a Face_train object
train = Face_train()
training_set_dir = '../fig/'

# start train
train.train(training_set_dir)

# get trained model
model = train.get_model()

# save it into file
name = 'A1'
utils.save_model(model,MODEL_PATH,name)