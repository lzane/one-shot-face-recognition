from skimage import io
import pickle
import os
import time
import datetime

def load_image(path):
    return io.imread(path,img_num=0).astype('uint8')


def save_model(model,path,name=None):
    if not name:
        name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(path,name+'.pickle'), 'wb+') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(path,name):
    with open(os.path.join(path,name+'.pickle'), 'rb') as handle:
        model = pickle.load(handle)
        return model
