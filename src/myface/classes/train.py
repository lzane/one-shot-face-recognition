import numpy as np
from skimage import io
import os

from .. import face
from ..utils import utils


class Face_train:
    def __init__(self):
        self.model = {
            'labels': [],
            'encodes': [],
            'label_map': {},
        }
        self.__label_cnt = 0

    def train(self, dir_path):
        """
        start training model with directory path
            :param self: 
            :param dir_path: 
        """   
        
        # load images with labels
        images = []
        labels = []
        label_directorys = next(os.walk(dir_path))[1]
        for label in label_directorys:
            label_directory = os.path.join(dir_path, label)
            images_path = [os.path.join(label_directory, image_name)
                           for image_name in os.listdir(label_directory)]
            images_for_label = [io.imread(path, img_num=0).astype(
                'uint8') for path in images_path]
            images += images_for_label
            # map label
            self.model['label_map'][self.__label_cnt] = label
            labels += [self.__label_cnt for _ in range(len(images_path))]
            self.__label_cnt += 1

        encode_result = [face.detect_face_and_encode(
            image)['encoded_faces'] for image in images]

        # remove detect with no face
        encode_result_remove_empty = list(
            filter(lambda x: len(x[0]) != 0, zip(encode_result,labels)))
        encodes_result,labels_result = zip(*encode_result_remove_empty)

        # only take the first face detected
        encodes_result = [encode[0] for encode in encodes_result]

        self.model['labels'] += list(labels_result)
        self.model['encodes'] += list(encodes_result)

    def get_model(self):
        return self.model

    



