import sys
import numpy as np

import face
import utils

from recognize import distance


class Face_test:
    def __init__(self, model):
        self.model = model
        self.labels = np.array(model['labels'])
        self.encodes = np.array(model['encodes'])

    def predict_with_encode_faces(self, encode_faces, tolerance=0.6):
        if not encode_faces:
            return []
        dis = distance(self.encodes, encode_faces)
        maybe_result = [np.where(d < tolerance) for d in dis]
        return [{'label': self.labels[res], 'posibility':dis[res]} for res, dis in zip(maybe_result, dis)]

    def predict_with_image(self, image, tolerance=0.6):
        # find all faces and encode
        detect_result = face.detect_face_and_encode(image)
        encoded_faces = detect_result['encoded_faces']
        recognize_result = []

        if encoded_faces:
            # for encoded_face in encoded_faces:
            recognize_result = self.predict_with_encode_faces(tolerance, 1)
        
        return {'recognize_result':recognize_result,
                'detect_result':detect_result}
        
