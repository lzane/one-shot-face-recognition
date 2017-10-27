import sys
import numpy as np

import myface.face as face
import myface.utils.utils as utils

image = utils.load_image('./fig/fig1.jpeg')
res = face.detect_face_and_encode(image)
print(res['encoded_faces'].__len__)

