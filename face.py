import dlib
import face_recognition
import face_recognition_models
import time

# HOG face detector using the built-in dlib class
FACE_DETECTOR = dlib.get_frontal_face_detector()

# download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
PREDICTOR_MODEL = "shape_predictor_68_face_landmarks.dat"
FACE_POSE_PREDICTOR = dlib.shape_predictor(PREDICTOR_MODEL)

FACE_RECOGNITION_MODEL = face_recognition_models.face_recognition_model_location()
FACE_ENCODER = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL)


def _detect_faces(image, upsample_num_times):
    """
    private function for detect faces using dlib
        :param image: 
        :param upsample_num_times: 
    """
    # Run the HOG face detector on the image data.
    return FACE_DETECTOR(image, upsample_num_times)


def detect_faces(image, upsample_num_times=1):
    return _detect_faces(image, upsample_num_times)


def crop_faces(image, face_rects):
    return [image[f.top():f.bottom(), f.left():f.right()] for i, f in enumerate(face_rects)]


def detect_faces_and_crop(image, upsample_num_times=1):
    """
    detect faces and crop the image into faces which have been detected. 
        -> list
        :param image: 
        :param upsample_num_times: 
    """
    faces = detect_faces(image, upsample_num_times)
    return crop_faces(image, faces)


def _detect_face_landmarks(image, face_rect):
    return FACE_POSE_PREDICTOR(image, face_rect)

def detect_face_landmarks(image, face_rect=None):
    """
    detect face landmarks,
    if face_rect is None, the face_rect is the same size as image
        -> object
        :param image: 
        :param face_rect: where the face is
    """
    if(face_rect == None):
        face_rect = dlib.rectangle(0, 0, image.shape[0], image.shape[1])
    return _detect_face_landmarks(image, face_rect)


def _encode_face(image, detect_marks):
    return FACE_ENCODER.compute_face_descriptor(image, detect_marks)


def encode_face(image, detect_marks=None):
    """
    encode face into 128 dims vector with image as input
        :param image: 
        :param detect_marks=None: 
    """
    if(detect_marks == None):
        detect_marks = detect_face_landmarks(image)
    return _encode_face(image, detect_marks)


def detect_face_and_encode(image):
    detected_faces = detect_faces(image)
    croped_faces = crop_faces(image, detected_faces)
    encoded_faces = []
    landmarked_faces = []
    for face in croped_faces:
        landmarks = detect_face_landmarks(face)
        encode = encode_face(face, landmarks)
        landmarked_faces.append(landmarks)
        encoded_faces.append(encode)
    return {'detected_faces': detected_faces,
            'croped_faces': croped_faces,
            'encoded_faces': encoded_faces,
            'landmarked_faces': landmarked_faces,
            'image': image}
