from ..utils import utils
from .. import face

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_image_and_plot_face_landmarks(path=None, image=None):
    if path != None:
        image = utils.load_image(path)

    res = face.detect_face_and_encode(image)
    plot_crop_and_landmarks(res)
    return res


def plot_rect_around_faces(image, detected_faces):
    # plot the image and detected faces
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image)

    for i, face_rect in enumerate(detected_faces):
        print("- Facess #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i,
                                                                                   face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
        # Draw a box around each face we found
        rect = patches.Rectangle((face_rect.left(), face_rect.top()),
                                 face_rect.width(),
                                 face_rect.height(),
                                 fill=False,
                                 color='red',
                                 lw=3
                                 )
        ax.add_patch(rect)


def plot_crop_and_landmarks(res):
    length = len(res['croped_faces'])
    fig = plt.figure()
    faces_per_row = 4
    rows = int(length / faces_per_row) + 1
    for i in range(length):
        ax = fig.add_subplot(rows, faces_per_row, i + 1)
        ax.imshow(res['croped_faces'][i])
        parts = res['landmarked_faces'][i].parts()
        ax.plot([p.x for p in parts], [p.y for p in parts], 'bo')


def plot_and_predit(tester, image_file_name, tolerance=0.6):
    detect_result = load_image_and_plot_face_landmarks(image_file_name)
    plot_rect_around_faces(detect_result['image'],detect_result['detected_faces'])
    recognize_result = tester.predict_with_encode_faces(
        detect_result['encoded_faces'], tolerance)
    print(recognize_result)
