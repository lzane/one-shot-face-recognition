import cv2
import face
import utils
from test import Face_test
import numpy as np

video_capture = cv2.VideoCapture(0)
# video_capture = cv2.VideoCapture('/Users/zane/Movies/video/告白气球/advertising ballon.mp4')

training_image = utils.load_image('./fig/fig1.jpeg')
training_encodes = face.detect_face_and_encode(training_image)['encoded_faces']
training_label = ['et', 'zane']

Test = Face_test({'labels':training_label,'encodes':training_encodes})

PROCESS_FRAME_RATE = 2
SCALE_FRAME = 2
frame_cnt = 0
encoded_faces = []
recognize_result = {}

while True:

    ret, frame = video_capture.read()

    if frame_cnt == 0:
        # begin process frame
        small_frame = cv2.resize(frame, (0, 0), fx=1/SCALE_FRAME, fy=1/SCALE_FRAME)

        # find all faces and encode
        detect_result = face.detect_face_and_encode(small_frame)
        encoded_faces = detect_result['encoded_faces']
        # recognize all faces
        recognize_result = Test.predict_with_encode_faces(encoded_faces,0.55)
        # print(recognize_result)

    # count the frames
    frame_cnt = frame_cnt + 1 if frame_cnt < PROCESS_FRAME_RATE - 1 else 0

    # display the results
    for rect, name in zip(detect_result['detected_faces'], recognize_result):
        top = rect.top()*SCALE_FRAME
        bottom = rect.bottom()*SCALE_FRAME
        left = rect.left()*SCALE_FRAME
        right = rect.right()*SCALE_FRAME
        
        label = ''
        if name['posibility'].size:
            target_index = np.argmin(name['posibility'])
            target_label = name['label'][target_index]
            target_distance = name['posibility'][target_index]
            label = target_label + ' : ' + str(target_distance.round(2))
        else :
            label = 'Unknown'
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()