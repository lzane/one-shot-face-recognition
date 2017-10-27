import numpy as np

def distance(face_list,face_targets):
    face_list = np.array(face_list)
    face_targets = np.array(face_targets)
    row_num_face_list = face_list.shape[0]
    row_num_face_targets = face_targets.shape[0]
    col_num_face_targets = face_targets.shape[1]
    face_targets_for_minus = np.repeat(face_targets,row_num_face_list,axis=0).reshape(row_num_face_targets,row_num_face_list,col_num_face_targets)
    diff = np.power(face_list-face_targets_for_minus,2)
    sum_result = np.sum(diff,axis=2)
    euclidean_distance = np.sqrt(sum_result)
    return euclidean_distance

# def distance2(face_list,face_targets):
#     face_list = np.array(face_list)
#     face_targets = np.array(face_targets)
#     diff = np.power(face_list-face_targets,2)
#     sum_result = np.sum(diff,axis=1)
#     euclidean_distance = np.sqrt(sum_result)
#     return euclidean_distance
