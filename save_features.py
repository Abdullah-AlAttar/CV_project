
import cv2
import numpy as np


def pickle_keypoints(keypoints, descriptors, rect):
    i = 0
    temp_array = []
    temp_array.append(rect)
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id, descriptors[i])
        i += 1
        temp_array.append(temp)
    return temp_array


def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    rect = None
    first = True
    for point in array:
        if first:
            rect = point
            first = False
            continue
        temp_feature = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1],
                                    _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors),rect
