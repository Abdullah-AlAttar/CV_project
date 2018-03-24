
import cv2
import numpy as np
from save_features import pickle_keypoints, unpickle_keypoints
import pickle
cap = cv2.VideoCapture(0)
scaling_factor = 0.5
roi_offset = 150
history = 100
# Create the background subtractor object
sift = cv2.xfeatures2d.SIFT_create()
# ret, frame = cap.read()


class HandFeature:

    def __init__(self, dest, kp):
        self.dest = dest
        self.kp = kp


orb = cv2.ORB_create()
saved = False
text = ["Openend Hand", "Closed hand"]
while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply the background subtraction model to the input frame
    # Convert from grayscale to 3-channel RGB

    keypoints, descriptors = [], []

    # keypoints = sift.detect(roi, None)
    # print(len(kp), len(dest))

    x_start, x_end, y_start, y_end = frame.shape[1] - \
        roi_offset, frame.shape[1], frame.shape[0] - roi_offset, frame.shape[0]

    roi = frame[y_start:y_end, x_start:x_end, :]

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    # ret, thresh = cv2.threshold(blur, 95, 255, cv2.THRESH_BINARY)

    kp, dest = sift.detectAndCompute(roi_gray, None)
    # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    for keypoint, descriptor in zip(kp, dest):
        x, y = keypoint.pt
        if x_start <= x <= x_end and y_start <= y <= y_end:
            keypoints.append(keypoint)
            descriptors.append(descriptor)

    # print(x_start, x_end, y_start, y_end)

    roi = cv2.drawKeypoints(
        roi_gray, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=None)
    # keypoints = orb.detect(thresh, None)
    # keypoints, descriptors = orb.compute(thresh, keypoints)
    # cv2.drawKeypoints(roi, kp, roi, color=(186, 85, 211))
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    cv2.putText(frame, text[0], (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255))

    cv2.imshow('Frame', frame)
    cv2.imshow('roi', roi)
    # cv2.imshow('thres', thresh)
    # cv2.imshow('mask1', mask1)
    c = cv2.waitKey(1)
    if c == 27:
        break
    if c == ord('s'):
        # cv2.imwrite('hand.jpg', roi)
        if not saved:
            openHand = HandFeature(dest, kp)

            saved = True
            text = text[::-1]
        else:
            closeHand = HandFeature(dest, kp)
            break
    if c == ord('c'):
        cv2.imwrite('roisift.png', roi)
# print(openHand.dest, openHand.kp)

# print([i.pt for i in openHand.kp[:5]])
print(openHand.dest[:5])
pickle.dump(pickle_keypoints(openHand.kp, openHand.dest),
            open("openHand.p", "wb"))
pickle.dump(pickle_keypoints(closeHand.kp, closeHand.dest),
            open("closeHand.p", "wb"))
cap.release()
cv2.destroyAllWindows()
