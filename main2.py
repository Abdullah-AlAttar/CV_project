
import cv2
import numpy as np
from save_features import pickle_keypoints, unpickle_keypoints
import pickle
cap = cv2.VideoCapture(0)
scaling_factor = 0.5
roi_offset = 250
history = 100
# Create the background subtractor object
sift = cv2.xfeatures2d.SIFT_create()
# ret, frame = cap.read()


class HandFeature:

    def __init__(self, dest, kp):
        self.dest = dest
        self.kp = kp


saved = False
text = ["Openend Hand", "Closed hand"]
while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply the background subtraction model to the input frame
    # Convert from grayscale to 3-channel RGB
    roi = frame[frame.shape[0] - roi_offset:,
                frame.shape[1] - roi_offset:].copy()

    # keypoints = sift.detect(roi, None)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kp, dest = sift.detectAndCompute(roi, None)
    print(len(kp), len(dest))
    roi = cv2.drawKeypoints(
        roi, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=None)
    # keypoints = orb.detect(thresh, None)
    # keypoints, descriptors = orb.compute(thresh, keypoints)
    # cv2.drawKeypoints(roi, kp, roi, color=(186, 85, 211))
    cv2.rectangle(frame,
                  (frame.shape[1] - roi_offset,
                   frame.shape[0] - roi_offset),
                  (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)

    cv2.putText(frame, text[0], (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255))

    cv2.imshow('Frame', frame)
    cv2.imshow('roi', roi)
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
print(openHand.dest, openHand.kp)

pickle.dump(pickle_keypoints(openHand.kp, openHand.dest),
            open("openHand.p", "wb"))
pickle.dump(pickle_keypoints(closeHand.kp, closeHand.dest),
            open("closeHand.p", "wb"))
cap.release()
cv2.destroyAllWindows()
