
import cv2
import numpy as np
from save_features import pickle_keypoints, unpickle_keypoints
import pickle
cap = cv2.VideoCapture(0)
roi_offset = 150
# Create the background subtractor object

sift = cv2.xfeatures2d.SIFT_create()
# ret, frame = cap.read()


class HandFeature:

    def __init__(self, dest, kp):
        self.dest = dest
        self.kp = kp


# orb = cv2.ORB_create()
saved = False
text = ["Openend Hand", "Closed hand"]
while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    x_start, x_end, y_start, y_end = frame.shape[1] - \
        roi_offset, frame.shape[1], frame.shape[0] - roi_offset, frame.shape[0]

    roi = frame[y_start:y_end, x_start:x_end, :]

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    kp, dest = sift.detectAndCompute(roi_gray, None)

    roi = cv2.drawKeypoints(
        roi_gray, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=None)

    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    cv2.putText(frame, text[0], (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255))

    cv2.imshow('Frame', frame)
    cv2.imshow('roi', roi)

    c = cv2.waitKey(1)
    if c == 27:
        break
    if c == ord('s'):
        if not saved:
            openHand = HandFeature(dest, kp)
            cv2.imwrite('open.png', roi)

            saved = True
            text = text[::-1]
        else:
            closeHand = HandFeature(dest, kp)
            cv2.imwrite('close.png', roi)
            break


pickle.dump(pickle_keypoints(openHand.kp, openHand.dest),
            open("openHand.p", "wb"))
pickle.dump(pickle_keypoints(closeHand.kp, closeHand.dest),
            open("closeHand.p", "wb"))
cap.release()
cv2.destroyAllWindows()
