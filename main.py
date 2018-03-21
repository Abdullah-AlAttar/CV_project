
import cv2
import numpy as np
from save_features import pickle_keypoints, unpickle_keypoints
import pickle
scaling_factor = 0.5
roi_offset = 250
history = 100
# Create the background subtractor object
sift = cv2.xfeatures2d.SIFT_create()
# ret, frame = cap.read()


keypoints_database = pickle.load(open("openHand.p", "rb"))
kpopen, descopen = unpickle_keypoints(keypoints_database)

keypoints_database = pickle.load(open("closeHand.p", "rb"))
kpclosed, descclosed = unpickle_keypoints(keypoints_database)
cap = cv2.VideoCapture(0)


# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('output.avi', -1, 20.0, (250, 250))


saved = False
text = ["Openend Hand", "Closed hand"]
cnt = 0
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply the background subtraction model to the input frame
    # Convert from grayscale to 3-channel RGB
    roi = frame[frame.shape[0] - roi_offset:,
                frame.shape[1] - roi_offset:].copy()
    # print(roi.shape)
    # keypoints = sift.detect(roi, None)
    roigray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    kp, dest = sift.detectAndCompute(gray_frame, None)
    # keypoints = orb.detect(thresh, None)
    # keypoints, descriptors = orb.compute(thresh, keypoints)

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
        saved = True
    if cnt == 5:
        break

    if saved:
        cnt += 1
        matches = flann.knnMatch(descopen, dest, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        # print(len(matches))
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        print(len(good))
        if len(good) > 10:
            src_pts = np.float32(
                [kpopen[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # print(len(kp))
            # dst_pts = []
            # for m in good:
            #     dst_pts.append(kp[m.trainIdx].pt)
            #     print(m.trainIdx, m.queryIdx)
            # print(src_pts)
            # print(dst_pts)
            # dst_pts = np.array(dst_pts)
            print(src_pts, dst_pts)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = roigray.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                              [w - 1, 0]]).reshape(-1, 1, 2)
            print(M)
            # print(pts.shape, M.shape)
            dst = cv2.perspectiveTransform(pts, M)

            roi = cv2.polylines(
                roi, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            cv2.imshow('abc', roi)
            cv2.waitKey(0)
    # out.write(roi)

# out.release()
cap.release()
cv2.destroyAllWindows()
