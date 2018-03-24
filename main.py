
import cv2
import numpy as np
from save_features import pickle_keypoints, unpickle_keypoints
import pickle
scaling_factor = 0.5
roi_offset = 150
history = 100
# Create the background subtractor object
sift = cv2.xfeatures2d.SIFT_create()
# ret, frame = cap.read()


keypoints_database = pickle.load(open("openHand.p", "rb"))
kpopen, descopen = unpickle_keypoints(keypoints_database)

keypoints_database = pickle.load(open("closeHand.p", "rb"))
kpclosed, descclosed = unpickle_keypoints(keypoints_database)
cap = cv2.VideoCapture(0)
# print([i.pt for i in kpopen[:5]])
# print(descopen[:5])

# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('output.avi', -1, 20.0, (250, 250))


saved = True
text = ["Openend Hand", "Closed hand"]
cnt = 0
orb = cv2.ORB_create()

flann_params = dict(algorithm=6, table_number=6,
                    key_size=12, multi_probe_level=1)
flann = cv2.FlannBasedMatcher(flann_params, {})
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply the background subtraction model to the input frame
    # Convert from grayscale to 3-channel RGB
    x_start, x_end, y_start, y_end = frame.shape[1] - \
        roi_offset, frame.shape[1], frame.shape[0] - roi_offset, frame.shape[0]

    roi = frame[y_start:y_end, x_start:x_end, :]
    # print(roi.shape)
    # keypoints = sift.detect(roi, None)
    roigray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    # ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
    # kp, dest = sift.detectAndCompute(roi, None)
    kp, dest = sift.detectAndCompute(gray_frame, None)
    # frame = cv2.drawKeypoints(
    #     frame, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=None)
    # keypoints = orb.detect(thresh, None)
    # keypoints, descriptors = orb.compute(thresh, keypoints)

    # cv2.rectangle(frame,
    #               (frame.shape[1] - roi_offset,
    #                frame.shape[0] - roi_offset),
    #               (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)

    # cv2.putText(frame, text[0], (50, 50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255))
    # if saved:
    #     cv2.imshow('Frame', frame)
    cv2.imshow('roi', roi)
    # cv2.imshow('mask1', mask1)
    c = cv2.waitKey(1)
    # print(saved)
    if c == 27:
        break
    if c == ord('s'):
        saved = True
    if cnt == 10:
        # saved = False
        cnt = 0

    if saved:
        cnt += 1

        matches = flann.knnMatch(descopen, dest, k=2)
        # store all the good matches as per Lowe's ratio test.
        matches = [match[0] for match in matches if len(
            match) == 2 and match[0].distance < match[1].distance * 0.7]
        # print(len(matches))
        good = []

        if len(matches) > 1:
            src_pts = np.float32(
                [kpopen[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # print(src_pts[:5],dst_pts[:5])
            # print(np.average(src_pts, axis=0).flatten())
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            mask = mask.ravel() != 0
            h, w = roigray.shape

            if mask.sum() < 1:
                continue
            pts = np.float32(
                [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(
                frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            c1, c2 = np.average(dst_pts, axis=0).flatten()
            cv2.circle(img2, (c1, c2), 2, (255, 255, 255), 2)
            cv2.imshow('Frame2', img2)
            # cv2.imshow('abc', roi)
            # cv2.waitKey(0)
    # out.write(roi)

# out.release()
cap.release()
cv2.destroyAllWindows()
