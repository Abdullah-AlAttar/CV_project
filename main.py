
import cv2
import numpy as np
from save_features import unpickle_keypoints
import pickle
from enum import Enum
scaling_factor = 0.5
roi_offset = 150
history = 100
# Create the background subtractor object


class State(Enum):
    RECORDING_OPEN = 1
    RECORDING_CLOSE = 2
    MATCHING = 3


state = State.MATCHING

sift = cv2.xfeatures2d.SIFT_create()
# ret, frame = cap.read()


keypoints_database = pickle.load(open("openHand.p", "rb"))
kpopen, descopen = unpickle_keypoints(keypoints_database)

keypoints_database = pickle.load(open("closeHand.p", "rb"))
kpclosed, descclosed = unpickle_keypoints(keypoints_database)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 20)
# print([i.pt for i in kpopen[:5]])
# print(descopen[:5])

# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('output.avi', -1, 20.0, (250, 250))


saved = True
text = ["Openend Hand", "Closed hand"]
cnt = 0
# orb = cv2.ORB_create()

# flann_params = dict(algorithm=6, table_number=6,
#                     key_size=12, multi_probe_level=1)

# flann = cv2.FlannBasedMatcher(flann_params, {})
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
ret, frame = cap.read()
# x_start, x_end, y_start, y_end = frame.shape[1] - \
#     roi_offset, frame.shape[1], frame.shape[0] - roi_offset, frame.shape[0]
x_start, x_end, y_start, y_end = 200, 350, 200, 350
rectangles = []
while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = frame[y_start:y_end, x_start:x_end, :]
    roigray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    kp, dest = sift.detectAndCompute(gray_frame, None)

    c = cv2.waitKey(1)

    if c == 27:
        break

    if c == ord('1'):
        state = State.MATCHING
    if c == ord('2'):
        state = State.RECORDING_OPEN
        cnt = 0
        rectangles = []
    if c == ord('3'):
        state = State.RECORDING_CLOSE
        cnt = 0
        rectangles = []

    if state == State.RECORDING_OPEN:
        if cnt == 24 * 5:
            state = State.MATCHING
        cnt += 1
        matchesOpen = flann.knnMatch(descopen, dest, k=2)
        matchesOpen = [match[0] for match in matchesOpen if len(match) == 2
                       and match[0].distance < match[1].distance * 0.7]

        if len(matchesOpen) > 1:

            src_pts = np.float32(
                [kpopen[m.queryIdx].pt for m in matchesOpen]).reshape(-1, 1, 2)

            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in matchesOpen]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            mask = mask.ravel() != 0
            h, w = roigray.shape

            if mask.sum() < 1:
                continue
            pts = np.float32(
                [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            mnx = dst[:, :, 0].min()
            mxx = dst[:, :, 0].max()
            mny = dst[:, :, 1].min()
            mxy = dst[:, :, 1].max()

            img2 = cv2.rectangle(frame, (x_start, y_start),
                                 (x_end, y_end), (255, 0, 0), 2)
            cv2.rectangle(
                img2, (mnx, mny), (mxx, mxy), (0, 255, 0), 5)

            c1, c2 = np.average(dst_pts, axis=0).flatten()
            cv2.circle(img2, (c1, c2), 2, (255, 255, 255), 2)
            cv2.putText(img2, str(cnt), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 0, 0))
            cv2.imshow('Frame2', img2)

            rectangles.append([mnx, mny, mxx, mxy])
            # Your Code Goes here
            # ----------------------------------------------------
            # only save the the calcaulted box, the ground truth box is constant
            #  write rectangles to a file

            # ----------------------------------------------------
    if state == State.RECORDING_CLOSE:
        if cnt == 24 * 5:
            state = State.MATCHING

        cnt += 1
        matchesClose = flann.knnMatch(descclosed, dest, k=2)
        matchesClose = [match[0] for match in matchesClose if len(match) == 2
                        and match[0].distance < match[1].distance * 0.7]

        if len(matchesClose) > 1:

            src_pts = np.float32(
                [kpclosed[m.queryIdx].pt for m in matchesClose]).reshape(-1, 1, 2)

            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in matchesClose]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            mask = mask.ravel() != 0
            h, w = roigray.shape

            if mask.sum() < 1:
                continue
            pts = np.float32(
                [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            mnx = dst[:, :, 0].min()
            mxx = dst[:, :, 0].max()
            mny = dst[:, :, 1].min()
            mxy = dst[:, :, 1].max()

            img2 = cv2.rectangle(frame, (x_start, y_start),
                                 (x_end, y_end), (255, 0, 0), 2)
            cv2.rectangle(
                img2, (mnx, mny), (mxx, mxy), (0, 255, 0), 5)

            c1, c2 = np.average(dst_pts, axis=0).flatten()
            cv2.circle(img2, (c1, c2), 2, (255, 255, 255), 2)
            cv2.putText(img2, str(cnt), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 0, 0))
            cv2.imshow('Frame2', img2)

            rectangles.append([mnx, mny, mxx, mxy])
            # Your Code Goes here
            # ----------------------------------------------------
            # only save the the calcaulted box, the ground truth box is constant
            #  write rectangles to a file

            # ----------------------------------------------------
    if state == State.MATCHING:

        matchesOpen = flann.knnMatch(descopen, dest, k=2)

        matchesClose = flann.knnMatch(descclosed, dest, k=2)

        sumOpen, sumClose = 0, 0
        for match in matchesOpen:
            sumOpen += match[0].distance
        for match in matchesClose:
            sumClose += match[0].distance

        # print(sumOpen, sumClose)
        winner = 'open' if sumOpen < sumClose else 'close'

        matches = matchesOpen if winner == 'open' else matchesClose

        matches = [match[0] for match in matches if len(match) == 2
                   and match[0].distance < match[1].distance * 0.7]
        # print(len(matches))
        cv2.imshow('Frame2', frame)
        if len(matches) > 1:
            if winner == 'open':
                src_pts = np.float32(
                    [kpopen[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            elif winner == 'close':
                src_pts = np.float32(
                    [kpclosed[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # print(src_pts[:5],dst_pts[:5])
            # print(np.average(src_pts, axis=0).flatten())
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # matchesMask = mask.ravel().tolist()
            mask = mask.ravel() != 0
            h, w = roigray.shape

            if mask.sum() < 1:
                continue
            pts = np.float32(
                [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            mnx = dst[:, :, 0].min()
            mxx = dst[:, :, 0].max()
            mny = dst[:, :, 1].min()
            mxy = dst[:, :, 1].max()
            # print(mnx, mxx, mny, mxy)
            img2 = cv2.polylines(
                frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            cv2.rectangle(
                img2, (mnx, mny), (mxx, mxy), (0, 255, 0), 5)

            cv2.putText(frame, winner, (int(mnx - 5), int(mny - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 0, 0))

            c1, c2 = np.average(dst_pts, axis=0).flatten()
            cv2.circle(img2, (c1, c2), 2, (255, 255, 255), 2)
            cv2.imshow('Frame2', img2)
            # cv2.imshow('abc', roi)
            # cv2.waitKey(0)
    # out.write(roi)

# out.release()
cap.release()
cv2.destroyAllWindows()
