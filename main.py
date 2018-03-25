
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


keypoints_database = pickle.load(open("openHand.p", "rb"))
kpopen, descopen = unpickle_keypoints(keypoints_database)

keypoints_database = pickle.load(open("closeHand.p", "rb"))
kpclosed, descclosed = unpickle_keypoints(keypoints_database)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 20)

ret, frame = cap.read()
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', -1, 20.0, (640, 480))

saved = True
text = ["Openend Hand", "Closed hand"]
cnt = 0


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# create BFMatcher object
bf = cv2.BFMatcher()

# Match descriptors.
# matches = bf.match(des1, des2)
ret, frame = cap.read()


x_start, x_end, y_start, y_end = 200, 350, 200, 350
rectangles = []
while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = frame[y_start:y_end, x_start:x_end, :]
    roigray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    kp, dest = sift.detectAndCompute(gray_frame, None)
    # frame = cv2.drawKeypoints(
    # frame, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=None)

    c = cv2.waitKey(1)

    if c == 27:
        break

    if c == ord('1'):
        state = State.MATCHING
    if c == ord('2'):
        state = State.RECORDING_OPEN
        cnt = 0
        rectangles = []
        rectangles.append([0, x_start, y_start, x_end, y_end])
    if c == ord('3'):
        state = State.RECORDING_CLOSE
        cnt = 0

        rectangles = []
        rectangles.append([0, x_start, y_start, x_end, y_end])

    if state == State.RECORDING_OPEN:
        if cnt == 20 * 5:
            state = State.MATCHING
        cnt += 1
        print('hello', cnt)
        # matchesOpen = flann.knnMatch(descopen, dest, k=2)
        # matchesOpen = [match[0] for match in matchesOpen if len(match) == 2
        #    and match[0].distance < match[1].distance * 0.7]
        matchesOpen = bf.match(descopen, dest)
        # bf.knnMatch()
        matches = [match for match in matchesOpen]
        matches = [match for match in matches if match.distance < 300]

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
            print(img2.shape)
            out.write(img2)
            rectangles.append([cnt, mnx, mny, mxx, mxy])

    if state == State.RECORDING_CLOSE:
        if cnt == 20 * 5:
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

            rectangles.append([cnt, mnx, mny, mxx, mxy])

    if state == State.MATCHING:

        # matchesOpen = flann.knnMatch(descopen, dest, k=2)
        # matchesClose = flann.knnMatch(descclosed, dest, k=2)
        matchesOpen = bf.match(descopen, dest)
        matchesClose = bf.match(descclosed, dest)
        # print(len(matchesClose), len(matchesOpen))

        sumOpen, sumClose = 0, 0
        # matchesOpen = sorted(matchesOpen, key=lambda x: x.distance)
        # matchesClose = sorted(matchesClose, key=lambda x: x.distance)
        i = 0
        for match in matchesOpen:
            i += 1
            sumOpen += match.distance
            # if i == 5:
            #     break
        # print(match[0])
        i = 0
        for match in matchesClose:
            i += 1
            sumClose += match.distance
            # if i == 5:
            #     break
            # print()
        sumOpen /= len(matchesOpen)
        sumClose /= len(matchesClose)
        print(sumOpen, sumClose)
        winner = 'open' if sumOpen < sumClose else 'close'
        # print(winner)
        matches = matchesOpen if winner == 'open' else matchesClose

        # matches = [match[0] for match in matches if len(match) == 2
        #    and match[0].distance < match[1].distance * 0.7]
        # print(matches[0].distance)
        matches = [match for match in matches if match.distance < 300]

        print(len(matches))
        cv2.imshow('Frame2', frame)
        if len(matches) > 5:
            if winner == 'open':
                src_pts = np.float32(
                    [kpopen[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            elif winner == 'close':
                src_pts = np.float32(
                    [kpclosed[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            h, w = roigray.shape

            mask = mask.ravel() != 0
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

            # c1, c2 = np.average(dst_pts, axis=0).flatten()
            # cv2.circle(img2, (c1, c2), 2, (255, 255, 255), 2)
            cv2.imshow('Frame2', img2)


# out.release()
f = open('rects.txt', mode='w')
for rect in rectangles:
    f.write(str(rect[0]) + " " + str(rect[1]) + " " +
            str(rect[2]) + " " + str(rect[3]) + " " +
            str(rect[4]) + '\n')
    # f.write('/n')

f.close()
cap.release()
out.release()
cv2.destroyAllWindows()
