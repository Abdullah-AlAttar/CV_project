import cv2
import numpy as np
from save_features import pickle_keypoints, unpickle_keypoints
import pickle


def initilize():
    global flann, sift
    sift = cv2.xfeatures2d.SIFT_create()
    # flann_params = dict(algorithm=6, table_number=6,
    #                     key_size=12, multi_probe_level=1)
    # flann = cv2.FlannBasedMatcher(flann_params, {})
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, {})


def show_webcam(mirror=False):
    global roi_offset, flann, sift
    # read saved features
    keypoints_database = pickle.load(open("openHand.p", "rb"))
    kpopen, descopen = unpickle_keypoints(keypoints_database)
    keypoints_database = pickle.load(open("closeHand.p", "rb"))
    kpclosed, descclosed = unpickle_keypoints(keypoints_database)

    KPlist = [kpopen, kpclosed]
    KPDesc = [descopen, descclosed]
    Names = ["Openend Hand", "Closed hand"]
    Minmum = [10, 3]

    cam = cv2.VideoCapture(0)
    # cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Webcam', 1080, 760)

    rev = True
    saved = False
    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x_start, x_end, y_start, y_end = frame.shape[1] - \
                roi_offset, frame.shape[1], frame.shape[0] - \
                roi_offset, frame.shape[0]
        roi = gray_frame[y_start:y_end, x_start:x_end]

        kp, dest = sift.detectAndCompute(gray_frame, None)
        roi2 = roi.copy()
        roi2 = cv2.drawKeypoints(roi2, kp,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=None)
        cv2.rectangle(frame, (x_start, y_start),
                      (x_end, y_end), (0, 255, 0), 2)

        if saved and len(kp) > 1:
            index = -1
            mn = 1e10
            matchestmp = []
            for i in range(0, 2):
                destquerry = KPDesc[i]
                matches = flann.knnMatch(destquerry, dest, k=2)

                matches = [match[0] for match in matches if len(match) == 2
                           and match[0].distance < match[1].distance * 0.7]
                print(len(matches))
                sum = 0
                for match in matches:
                    sum += match.distance
                if(len(matches) < Minmum[i]):
                    continue
                if sum < mn:
                    mn = sum
                    index = i
                    matchestmp = matches
            if index != -1:

                kpquerry = KPlist[index]
                src_pts = np.float32(
                    [kpquerry[m.queryIdx].pt for m in matchestmp]).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kp[m.trainIdx].pt for m in matchestmp]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                h, w = roi.shape
                pts = np.float32(
                    [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                dst = cv2.perspectiveTransform(pts, M)
                dst = np.int32(dst)
                mnx = dst[:, :, 0].min()
                mxx = dst[:, :, 0].max()
                mny = dst[:, :, 1].min()
                mxy = dst[:, :, 1].max()
                print(mnx, mxx, mny, mxy)
                # A=dst[0][0]
                # B=dst[0][1]
                # C=dst[0][2]
                # D=dst[0][3]
                # frame = cv2.polylines(
                #          frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

                frame = cv2.rectangle(
                    frame, (mnx, mny), (mxx, mxy), (0, 255, 0), 5)
                cv2.putText(frame, Names[index], (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255))

        # if rev :
            # frame=cv2.flip(frame,1)
        cv2.imshow('Webcam', frame)
        cv2.imshow('roi', roi2)
        c = cv2.waitKey(1)
        if c == 27:
            break
        if c == ord('s'):
            saved = not saved


# Main function
if __name__ == '__main__':
    flann = 0
    roi_offset = 250
    sift = 0
    initilize()

    show_webcam(mirror=True)
