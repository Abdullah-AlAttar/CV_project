

import cv2
import numpy as np
import pickle
from save_features import pickle_keypoints, unpickle_keypoints
from features_matching import BruteForceMatcher

sift = cv2.xfeatures2d.SIFT_create()


class ROISelector(object):
    def __init__(self, win_name, init_frame, callback_func):
        self.callback_func = callback_func
        self.selected_rect = None
        self.drag_start = None
        self.tracking_state = 0
        event_params = {"frame": init_frame}
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, self.mouse_event, event_params)

    def mouse_event(self, event, x, y, flags, param):
        x, y = np.int16([x, y])

        # Detecting the mouse button down event
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.tracking_state = 0

        if self.drag_start:
            if event == cv2.EVENT_MOUSEMOVE:
                h, w = param["frame"].shape[:2]
                xo, yo = self.drag_start
                x0, y0 = np.maximum(0, np.minimum([xo, yo], [x, y]))
                x1, y1 = np.minimum([w, h], np.maximum([xo, yo], [x, y]))

                self.selected_rect = None
                if x1 - x0 > 0 and y1 - y0 > 0:
                    self.selected_rect = (x0, y0, x1, y1)

            elif event == cv2.EVENT_LBUTTONUP:
                self.drag_start = None
                if self.selected_rect is not None:
                    self.callback_func(self.selected_rect)
                    self.selected_rect = None

    def draw_rect(self, img, rect):
        if not rect:
            return False
        x_start, y_start, x_end, y_end = rect
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 3)


class HandsCapture(object):
    def __init__(self, capId, scaling_factor, win_name):
        self.cap = cv2.VideoCapture(capId)
        # self.pose_tracker = PoseEstimator()
        self.win_name = win_name
        self.scaling_factor = scaling_factor

        ret, frame = self.cap.read()
        self.rect = None
        # print(ret,frame)
        # self.frame = cv2.resize(
        #     frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        self.frame = frame

        self.roi_selector = ROISelector(win_name, self.frame, self.set_rect)

    def set_rect(self, rect):
        self.rect = rect

    def start(self):
        paused = False
        saved = True
        while True:
            if not paused or self.frame is None:
                ret, frame = self.cap.read()
                # frame = cv2.resize(frame, None, fx=self.scaling_factor,
                #                    fy=self.scaling_factor, interpolation=cv2.INTER_AREA)

                self.frame = frame.copy()

            img = self.frame.copy()
            if False:
                tracked = self.pose_tracker.track_target(self.frame)
                # tracked = []
                for item in tracked:
                    cv2.polylines(img, [np.int32(item.quad)],
                                  True, (255, 255, 255), 2)
                    for (x, y) in np.int32(item.points_cur):
                        cv2.circle(img, (x, y), 2, (255, 255, 255))

            # self.roi_selector.draw_rect(img, self.rect)

            if self.rect:
                x_start, y_start, x_end, y_end = self.rect
                roi = img[y_start:y_end, x_start:x_end]

                kp, desc = sift.detectAndCompute(roi, None)

                roi = cv2.drawKeypoints(
                    roi, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=None)

                img[y_start:y_end, x_start:x_end] = roi
                # cv2.imshow('roi', roi)
                self.roi_selector.draw_rect(img, self.rect)

            cv2.imshow(self.win_name, img)

            ch = cv2.waitKey(1)
            if ch == ord(' '):
                paused = not paused
            if ch == ord('s'):
                print(self.rect, len(kp), len(desc))
                if saved:
                    pickle.dump(pickle_keypoints(kp, desc, self.rect),
                                open("oh.p", "wb"))
                    saved = False
                else:
                    pickle.dump(pickle_keypoints(kp, desc, self.rect),
                                open("ch.p", "wb"))
                    break

            if ch == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows()


class HandsMatcher(object):
    def __init__(self, capId, scaling_factor, win_name):
        self.cap = cv2.VideoCapture(capId)
        self.win_name = win_name
        self.scaling_factor = scaling_factor
        self.matcher = BruteForceMatcher()
        ret, frame = self.cap.read()
        self.rect = None
        # self.frame = cv2.resize(
        #     frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        self.frame = frame

        data = pickle.load(open("oh.p", "rb"))
        kp_open, desc_open, rect = unpickle_keypoints(data)
        self.kp_open = kp_open
        self.desc_open = desc_open
        self.rect = rect
        data = pickle.load(open("ch.p", "rb"))
        kp_close, desc_close, rect = unpickle_keypoints(data)
        self.kp_close = kp_close
        self.desc_close = desc_close

    def set_rect(self, rect):
        self.rect = rect

    def start(self):
        paused = False
        while True:
            if not paused or self.frame is None:
                ret, frame = self.cap.read()
                frame = cv2.resize(frame, None, fx=self.scaling_factor,
                                   fy=self.scaling_factor, interpolation=cv2.INTER_AREA)

                self.frame = frame.copy()

            img = self.frame.copy()
            kp, desc = sift.detectAndCompute(img, None)

            sum_open, matches_open = self.matcher.match(
                self.desc_open, desc, 250)

            sum_close, matches_close = self.matcher.match(
                self.desc_close, desc, 250)

            # sum_open = 2
            # sum_close = 1
            # print(len(matches_open), len(matches_close))
            matches = matches_open if sum_open < sum_close else matches_close
            winner = 'open' if sum_open < sum_close else 'close'
            kp_query = self.kp_open if sum_open < sum_close else self.kp_close

            matches = [match for match in matches]
            # print(sum_open, sum_close, len(matches))

            if len(matches) > 4:
                w = abs(self.rect[0] - self.rect[2])
                h = abs(self.rect[1] - self.rect[3])

                (mnx, mny, mxx, mxy) = self.matcher.get_rectangle_around_features(
                    matches, kp_query, kp, w, h)
                # print(mnx, mny, mxx, mxy)
                cv2.rectangle(
                    img, (mnx, mny), (mxx, mxy), (0, 255, 0), 4)
                if mnx < 0 or mny < 0 or mxx < 0 or mxy < 0:
                    continue
                cv2.putText(img, winner, (int(mnx - 5), int(mny - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 0, 0))
                # self.roi_selector.draw_rect(img, self.rect)

            cv2.imshow(self.win_name, img)

            if self.rect:
                x_start, y_start, x_end, y_end = self.rect
                cv2.imshow('roi', img[y_start:y_end, x_start:x_end])
            ch = cv2.waitKey(1)
            if ch == ord(' '):
                paused = not paused

            if ch == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows()


# if __name__ == '__main__':
#     vh = VideoHandler(0, 0.8, 'Tracker')
#     vh.start()
