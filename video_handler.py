import cv2
import numpy as np
import pickle
from save_features import pickle_keypoints, unpickle_keypoints
from features_matching import BruteForceMatcher
from collections import deque
from keras.models import model_from_json
import imutils
# sift = cv2.xfeatures2d.SIFT_create()


class ROISelector():
    def __init__(self, win_name, init_frame, callback_func):
        self.callback_func = callback_func
        self.selected_rect = None
        self.drag_start = None
        self.tracking_state = 0
        self.is_drawing = False
        self.moving = False
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

                self.is_drawing = True
                h, w = param["frame"].shape[:2]
                xo, yo = self.drag_start
                x0, y0 = np.maximum(0, np.minimum([xo, yo], [x, y]))
                x1, y1 = np.minimum([w, h], np.maximum([xo, yo], [x, y]))

                self.selected_rect = None
                if x1 - x0 > 0 and y1 - y0 > 0:
                    self.selected_rect = (x0, y0, x1, y1)

            elif event == cv2.EVENT_LBUTTONUP:
                self.drag_start = None
                self.is_drawing = False
                if self.selected_rect is not None:
                    self.callback_func(self.selected_rect)
                    self.selected_rect = None

    def draw_rect(self, img, rect):
        if not rect:
            return False
        x_start, y_start, x_end, y_end = rect
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 3)


class HandsCapture():
    def __init__(self, capId, feature_detector='sift', win_name='default'):
        self.cap = cv2.VideoCapture(capId)
        self.win_name = win_name
        ret, frame = self.cap.read()
        self.rect = None
        # print(ret,frame)
        if feature_detector == 'sift':
            self.feature_detector = cv2.xfeatures2d.SIFT_create()
        elif feature_detector == 'orb':
            self.feature_detector = cv2.ORB_create()

        self.frame = frame

        self.roi_selector = ROISelector(win_name, self.frame, self.set_rect)

    def set_rect(self, rect):
        self.rect = rect
        self.roi_selector.moving = True

    def draw_rect_while_selecting(self, img, rect):
        x_start, y_start, x_end, y_end = rect
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 3)

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
            if self.roi_selector.is_drawing and self.roi_selector.selected_rect:
                x_start, y_start, x_end, y_end = self.roi_selector.selected_rect
                cv2.rectangle(img, (x_start, y_start),
                              (x_end, y_end), (0, 100, 255), 3)

            if self.rect:
                x_start, y_start, x_end, y_end = self.rect
                roi = img[y_start:y_end, x_start:x_end]

            if self.roi_selector.moving:

                # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # roi = cv2.GaussianBlur(roi, (5, 5), 0)
                # ret, roi = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)
                # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                kp, desc = self.feature_detector.detectAndCompute(roi, None)
                roi = cv2.drawKeypoints(
                    roi,
                    kp,
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                    outImage=None)

                # cv2.imshow('blur', blur)
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
                    pickle.dump(
                        pickle_keypoints(kp, desc, self.rect),
                        open("oh.p", "wb"))
                    saved = False
                else:
                    pickle.dump(
                        pickle_keypoints(kp, desc, self.rect),
                        open("ch.p", "wb"))
                    break

            if ch == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows()


class HandsMatcher():
    def __init__(self, capId, feature_detector='sift', win_name='default'):
        self.cap = cv2.VideoCapture(capId)
        self.win_name = win_name
        self.matcher = BruteForceMatcher(feature_detector)
        ret, frame = self.cap.read()
        self.rect = None
        # self.frame = cv2.resize(
        #     frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        self.frame = frame

        if feature_detector == 'sift':
            self.feature_detector = cv2.xfeatures2d.SIFT_create()
        elif feature_detector == 'orb':
            self.feature_detector = cv2.ORB_create()

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
        # centers = deque(maxlen=5)

        while True:
            if not paused or self.frame is None:
                ret, frame = self.cap.read()

                self.frame = frame.copy()
            img = self.frame.copy()
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # print(frame)
            kp, desc = self.feature_detector.detectAndCompute(img, None)

            # img = cv2.drawKeypoints(
            # img, kp[20:], flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT, outImage=None)
            amount = min(len(self.desc_close), len(self.desc_open)) * 3//4

            sum_open, matches_open = self.matcher.match(
                self.desc_open, desc, amount)

            sum_close, matches_close = self.matcher.match(
                self.desc_close, desc, amount)

            matches = matches_open if sum_open < sum_close else matches_close
            winner = 'open' if sum_open < sum_close else 'close'
            kp_query = self.kp_open if sum_open < sum_close else self.kp_close

            matches = [match for match in matches if match.distance < 300]
            # print(sum_open, sum_close, len(matches))
            # print(len(matches))
            if len(matches) > 4:
                w = abs(self.rect[0] - self.rect[2])
                h = abs(self.rect[1] - self.rect[3])

                (mnx, mny, mxx, mxy, c1,
                 c2) = self.matcher.get_rectangle_around_features(
                     matches, kp_query, kp, w, h)
                check = False
                if mnx < 0 or mnx > frame.shape[0]:
                    check = True
                if mxx < 0 or mxx > frame.shape[0]:
                    check = True
                if mny < 0 or mny > frame.shape[0]:
                    check = True
                if mxy < 0 or mxy > frame.shape[0]:
                    check = True

                # print(check   )
                # print(len(centers))
                if not check:

                    cv2.circle(img, (c1, c2), 4, (255, 0, 255), 4)

                    cv2.rectangle(img, (mnx, mny), (mxx, mxy), (0, 255, 0), 4)
                    cv2.putText(
                        img,
                        winner, (int(mnx - 5), int(mny - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color=(255, 255, 255),
                        thickness=4)
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


class HandsDrawer():

    def __init__(self, capId, image_dims, win_name='default', model_path='./'):
        self.cap = cv2.VideoCapture(capId)
        self.win_name = win_name
        self.model = self.__load_model(model_path)
        self.image_dims = image_dims
        ret, frame = self.cap.read()
        self.frame = frame
        self.draw_mask = np.zeros(frame.shape)

    def __load_model(self, model_path):
        json_file = open(model_path + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(model_path + ".h5")

        loaded_model.compile(optimizer='rmsprop',
                             loss='categorical_crossentropy', metrics=['accuracy'])
        return loaded_model

    def get_border(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            box = self.draw_smallest_rect(img, c)
            cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
            return thresh, box
        return thresh, None

    def draw_smallest_rect(self, frame, contor):
        rect = cv2.minAreaRect(contor)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        return box

    def draw_ParallelSide_rect(frame, contor):
        x, y, w, h = cv2.boundingRect(contor)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def start(self):
        paused = False
        preds = deque(maxlen=5)
        while True:
            if not paused or self.frame is None:
                ret, frame = self.cap.read()
                model_img = cv2.resize(
                    frame, self.image_dims, interpolation=cv2.INTER_AREA)
                pred = self.model.predict(
                    model_img.reshape(-1, self.image_dims[0], self.image_dims[1], 3) / 255)
                pred_label = np.argmax(pred, axis=1)
                hand_status = "close" if pred_label[0] == 0 else "open"
                thresh, box = self.get_border(frame)

                if box is None:
                    preds.append(-1)
                else:
                    preds.append(pred_label[0])
                    c1, c2 = np.average(box, axis=0)
                    c1, c2 = int(c1), int(c2)
                    for i in box:
                        cv2.circle(frame, (int(i[0]), int(
                            i[1])), 10, (0, 255, 0), -1)

                    cv2.circle(frame, (c1, c2), 10, (0, 128, 255), -1)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, hand_status, (int(
                        frame.shape[0]/2), 50), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

                    if len(preds) > 4:
                        a = np.unique(preds)
                        if len(a) == 1:
                            if a[0] == 1:
                                cv2.putText(frame, "drawing", (int(
                                    frame.shape[0]/2), 150), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
                                cv2.circle(self.draw_mask, (c1, c2),
                                           10, (0, 128, 255), -1)

                self.frame = frame.copy()
                # self.roi_selector.draw_rect(img, self.rect)
            img = self.frame.copy()
            cv2.imshow('mask', self.draw_mask)
            cv2.imshow('thresh', thresh)
            cv2.imshow(self.win_name, img)

            ch = cv2.waitKey(1)
            if ch == ord(' '):
                paused = not paused
            if ch == ord('c'):
                self.draw_mask = np.zeros(self.draw_mask.shape)
            if ch == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows()
