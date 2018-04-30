import imutils
import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from collections import deque

openHand_dir = './opened_mask/'
closeHand_dir = './closed_mask/'
paths_open_images = [openHand_dir+i for i in os.listdir(openHand_dir)]
paths_close_images = [closeHand_dir+i for i in os.listdir(closeHand_dir)]


class Classify:
    def __init__(self):
      pass

    def LabelsAndFeatures(self):
        X_open = []
        X_close = []
        y_open = []
        y_close = []
        positions_open = []
        positions_close = []
        for i in range(0, 70):
            if 'opened' in paths_open_images[i]:
                y_open.append(1)
                img = cv2.imread(paths_open_images[i])
                thresh, box = self.get_border(img.copy())
                if box is not None:
                    img = img[box[1]:box[3], box[0]:box[2]]
                # exit(0)
                img = cv2.resize(img, (64, 64))
                lst = img.reshape((3*64*64))
                X_open.append(lst)
            if 'closed' in paths_close_images[i]:
                y_close.append(-1)
                img = cv2.imread(paths_close_images[i])
                thresh, box = self.get_border(img.copy())
                if box is not None:
                    img = img[box[1]:box[3], box[0]:box[2]]
                img = cv2.resize(img, (64, 64))
                lst = img.reshape((3*64*64))
                X_close.append(lst)
        X_open = np.array(X_open)
        y_open = np.array(y_open)
        X_close = np.array(X_close)
        y_close = np.array(y_close)
        X = np.concatenate((X_open, X_close), axis=0)
        y = np.concatenate((y_open, y_close))
        # print(positions_open)
        # print(positions_close)
        self.SVM(X, y)
        return X, y, positions_open, positions_close

    def Hog(self, img):
        bin_n = 16
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bins = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bins[:32, :32], bins[32:,
                                         :32], bins[:32, 32:], bins[32:, 32:]
        mag_cells = mag[:32, :32], mag[32:, :32], mag[:32, 32:], mag[32:, 32:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n)
                 for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)
        return hist

    def SVM(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=2)
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.int32)
        self.clf = svm.SVC()
        print(X_train.shape)
        self.clf.fit(X_train, y_train)
        y_predict = self.clf.predict(X_test)
        print(y_predict, y_test)
        msk = y_predict == y_test
        print(np.bincount(msk))
        print(np.bincount(msk)[1]/y_predict.shape[0]*100)
        # exit(0)
        return y_predict, y_test

    def get_border(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = self.skinDetect(img)
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            box = self.draw_ParallelSide_rect(img, c)
            # cv2.drawre
            # cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
            cv2.rectangle(img, (box[0], box[1]),
                          (box[2], box[3]), (0, 255, 0), 2)

            return thresh, box
        return thresh, None

    def draw_smallest_rect(self, frame, contor):
        rect = cv2.minAreaRect(contor)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        return box

    def skinDetect(self, frame):
        lower = np.array([6, 60, 0], dtype="uint8")
        upper = np.array([40, 100, 150], dtype="uint8")
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        # kernel = np.ones((11, 11))
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)
        skinMask = cv2.erode(skinMask, kernel, iterations=1)

        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skin = cv2.bitwise_and(frame, frame, mask=skinMask)
        return skinMask

    def draw_ParallelSide_rect(self, frame, contor):
        x, y, w, h = cv2.boundingRect(contor)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return [x, y, x+w, y+h]

    def run(self):
        self.cap = cv2.VideoCapture(0)
        ret, self.frame = self.cap.read()
        self.draw_mask = np.zeros(self.frame.shape)
        preds = deque(maxlen=5)
        while True:
            ret, frame = self.cap.read()
            model_img = frame.copy()
            thresh, box = self.get_border(frame.copy())
            if box is not None:
                h = box[2]-box[0]
                w = box[3]-box[1]
                # print(box, frame.shape)
                if w*h > 0:
                    model_img = model_img[box[1]:box[3], box[0]:box[2]]
            # # exit(0)
            # print(model_img.shape)
            model_img = cv2.resize(model_img, (64, 64))
            model_img = model_img.reshape((-1, 3*64*64))
            pred = self.clf.predict(
                model_img)
            thresh, box = self.get_border(frame)
            if box is None:
                preds.append(-1)
            else:
                preds.append(pred[0])
            if len(preds) > 4:
                    a = np.unique(preds)
                    if len(a) == 1:
                        if a[0] == 1:
                            
                            c1=int((box[0]+box[2])/2)
                            c2=int((box[1]+box[3])/2)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(frame, "drawing", (int(
                                frame.shape[0]/2), 150), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.circle(self.draw_mask, (c1, c2),
                                        10, (0, 128, 255), -1)

            # print(box)
            # hand_status = "close" if pred[0] == -1 else "open"
            # thresh, box = self.get_border(frame)
            print(pred)
            cv2.imshow('abc', frame)
            cv2.imshow('msk',self.draw_mask)
            ch = cv2.waitKey(1)
            if ch == ord(' '):
                paused = not paused
            if ch == ord('c'):
                self.draw_mask = np.zeros(self.draw_mask.shape)
            if ch == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows()


obj = Classify()
obj.LabelsAndFeatures()
obj.run()
