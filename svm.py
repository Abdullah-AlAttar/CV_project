import imutils
import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
openHand_dir = "C:\\Users\\mohamed ismail\\Desktop\\Opened\\"
closeHand_dir = "C:\\Users\\mohamed ismail\\Desktop\\Closed\\"
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
                img = cv2.resize(img, (64, 64))
                left, right, top, bottom = self.Borders(img)
                positions_open.append((left, right, top, bottom))
                lst = self.Hog(img)
                X_open.append(lst)
            if 'closed' in paths_close_images[i]:
                y_close.append(-1)
                img = cv2.imread(paths_close_images[i])
                img = cv2.resize(img, (64, 64))
                left, right, top, bottom = self.Borders(img)
                positions_close.append((left, right, top, bottom))
                lst = self.Hog(img)
                X_close.append(lst)
        X_open = np.array(X_open)
        y_open = np.array(y_open)
        X_close = np.array(X_close)
        y_close = np.array(y_close)
        X = np.concatenate((X_open, X_close), axis=0)
        y = np.concatenate((y_open, y_close))
        print(positions_open)
        print(positions_close)
        return X, y, positions_open, positions_close

    def Hog(self, img):
        bin_n = 16
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bins = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n)
                for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)
        return hist

    def SVM(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1)
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.int32)
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)
        clf = svm.SVC()
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        return y_predict, y_test

    def Borders(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        return extLeft, extRight, extTop, extBot


obj = Classify()
obj.LabelsAndFeatures()
