
import os
import numpy as np
import pandas as pd
import cv2
import skin_detector
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import imutils

opened_data = ['./Opened/' + i for i in os.listdir('./Opened')]
closed_data = ['./Closed/' + i for i in os.listdir('./Closed')]

image_dims = (80, 80)
# for path in closed_data:
#     img = cv2.imread(path)
#     img = cv2.resize(img, image_dims, interpolation=cv2.INTER_AREA)

#     mask = skin_detector.process(img, 0.3)
#     res = cv2.bitwise_and(img, img, mask=mask)
#     print(path.split('/'))
#     cv2.imwrite('./closed_mask/mask_'+path.split('/')[-1], res)

# for path in opened_data:
#     img = cv2.imread(path)
#     img = cv2.resize(img, image_dims, interpolation=cv2.INTER_AREA)

#     mask = skin_detector.process(img, 0.3)
#     res = cv2.bitwise_and(img, img, mask=mask)
#     print(path.split('/'))
#     cv2.imwrite('./opened_mask/mask_'+path.split('/')[-1], res)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

loaded_model.compile(optimizer='rmsprop',
                     loss='categorical_crossentropy', metrics=['accuracy'])


X = []
y = []
for path in closed_data:
    img = cv2.imread(path)
    img = cv2.resize(img, image_dims, interpolation=cv2.INTER_AREA)

    # mask = skin_detector.process(img, 0.3)
    # res = cv2.bitwise_and(img, img, mask=mask)
    X.append(img/255)
    y.append(0)

for path in opened_data:
    img = cv2.imread(path)
    img = cv2.resize(img, image_dims, interpolation=cv2.INTER_AREA)

    # mask = skin_detector.process(img, 0.3)
    # res = cv2.bitwise_and(img, img, mask=mask)
    # print(path.split('/'))
    # cv2.imwrite('./opened_mask/mask_'+path.split('/')[-1], res)
    X.append(img/255)
    y.append(1)

X = np.array(X)
y = np.array(y)

pred = loaded_model.predict(X)
class_labels = np.argmax(pred, axis=1)
print(class_labels)
for i in range(len(pred)):
    print(y[i], class_labels[i])
print(np.sum(y == class_labels).sum()/len(y))


def get_border(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])
            return thresh, extLeft, extRight, extTop, extBot
        return thresh, -1, -1, -1, -1


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # mask = skin_detector.process(frame, 0.4)
    # res = cv2.bitwise_and(frame, frame, mask=mask)
    img = cv2.resize(frame, image_dims, interpolation=cv2.INTER_AREA)
    roi = frame[:200, :200, :]
    roi_tmp = cv2.resize(roi, image_dims, interpolation=cv2.INTER_AREA)
    cv2.rectangle(frame, (0, 0), (200, 200), (0, 0, 255), 2)

    pred = loaded_model.predict(roi_tmp.reshape(-1, 80, 80, 3) / 255)
    print(pred)
    class_labels = np.argmax(pred, axis=1)
    # print("close" if class_labels[0] == 0 else "open")
    hand_status = "close" if class_labels[0] == 0 else "open"
    thresh, left, right, top, bot = get_border(roi)
    # print(left, right, top, bot)
    if left != -1:
        cv2.circle(roi, left, 10, (0, 255, 0), -1)
        cv2.circle(roi, right, 10, (0, 255, 0), -1)
        cv2.circle(roi, top, 10, (0, 255, 0), -1)
        cv2.circle(roi, bot, 10, (0, 255, 0), -1)
        c1 = int((left[0] + right[0] + top[0] + bot[0]) / 4)
        c2 = int((left[1] + right[1] + top[1] + bot[1]) / 4)
        cv2.circle(roi, (c1, c2), 10, (0, 128, 255), -1)

        max_x = max(left[0], top[0], bot[0], right[0])
        max_y = max(left[1], top[1], bot[1], right[1])
        min_x = min(left[0], top[0], bot[0], right[0])
        min_y = min(left[1], top[1], bot[1], right[1])
        cv2.rectangle(frame, (max_x, max_y), (min_x, min_y), (255, 0, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, hand_status, (int(
            frame.shape[0]/2), 50), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    # cv2.imshow("mask", mask)
    cv2.imshow('frame', frame)
    cv2.imshow('thresh', thresh)
    cv2.imshow('roi', roi_tmp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
