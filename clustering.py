import numpy as np
import cv2

# img = cv2.imread('./Closed/g7 (22).jpg')

# print(img.shape, Z.shape)
# # define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
# ret, label, center = cv2.kmeans(
#     Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# # Now convert back into uint8, and make original image
# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape((img.shape))
cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (300,300), interpolation=cv2.INTER_AREA)
    Z = frame.reshape((-1, 3))

# convert to np.float32
    Z = np.float32(Z)
    # Our operations on the frame come here
    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((frame.shape))
    cv2.imshow('frame', res2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# print(np.unique(res2))
# print(res)

# cv2.imshow('res2', res2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
