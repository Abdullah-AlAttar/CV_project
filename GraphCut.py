import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('./Opened/g2 (18).jpg')
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (0, 0, img.shape[0], img.shape[1])
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img2 = img * mask2[:, :, np.newaxis]
plt.imshow(img2), plt.colorbar(), plt.show()

newmask = cv2.imread("m.png", 0)
print(newmask)
mask2[newmask == 0] = 0
mask2[newmask == 255] = 1
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

mask2, bgdModel, fgdModel = cv2.grabCut(img2, mask2, None, bgdModel, fgdModel, 5,
                                        cv2.GC_INIT_WITH_MASK)

mask2 = np.where((mask2 == 2) | (mask2 == 0), 0, 1).astype('uint8')
img2 = img2 * mask2[:, :, np.newaxis]
plt.imshow(img2), plt.colorbar(), plt.show()
