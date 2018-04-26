

import cv2
import numpy as np


class BruteForceMatcher:

    def __init__(self, features_type='sift'):
        if features_type == 'sift':
            self.matcher = cv2.BFMatcher()
        elif features_type == 'orb':
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match(self, des1, des2, amount):

        matches = self.matcher.match(des1, des2)

        s = 0
        for match in matches:
            # print(match.distance)
            s += match.distance

        s /= len(matches)

        # matches = [match for match in matches if match.distance < threshold]
        matches = [match for match in matches]
        matches = sorted(matches, key=lambda x: x.distance)

        return s, matches[:amount]

    def get_rectangle_around_features(self, matches, kp_query, kp_train, w, h):

        src_pts = np.float32(
            [kp_query[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp_train[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        mask = mask.ravel() != 0
        if mask.sum() < 1:
            return -1, -1, -1, -1, -1, -1
        pts = np.float32(
            [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, M)
        # print(dst)
        mnx = dst[:, :, 0].min()
        mxx = dst[:, :, 0].max()
        mny = dst[:, :, 1].min()
        mxy = dst[:, :, 1].max()

        c1, c2 = np.average(dst_pts, axis=0).flatten()
        return (mnx, mny, mxx, mxy, c1, c2)
