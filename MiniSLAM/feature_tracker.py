import cv2
import numpy as np

class ORBTracker:
    def __init__(self, max_features=2000, ratio=0.8, flann_checks=50):
        self.max_features = max_features
        self.ratio = ratio
        self.flann_checks = flann_checks

        self.orb = cv2.ORB_create(nfeatures=self.max_features)

        index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=self.flann_checks)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def detect(self, image):
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors            

    def match(self, kp1, des1, kp2, des2):
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return [], np.array([]), np.array([])

        matches = self.matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        pts1 = []
        pts2 = []

        for m_n in matches:
            if len(m_n) != 2:
                continue

            m, n = m_n 
            if m.distance < self.ratio * n.distance:
                good_matches.append(m)
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)

        return good_matches, np.array(pts1), np.array(pts2)

    def draw_matches(self, img1, kp1, img2, kp2, matches):
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return match_img

    def draw_inlier_matches(self, img1, kp1, img2, kp2, matches, mask):
        inlier_matches = [m for i, m in enumerate(matches) if mask[i]]
        inlier_img = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None,
                                     matchColor=(0, 255, 0), singlePointColor=(255, 0, 0))
        return inlier_img

    def filter_with_ransac(self, pts1, pts2, K):
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or mask is None:
            return None, None, None
        inliers1 = pts1[mask.ravel() == 1]
        inliers2 = pts2[mask.ravel() == 1]
        return inliers1, inliers2, mask.ravel().astype(bool)

    def update_params(self, max_features=None, ratio=None, flann_checks=None):
        if max_features is not None:
            self.max_features = max_features
            self.orb = cv2.ORB_create(nfeatures=self.max_features)
        if ratio is not None:
            self.ratio = ratio
        if flann_checks is not None:
            self.flann_checks = flann_checks
            index_params = dict(algorithm=6,
                                table_number=6,
                                key_size=12,
                                multi_probe_level=1)
            search_params = dict(checks=self.flann_checks)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
