import cv2
import numpy as np

class ORBTracker:
    def __init__(self, max_features=2000):
        self.orb = cv2.ORB_create(nfeatures=max_features)

        # FLANN pour ORB (avec LSH = Locality-Sensitive Hashing)
        index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def detect(self, image):
        """
        Détecte les keypoints et calcule les descripteurs ORB
        :param image: image (gauche) actuelle
        :return: keypoints, descripteurs
        """
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors            

    def match(self, kp1, des1, kp2, des2, ratio=0.75):
        """
        Matche les descripteurs entre deux frames avec le ratio test
        :param kp1: keypoints image t-1
        :param des1: descripteurs t-1
        :param kp2: keypoints image t
        :param des2: descripteurs t
        :return: matches filtrés, pts1, pts2
        """
        matches = self.matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        pts1 = []
        pts2 = []

        for m, n in matches:
            if len(m, n) != 2:
                continue  # on saute les cas où il n'y a pas 2 matches

            if m.distance < ratio * n.distance:
                good_matches.append(m)
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)

        return good_matches, np.array(pts1), np.array(pts2)