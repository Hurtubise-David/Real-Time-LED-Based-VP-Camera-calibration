import cv2
import numpy as np

Class ORBTracker:
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