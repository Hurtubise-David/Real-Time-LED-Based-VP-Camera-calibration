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

    def match(self, kp1, des1, kp2, des2, ratio=0.8):
        """
        Matche les descripteurs entre deux frames avec le ratio test
        :param kp1: keypoints image t-1
        :param des1: descripteurs t-1
        :param kp2: keypoints image t
        :param des2: descripteurs t
        :return: matches filtrés, pts1, pts2
        """
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return [], np.array([]), np.array([])  # aucun match possible
    
        matches = self.matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        pts1 = []
        pts2 = []

        for m_n in matches:
            if len(m_n) != 2:
                continue  # on saute les cas où il n'y a pas 2 matches

            m, n = m_n 

            if m.distance < ratio * n.distance:
                good_matches.append(m)
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)

        return good_matches, np.array(pts1), np.array(pts2)
    
    def set_parameters(self, ratio, nfeatures, checks):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        index_params = dict(algorithm=6,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=int(checks))
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.ratio = ratio

    def draw_matches(self, img1, kp1, img2, kp2, matches):
        return cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    def draw_inlier_matches(self, img1, kp1, img2, kp2, matches, mask):
        inlier_matches = [m for i, m in enumerate(matches) if mask[i]]
        return self.draw_matches(img1, kp1, img2, kp2, inlier_matches)
    
    def filter_with_ransac(self, pts1, pts2, K):
        """
        Filtre les correspondances avec la méthode RANSAC sur la matrice essentielle.
        :param pts1: points de l'image 1 (Nx2)
        :param pts2: points de l'image 2 (Nx2)
        :param K: matrice intrinsèque de la caméra
        :return: inliers1, inliers2, mask (mask est booléen pour les matches valides)
        """
        if len(pts1) < 5 or len(pts2) < 5:
            return pts1, pts2, None  # pas assez de points

        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        if E is None or mask is None:
            return pts1, pts2, None

        inliers1 = pts1[mask.ravel() == 1]
        inliers2 = pts2[mask.ravel() == 1]
        inlier_indices = np.where(mask.ravel() == 1)[0]
        return inliers1, inliers2, mask.ravel().astype(bool), inlier_indices