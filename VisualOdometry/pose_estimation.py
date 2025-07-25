import cv2
import numpy as np

def estimate_pose_essential(pts1, pts2, K):
    """
    Estime la pose caméra entre deux frames à partir de points 2D.
    :param pts1: points 2D dans l'image précédente
    :param pts2: points 2D dans l'image actuelle
    :param K: matrice intrinsèque de la caméra
    :return: R (3x3), t (3x1)
    """
    if len(pts1) < 6 or len(pts2) < 6:
        return None, None  # Pas assez de points

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, None

    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    return R, t