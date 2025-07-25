import cv2
import numpy as np

def triangulate_points(pts_left, pts_right, P_left, P_right):
    """
    Triangule des points 2D correspondants entre deux images stéréo.

    Arguments :
    - pts_left : np.ndarray de shape (N, 2), points dans l'image gauche
    - pts_right : np.ndarray de shape (N, 2), points dans l'image droite
    - P_left : Matrice de projection 3x4 de la caméra gauche
    - P_right : Matrice de projection 3x4 de la caméra droite

    Retourne :
    - points_3d : np.ndarray de shape (N, 3), points 3D triangulés
    """

    pts_left_h = cv2.convertPointsToHomogeneous(pts_left).reshape(-1, 3).T
    pts_right_h = cv2.convertPointsToHomogeneous(pts_right).reshape(-1, 3).T

    points_4d = cv2.triangulatePoints(P_left, P_right, pts_left.T, pts_right.T)
    points_3d = (points_4d / points_4d[3])[:3].T  # homogène → cartésien

    return points_3d