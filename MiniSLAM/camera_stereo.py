import cv2
import numpy as np


class StereoCamera:
    def __init__(self, camera_id=0, width=1280, height=480, fps=60)
        """
        Initialise une caméra stéréo avec flux côte à côte (ELP).
        :param camera_id: index de la caméra combinée
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Paramètres de calibration fictifs (remplacer par réels plus tard)
        self.K = np.array([[640, 0, 320],
                           [0, 640, 240],
                           [0,   0,   1]], dtype=np.float32)
