import cv2
import numpy as np


class StereoCamera:
    def __init__(self, camera_id=5, width=1280, height=480, fps=60):
        """
        Initialise une caméra stéréo avec flux côte à côte (ELP).
        :param camera_id: index de la caméra combinée
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Paramètres de calibration fictifs (remplacer par réels plus tard)
        self.K_left = np.array([[640, 0, 320],
                           [0, 640, 240],
                           [0,   0,   1]], dtype=np.float32)
        
        self.K_right = np.array([[640, 0, 320],
                           [0, 640, 240],
                           [0,   0,   1]], dtype=np.float32)
        
        # Baseline a charger de YAML To DOOOO : self.baseline = float(data["baseline"])
        self.baseline = 0.06  # baseline de 6 cm (à ajuster selon la StereoCamera)  
        self.R = np.eye(3)
        self.T = np.array([-self.baseline, 0, 0])  # translation droite par rapport à gauche      

    def get_frames(self):
        """
        Capture une image combinée et la découpe en deux (gauche/droite)
        :return: (frame_left, frame_right)
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        h, w, _ = frame.shape
        w_half = w // 2
        frame_left = frame[:, :w_half]
        frame_right = frame[:, w_half:]

        return frame_left, frame_right
    
    def get_left_frame(self):
        left, _ = self.get_frames()
        return left

    def release(self):
        self.cap.release()

