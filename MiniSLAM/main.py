from camera_stereo import StereoCamera
from feature_tracker import ORBTracker
from pose_estimation import estimate_pose_essential
import numpy as np
import cv2


def main():
    camera = StereoCamera()
    tracker = ORBTracker()

    pose = np.eye(4) # Matrice 4x4 de la pose cumulée (monde → caméra)
    trajectory = []

    prev_frame = camera.get_left_frame()
    prev_kp, prev_des = tracker.detect(prev_frame)
