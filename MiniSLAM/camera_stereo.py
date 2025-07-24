import cv2
import numpy as np


class StereoCamera:
    def __init__(self, camera_id=0, width=1280, height=480, fps=60)