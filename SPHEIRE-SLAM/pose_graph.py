import numpy as np

class PoseNode:
    def __init__(self, frame_id, pose):
        self.frame_id = frame_id
        self.pose = pose  # 4x4 numpy array