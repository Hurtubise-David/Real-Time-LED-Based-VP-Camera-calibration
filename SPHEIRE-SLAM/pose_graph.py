import numpy as np

class PoseNode:
    def __init__(self, frame_id, pose):
        self.frame_id = frame_id
        self.pose = pose  # 4x4 numpy array

class PoseEdge:
    def __init__(self, from_id, to_id, relative_pose, information=np.eye(6)):
        self.from_id = from_id
        self.to_id = to_id
        self.relative_pose = relative_pose  # 4x4 relative transformation
        self.information = information      # 6x6 information matrix (optional for now)