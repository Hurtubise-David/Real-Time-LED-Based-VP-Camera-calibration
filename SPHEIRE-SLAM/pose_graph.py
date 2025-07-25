import numpy as np

class PoseNode:
    def __init__(self, frame_id, pose):
        self.frame_id = frame_id
        self.pose = pose  # 4x4 numpy array

class PoseEdge:
    def __init__(self, from_id, to_id, relative_pose, information=np.eye(6)):
        self.from_id = from_id
        self.to_id = to_id
        self.relative_pose = relative_pose  # 4x4 transformation relative
        self.information = information      # 6x6 information matrix (optionel)

class PoseGraph:
    def __init__(self):
        self.nodes = {}  # key: frame_id, value: PoseNode
        self.edges = []  # liste de PoseEdge

    def add_node(self, frame_id, pose):
        if frame_id not in self.nodes:
            self.nodes[frame_id] = PoseNode(frame_id, pose)