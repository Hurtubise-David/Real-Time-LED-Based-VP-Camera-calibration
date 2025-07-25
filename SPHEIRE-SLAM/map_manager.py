import numpy as np

class MapPoint:
    def __init__(self, position, descriptor=None):
        self.position = np.array(position, dtype=np.float32)  # shape (3,)
        self.descriptor = descriptor
        self.observations = {}  # key: keyframe_id, value: keypoint_idx
        self.is_valid = True

    def add_observation(self, keyframe_id, keypoint_idx):
        self.observations[keyframe_id] = keypoint_idx

    def remove_observation(self, keyframe_id):
        if keyframe_id in self.observations:
            del self.observations[keyframe_id]