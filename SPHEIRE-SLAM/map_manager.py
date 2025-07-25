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


class Keyframe:
    def __init__(self, frame_id, pose, keypoints, descriptors):
        self.frame_id = frame_id
        self.pose = pose  # 4x4 matrix de transformation
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.map_point_indices = [-1] * len(keypoints)  # -1 si pas associé

    def set_mappoint(self, kp_idx, mp_idx):
        self.map_point_indices[kp_idx] = mp_idx

class MapManager:
    def __init__(self):
        self.map_points = []
        self.keyframes = []

    def add_keyframe(self, keyframe):
        self.keyframes.append(keyframe)
        return len(self.keyframes) - 1

    def add_mappoint(self, mappoint):
        self.map_points.append(mappoint)
        return len(self.map_points) - 1

    def get_valid_mappoints(self):
        return [mp for mp in self.map_points if mp.is_valid]