import numpy as np

class MapPoint:

    #Représente un point 3D triangulé dans l'espace monde.
    
    def __init__(self, position, descriptor=None):
        self.position = np.array(position, dtype=np.float32)  # Coordonnées 3D [X, Y, Z]
        self.descriptor = descriptor  # Descripteur ORB associé (pour association)
        self.observations = {}  # Liste des keyframes et points d'image où ce point a été observé
        self.is_valid = True

    def add_observation(self, keyframe_id, keypoint_idx):
        self.observations[keyframe_id] = keypoint_idx # Associe ce point à une keyframe et un index de keypoint (dans cette image).

    def remove_observation(self, keyframe_id):
        if keyframe_id in self.observations:
            del self.observations[keyframe_id]

    def num_observations(self):
        return len(self.observations)


class Keyframe:

    # Représente une image clé avec sa pose et ses points détectés.

    def __init__(self, frame_id, pose, keypoints, descriptors):
        self.frame_id = frame_id
        self.pose = pose  # 4x4 matrix de transformation
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.map_point_indices = [-1] * len(keypoints)  # -1 si pas associé

    def set_mappoint(self, kp_idx, mp_idx):
        self.map_point_indices[kp_idx] = mp_idx

    def get_linked_mappoints(self):
        return [idx for idx in self.map_point_indices if idx != -1]

class MapManager:
    def __init__(self):
        self.map_points = []
        self.keyframes = []

    def add_keyframe(self, keyframe):
        self.keyframes.append(keyframe)
        return len(self.keyframes) - 1
    
    def reset(self):
        self.keyframes.clear()
        self.mappoints.clear()
        self.next_kf_id = 0
        self.next_mp_id = 0

    def add_mappoint(self, mappoint):
        self.map_points.append(mappoint)
        return len(self.map_points) - 1

    def get_valid_mappoints(self):
        return [mp for mp in self.map_points if mp.is_valid]