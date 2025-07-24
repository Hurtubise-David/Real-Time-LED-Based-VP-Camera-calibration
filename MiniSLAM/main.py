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

    while True:
        frame = camera.get_left_frame()
        if frame is None:
            continue

        kp, des = tracker.detect(frame)
        matches, pts1, pts2 = tracker.match(prev_kp, prev_des, kp, des)

        if len(pts1) >= 6:
                    R, t = estimate_pose_essential(pts1, pts2, camera.K)
                    if R is not None and t is not None:
                        T = np.eye(4)
                        T[:3, :3] = R
                        T[:3, 3] = t.flatten()

                        pose = pose @ np.linalg.inv(T)  # composition (accumulation)
                        position = pose[:3, 3]
                        trajectory.append(position)

                        print(f"Position: x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f}")

        prev_frame = frame
        prev_kp, prev_des = kp, des

        # ESC pour quitter
        if cv2.waitKey(1) & 0xFF == 27:
            break