from camera_stereo import StereoCamera
from feature_tracker import ORBTracker
from pose_estimation import estimate_pose_essential
from visualization import Visualizer
import numpy as np
import cv2

# Variables globales
pose = np.eye(4)
trajectory = []

def reset_pose():
    global pose, trajectory
    pose = np.eye(4)
    trajectory = []
    print("Camera pose reset to (0, 0, 0)")

def main():
    global pose, trajectory

    camera = StereoCamera()
    tracker = ORBTracker()

    # Attendre une première image valide
    prev_frame = None
    while prev_frame is None:
        prev_frame = camera.get_left_frame()

    prev_kp, prev_des = tracker.detect(prev_frame)

    ui = Visualizer(reset_callback=reset_pose)

    def update():
        nonlocal prev_frame, prev_kp, prev_des
        frame = camera.get_left_frame()
        if frame is not None:
            kp, des = tracker.detect(frame)
            matches, pts1, pts2 = tracker.match(prev_kp, prev_des, kp, des)

            position = np.array([0.0, 0.0, 0.0])
            if len(pts1) >= 6:
                R, t = estimate_pose_essential(pts1, pts2, camera.K)
                if R is not None and t is not None:
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = t.flatten()
                    pose[:] = pose @ np.linalg.inv(T)
                    position = pose[:3, 3]
                    trajectory.append(position)
                    print(f"Position: x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f}")

            ui.update_view(frame, trajectory, position)
            prev_frame = frame
            prev_kp, prev_des = kp, des

        ui.root.after(1, update)

    ui.root.after(1, update)
    ui.run()

    camera.release()
    np.savetxt("trajectory.txt", np.array(trajectory))

if __name__ == "__main__":
    main()
