from camera_stereo import StereoCamera
from feature_tracker import ORBTracker
from pose_estimation import estimate_pose_essential
from visualization import Visualizer
import numpy as np
import cv2
import threading

shared_state = {
    "pose": np.eye(4),
    "trajectory": [],
    "frame": None,
    "position": np.array([0.0, 0.0, 0.0]),
    "matches_img": None,
    "ransac_img": None
}

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

    def update_tracker_params(ratio, nfeatures, checks):
        tracker.set_parameters(ratio, int(nfeatures), int(checks))

    ui.set_param_callback(update_tracker_params)

    def update():
        nonlocal prev_frame, prev_kp, prev_des
        frame = camera.get_left_frame()
        if frame is not None:
            kp, des = tracker.detect(frame)
            matches, pts1, pts2 = tracker.match(prev_kp, prev_des, kp, des)

            pts1_np = np.array(pts1)
            pts2_np = np.array(pts2)
            matches_img = tracker.draw_matches(prev_frame, prev_kp, frame, kp, matches)
            inliers1, inliers2, ransac_mask = tracker.filter_with_ransac(pts1_np, pts2_np, camera.K)
            if ransac_mask is not None:
                ransac_img = tracker.draw_inlier_matches(prev_frame, prev_kp, frame, kp, matches, ransac_mask)
            else:
                ransac_img = matches_img.copy()

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

            ui.update_view(frame, trajectory, position, matches_img, ransac_img)
            prev_frame = frame
            prev_kp, prev_des = kp, des

        
    # Lancer dans un thread
    threading.Thread(target=update, daemon=True).start()
    ui.run()

    camera.release()
    np.savetxt("trajectory.txt", np.array(trajectory))

if __name__ == "__main__":
    main()
