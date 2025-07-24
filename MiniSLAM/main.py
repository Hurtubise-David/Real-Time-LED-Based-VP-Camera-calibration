from camera_stereo import StereoCamera
from feature_tracker import ORBTracker
from pose_estimation import estimate_pose_essential
from visualization import Visualizer
import numpy as np
import cv2

# Variables globales
pose = np.eye(4) # Matrice 4x4 de la pose cumulée (monde → caméra)
trajectory = []

# Callback pour reset
def reset_pose():
    global pose, trajectory
    pose = np.eye(4)
    trajectory = []
    print("Camera pose reset to (0, 0, 0)")

# Lancer la UI
ui = Visualizer(reset_callback=reset_pose)

def main():
    camera = StereoCamera()
    tracker = ORBTracker()

    prev_frame = camera.get_left_frame()
    prev_kp, prev_des = tracker.detect(prev_frame)

    while True:

        frame = camera.get_left_frame()
        if frame is None:
            continue

        kp, des = tracker.detect(frame)
        matches, pts1, pts2 = tracker.match(prev_kp, prev_des, kp, des)

        position = np.array([0.0, 0.0, 0.0])  # <- définir une valeur par défaut avant l'affichage

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

        ui.update_view(frame, trajectory, position)
        
        prev_frame = frame
        prev_kp, prev_des = kp, des

        # ESC pour quitter
        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()

    # Sauvegarde de la trajectoire pour visualisation
    np.savetxt("trajectory.txt", np.array(trajectory))

if __name__ == "__main__":
    main()