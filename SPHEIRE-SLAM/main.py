from camera_stereo import StereoCamera
from feature_tracker import ORBTracker
from pose_estimation import estimate_pose_essential
from visualization import Visualizer
from triangulation import triangulate_points
import numpy as np
import cv2
import threading
import time

shared_state = {
    "pose": np.eye(4),
    "trajectory": [],
    "frame": None,
    "position": np.array([0.0, 0.0, 0.0]),
    "matches_img": None,
    "ransac_img": None,
    "map_points": []  
}

def reset_pose():
    shared_state["pose"] = np.eye(4)
    shared_state["trajectory"] = []
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

    ui = Visualizer(reset_callback=reset_pose, shared_state=shared_state)

    def update_tracker_params(ratio, nfeatures, checks):
        tracker.set_parameters(ratio, int(nfeatures), int(checks))

    ui.set_param_callback(update_tracker_params)

    def update():
        nonlocal prev_frame, prev_kp, prev_des
        while True:  # 🔁 Boucle infinie ici pour garder le thread en vie
            frame = camera.get_left_frame()
            if frame is not None:
                kp, des = tracker.detect(frame)
                matches, pts1, pts2 = tracker.match(prev_kp, prev_des, kp, des)

                pts1_np = np.array(pts1)
                pts2_np = np.array(pts2)
                matches_img = tracker.draw_matches(prev_frame, prev_kp, frame, kp, matches)
                inliers1, inliers2, ransac_mask = tracker.filter_with_ransac(pts1_np, pts2_np, camera.K_left)

                # Triangulation si on a assez d'inliers
                if len(inliers1) >= 6 and len(inliers2) >= 6:
                    P_left = camera.K_left @ np.hstack((np.eye(3), np.zeros((3,1))))
                    P_right = camera.K_right @ np.hstack((camera.R, camera.T.reshape(3,1)))
                    
                    # 1. Triangulation stéréo
                    points_3d = triangulate_points(np.array(inliers1), np.array(inliers2), P_left, P_right)

                    # 2. Transformation des points dans le repère monde
                    ones = np.ones((points_3d.shape[0], 1))
                    points_homogeneous = np.hstack((points_3d, ones)).T  # shape (4, N)

                    # 3. Obtenir la pose actuelle de la caméra (copie, pas référence)
                    pose = shared_state["pose"].copy()  # (4, 4)

                    # 4. Projeter dans le monde
                    points_world = (pose @ points_homogeneous).T[:, :3]  # shape (N, 3)

                    # 5. Ajouter à la carte
                    shared_state.setdefault("map_points", []).extend(points_world)

                    # Conversion en liste de points (filtrés)
                    for pt in points_3d:
                        if np.isfinite(pt).all():
                            shared_state["map_points"].append(pt)
                    

                if ransac_mask is not None:
                    ransac_img = tracker.draw_inlier_matches(prev_frame, prev_kp, frame, kp, matches, ransac_mask)
                else:
                    ransac_img = matches_img.copy()

                position = np.array([0.0, 0.0, 0.0])
                if len(pts1) >= 6:
                    R, t = estimate_pose_essential(pts1, pts2, camera.K_left)
                    if R is not None and t is not None:
                        T = np.eye(4)
                        T[:3, :3] = R
                        T[:3, 3] = t.flatten()
                        shared_state["pose"][:] = shared_state["pose"] @ np.linalg.inv(T)
                        position = shared_state["pose"][:3, 3]
                        shared_state["trajectory"].append(position)
                        print(f"Position: x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f}")

                shared_state["frame"] = frame
                shared_state["position"] = position
                shared_state["matches_img"] = matches_img
                shared_state["ransac_img"] = ransac_img
                prev_frame = frame
                prev_kp, prev_des = kp, des

            # pour éviter de boucler trop vite
            time.sleep(0.01)  
        
    # Lancer dans un thread
    threading.Thread(target=update, daemon=True).start()
    ui.run()

    camera.release()
    np.savetxt("trajectory.txt", np.array(trajectory))

if __name__ == "__main__":
    main()
