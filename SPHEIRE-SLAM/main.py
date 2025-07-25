from camera_stereo import StereoCamera
from feature_tracker import ORBTracker
from pose_estimation import estimate_pose_essential
from visualization import Visualizer
from triangulation import triangulate_points
from map_manager import MapManager, Keyframe, MapPoint
from pose_graph import PoseGraph
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
    "map_points": [],
    "frame_id": 0,
    "pose_graph": {"nodes": [], "edges": []}  
}

map_manager = MapManager()  # Gestionnaire central de la carte 3D
pose_graph = PoseGraph()  # instancier le graphe global

def reset_pose():
    shared_state["pose"] = np.eye(4)
    shared_state["trajectory"] = []
    shared_state["map_points"] = []
    shared_state["frame_id"] = 0
    map_manager.reset()  
    print("Camera pose and map reset to initial state.")

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
        while True:  
            frame_left, frame_right = camera.get_frames()
            if frame_left is not None and frame_right is not None:
                # === 1. Tracking temporel (frame_left(t-1) vs frame_left(t)) ===
                kp_left, des_left = tracker.detect(frame_left)
                matches_mono, pts1, pts2 = tracker.match(prev_kp, prev_des, kp_left, des_left)

                pts1_np = np.array(pts1)
                pts2_np = np.array(pts2)
                matches_img = tracker.draw_matches(prev_frame, prev_kp, frame_left, kp_left, matches_mono)

                inliers1, inliers2, ransac_mask, _ = tracker.filter_with_ransac(pts1_np, pts2_np, camera.K_left)

                # === Ajout d'un keyframe ===
                kf = Keyframe(
                    frame_id=shared_state["frame_id"],
                    pose=shared_state["pose"].copy(),  # On copie pour éviter les références
                    keypoints=kp_left,
                    descriptors=des_left
                )
                kf_id = map_manager.add_keyframe(kf)
                shared_state["frame_id"] += 1

                # Ajouter un nœud au graphe
                pose_graph.add_node(kf_id, shared_state["pose"].copy())

                # Si on a déjà un keyframe précédent, ajouter une arête
                if kf_id > 0:
                    relative_pose = np.linalg.inv(map_manager.keyframes[kf_id - 1].pose) @ shared_state["pose"]
                    pose_graph.add_edge(kf_id - 1, kf_id, relative_pose)

                shared_state["pose_graph"]["nodes"] = [T[:3, 3].tolist() for T in pose_graph.nodes]
                shared_state["pose_graph"]["edges"] = [(a, b) for (a, b, _) in pose_graph.edges]


                # === Triangulation stéréo (frame_left vs frame_right) ===
                kp_r, des_r = tracker.detect(frame_right)
                matches_stereo, ptsL, ptsR = tracker.match(kp_left, des_left, kp_r, des_r)

                if len(ptsL) >= 6 and len(ptsR) >= 6:
                    inliersL, inliersR, stereo_mask, inlier_indices = tracker.filter_with_ransac(np.array(ptsL), np.array(ptsR), camera.K_left)

                    # Triangulation si on a assez d'inliers
                    if len(inliersL) >= 6:
                        P_left = camera.K_left @ np.hstack((np.eye(3), np.zeros((3,1))))
                        P_right = camera.K_right @ np.hstack((camera.R, camera.T.reshape(3,1)))
                        
                        # 1. Triangulation stéréo
                        points_3d = triangulate_points(np.array(inliersL), np.array(inliersR), P_left, P_right)

                        # 2. Transformation des points dans le repère monde
                        ones = np.ones((points_3d.shape[0], 1))
                        points_homogeneous = np.hstack((points_3d, ones)).T  # shape (4, N)

                        # 3. Obtenir la pose actuelle de la caméra (copie, pas référence)
                        pose = shared_state["pose"].copy()  # (4, 4)

                        # 4. Projeter dans le monde
                        points_world = (pose @ points_homogeneous).T[:, :3]  # shape (N, 3)

                        # 5. Ajouter les MapPoints à la carte SLAM (MapManager)
                        for i, pt in enumerate(points_world):
                            if np.isfinite(pt).all():
                                try:
                                    idx_inlier = inlier_indices[i]
                                    match = matches_stereo[idx_inlier]  # accès au DMatch
                                    descriptor = des_left[match.queryIdx]  # descripteur du keypoint gauche
                                except IndexError:
                                    descriptor = None  # fallback
                                mp = MapPoint(position=pt, descriptor=descriptor)
                                mp_id = map_manager.add_mappoint(mp)

                                try:
                                    kf.set_mappoint(match.queryIdx, mp_id)
                                    mp.add_observation(kf_id, match.queryIdx)
                                except Exception:
                                    pass  # évite le crash si match n'est pas défini
                                                            

                if ransac_mask is not None:
                    ransac_img = tracker.draw_inlier_matches(prev_frame, prev_kp, frame_left, kp_left, matches_mono, ransac_mask)
                else:
                    ransac_img = matches_img.copy()

                # === Estimation de pose par Essential Matrix (motion) ===
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

                # === UI update ===
                shared_state["frame"] = frame_left
                shared_state["position"] = position
                shared_state["matches_img"] = matches_img
                shared_state["ransac_img"] = ransac_img
                shared_state["map_points"] = [mp.position.tolist() for mp in map_manager.get_valid_mappoints()]

                prev_frame = frame_left
                prev_kp, prev_des = kp_left, des_left

            # pour éviter de boucler trop vite
            time.sleep(0.01)  
        
    # Lancer dans un thread
    threading.Thread(target=update, daemon=True).start()
    ui.run()

    camera.release()
    np.savetxt("trajectory.txt", np.array(trajectory))

if __name__ == "__main__":
    main()
