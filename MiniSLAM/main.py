import cv2
from camera_stereo import StereoCamera
from feature_tracker import ORBTracker

camera = StereoCamera()
tracker = ORBTracker()

prev = camera.get_left_frame()
prev_kp, prev_des = tracker.detect(prev)

while True:
    frame = camera.get_left_frame()
    kp, des = tracker.detect(frame)

    matches, pts1, pts2 = tracker.match(prev_kp, prev_des, kp, des)

    # Dessiner les matches
    img_matches = cv2.drawMatches(prev, prev_kp, frame, kp, matches[:20], None, flags=2)
    cv2.imshow("ORB Feature Matches", img_matches)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    prev = frame
    prev_kp, prev_des = kp, des

camera.release()
cv2.destroyAllWindows()