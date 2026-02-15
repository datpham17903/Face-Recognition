import cv2
import platform


def list_available_cameras(max_check=10):
    available = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available


def get_camera_name(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        cap.release()
        return f"Camera {camera_id}"
    
    if platform.system() == "Windows":
        name = cap.get(cv2.CAP_PROP_BACKEND)
        if name:
            return f"Camera {camera_id} ({name})"
    
    cap.release()
    return f"Camera {camera_id}"


def get_all_cameras_with_names(max_check=10):
    cameras = list_available_cameras(max_check)
    return [(cam_id, get_camera_name(cam_id)) for cam_id in cameras]


def test_camera_connection(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return False, f"Cannot open camera {camera_id}"
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        return False, f"Cannot read frame from camera {camera_id}"
    
    return True, f"Camera {camera_id} OK - {frame.shape}"
