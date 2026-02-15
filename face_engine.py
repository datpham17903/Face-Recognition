import numpy as np
import cv2
from deepface import DeepFace
import config


class FaceEngine:
    def __init__(self):
        self.model_name = config.MODEL_NAME
        self.detector = config.DETECTOR_BACKEND

    def detect_faces(self, img_bgr: np.ndarray) -> list:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            result = DeepFace.represent(
                img_path=img_rgb,
                model_name=self.model_name,
                detector_backend=self.detector,
                enforce_detection=config.ENFORCE_DETECTION,
            )
            faces = []
            for detection in result:
                area = detection.get("facial_area", {})
                bbox = [area.get("x", 0), area.get("y", 0), area.get("w", 0), area.get("h", 0)]
                faces.append(bbox)
            return faces
        except Exception:
            return []

    def extract_embeddings(
        self, img_bgr: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            result = DeepFace.represent(
                img_path=img_rgb,
                model_name=self.model_name,
                detector_backend=self.detector,
                enforce_detection=config.ENFORCE_DETECTION,
            )
            embeddings = []
            for detection in result:
                area = detection.get("facial_area", {})
                x, y, w, h = area.get("x", 0), area.get("y", 0), area.get("w", 0), area.get("h", 0)
                bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
                embedding = np.array(detection["embedding"], dtype=np.float32)
                confidence = detection.get("confidence", 1.0)
                embeddings.append((bbox, embedding, confidence))
            return embeddings
        except Exception:
            return []
