import numpy as np
import cv2
from insightface.app import FaceAnalysis
import config


class FaceEngine:
    def __init__(self):
        self.model_name = config.MODEL_NAME
        self.detector = config.DETECTOR_BACKEND
        
        # Initialize InsightFace with buffalo_l model for GPU acceleration
        # Using ONNX runtime with CUDA for fast inference
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
    def detect_faces(self, img_bgr: np.ndarray) -> list:
        try:
            faces = self.app.get(img_bgr)
            bboxes = []
            for face in faces:
                bbox = [int(face.bbox[0]), int(face.bbox[1]), 
                        int(face.bbox[2] - face.bbox[0]), 
                        int(face.bbox[3] - face.bbox[1])]
                bboxes.append(bbox)
            return bboxes
        except Exception:
            return []

    def extract_embeddings(
        self, img_bgr: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        try:
            faces = self.app.get(img_bgr)
            embeddings = []
            for face in faces:
                # bbox format: [x1, y1, x2, y2]
                bbox = np.array(face.bbox, dtype=np.float32)
                # embedding is already normalized in InsightFace
                embedding = np.array(face.embedding, dtype=np.float32)
                # det_score is detection confidence
                confidence = float(face.det_score) if hasattr(face, 'det_score') else 1.0
                embeddings.append((bbox, embedding, confidence))
            return embeddings
        except Exception:
            return []
