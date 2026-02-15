import numpy as np
from insightface.app import FaceAnalysis
import config


class FaceEngine:
    def __init__(self):
        self.app = FaceAnalysis(
            name=config.MODEL_NAME,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(
            ctx_id=config.CTX_ID,
            det_size=config.DET_SIZE,
            det_thresh=config.DET_THRESH,
        )
        warmup_img = np.zeros(
            (config.DET_SIZE[0], config.DET_SIZE[1], 3), dtype=np.uint8
        )
        self.app.get(warmup_img)

    def detect_faces(self, img_bgr: np.ndarray) -> list:
        return self.app.get(img_bgr)

    def extract_embeddings(
        self, img_bgr: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        faces = self.app.get(img_bgr)
        results = []
        for face in faces:
            bbox = face.bbox.astype(np.float32)
            embedding = face.normed_embedding.astype(np.float32)
            score = float(face.det_score)
            results.append((bbox, embedding, score))
        return results
