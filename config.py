"""Configuration for Face Recognition System."""

import os

# ── DeepFace ─────────────────────────────────────────────────
MODEL_NAME = "Facenet512"        # 512-dimensional embeddings
DETECTOR_BACKEND = "opencv"      # opencv | retinaface | mtcnn | ssd
ENFORCE_DETECTION = True         # raise error if no face detected

# ── Embedding ────────────────────────────────────────────────
EMBEDDING_DIM = 512

# ── FAISS / Recognition ─────────────────────────────────────
SIMILARITY_THRESHOLD = 0.4   # cosine similarity threshold for positive match
TOP_K = 1                    # number of nearest neighbours to retrieve

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
FAISS_INDEX_PATH = os.path.join(DATABASE_DIR, "faces.index")
FAISS_META_PATH = os.path.join(DATABASE_DIR, "faces_meta.pkl")
LFW_DATA_DIR = os.path.join(BASE_DIR, "data", "lfw")

# ── Webcam ───────────────────────────────────────────────────
WEBCAM_ID = 0
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
FRAME_SKIP = 2  # process every N-th frame for performance
