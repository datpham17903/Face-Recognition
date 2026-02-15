# Face Recognition System

Real-time face recognition using **InsightFace** (buffalo_l) for detection/embedding and **FAISS** for fast vector similarity search.

## Features

- Face detection and 512-dim embedding extraction (InsightFace buffalo_l)
- FAISS vector database with cosine similarity search
- Build database from LFW (Labeled Faces in the Wild) dataset
- Register new faces via webcam or image file
- Real-time webcam face recognition with bounding boxes and labels

## Setup

```bash
pip install -r requirements.txt
```

> **GPU support**: Replace `onnxruntime` with `onnxruntime-gpu` and `faiss-cpu` with `faiss-gpu`.

## Usage

### 1. Build database from LFW

```bash
python build_database.py
```

Downloads LFW dataset (~233MB), extracts face embeddings, and saves to `database/`.

### 2. Register a face

```bash
# From webcam (press SPACE to capture)
python register_face.py --name "Your Name"

# From image file
python register_face.py --name "Your Name" --image path/to/photo.jpg
```

### 3. Run real-time recognition

```bash
python webcam_recognizer.py
```

Press **ESC** to quit.

## Configuration

Edit `config.py` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SIMILARITY_THRESHOLD` | 0.4 | Cosine similarity threshold for matching |
| `DET_THRESH` | 0.5 | Minimum face detection confidence |
| `FRAME_SKIP` | 2 | Process every N-th frame |
| `CTX_ID` | 0 | GPU device ID (-1 for CPU) |

## Project Structure

```
├── config.py              # Configuration constants
├── face_engine.py         # InsightFace wrapper
├── face_database.py       # FAISS vector database
├── build_database.py      # Populate DB from LFW dataset
├── register_face.py       # Register face via webcam/image
├── webcam_recognizer.py   # Real-time recognition
├── requirements.txt
├── database/              # FAISS index + metadata (auto-created)
└── data/                  # LFW dataset (auto-downloaded)
```

## Tech Stack

- [InsightFace](https://github.com/deepinsight/insightface) — Face detection & embedding
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [OpenCV](https://opencv.org/) — Image/video processing
- [scikit-learn](https://scikit-learn.org/) — LFW dataset loader
