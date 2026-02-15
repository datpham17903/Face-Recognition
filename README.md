# Face Recognition System

Real-time face recognition using **InsightFace** (buffalo_l) with GPU acceleration via ONNX Runtime CUDA, and **FAISS** for fast vector similarity search.

## Features

- GPU-accelerated face detection and 512-dim embedding extraction (InsightFace buffalo_l)
- FAISS vector database with cosine similarity search
- Build database from LFW (Labeled Faces in the Wild) dataset
- Register new faces via webcam or image file
- Real-time webcam face recognition with bounding boxes and labels
- Camera selection dropdown for multiple cameras
- Optimized CPU usage for GUI

## Setup

```bash
pip install -r requirements.txt
```

> **First Run**: InsightFace will download buffalo_l models (~100MB) on first use.  
> **GPU**: Requires NVIDIA GPU with CUDA support. Install `onnxruntime-gpu` for GPU acceleration.

## Usage

### 🎯 Quick Start (GUI App)

```bash
python app.py
```

The GUI provides three tabs:
- **Register Face**: Capture from webcam or load from image file
- **Live Recognition**: Real-time face recognition with bounding boxes
- **Database**: Manage registered faces, build from LFW dataset

---

### 💻 Command Line (Alternative)

#### 1. Build database from LFW

```bash
python build_database.py
```

Downloads LFW dataset (~233MB), extracts face embeddings, and saves to `database/`.

#### 2. Register a face

```bash
# From webcam (press SPACE to capture)
python register_face.py --name "Your Name"

# From image file
python register_face.py --name "Your Name" --image path/to/photo.jpg
```

#### 3. Run real-time recognition

```bash
python webcam_recognizer.py
```

Press **ESC** to quit.

## Configuration

Edit `config.py` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SIMILARITY_THRESHOLD` | 0.4 | Cosine similarity threshold for matching |
| `FRAME_SKIP` | 5 | Process every N-th frame (higher = faster) |
| `WEBCAM_WIDTH` | 640 | Webcam frame width |
| `WEBCAM_HEIGHT` | 480 | Webcam frame height |

## Project Structure

```
├── app.py                 # 🎯 GUI Application (tkinter)
├── config.py              # Configuration constants
├── face_engine.py         # InsightFace GPU wrapper
├── face_database.py       # FAISS vector database
├── camera_utils.py        # Camera detection utilities
├── build_database.py      # Populate DB from LFW dataset
├── register_face.py       # Register face via webcam/image (CLI)
├── webcam_recognizer.py   # Real-time recognition (CLI)
├── requirements.txt
├── database/              # FAISS index + metadata (auto-created)
└── data/                 # LFW dataset (auto-downloaded)
```

## Tech Stack

- [InsightFace](https://github.com/deepinsight/insightface) — Face detection & embedding (buffalo_l)
- [ONNX Runtime](https://onnxruntime.ai/) — GPU acceleration via CUDA
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [OpenCV](https://opencv.org/) — Image/video processing
- [scikit-learn](https://scikit-learn.org/) — LFW dataset loader
