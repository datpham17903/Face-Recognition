import os
import pickle
import numpy as np
import faiss
import config


class FaceDatabase:
    def __init__(self):
        base_index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
        self.index = faiss.IndexIDMap(base_index)
        self.id_to_name: dict[int, str] = {}
        self.next_id: int = 0

    def add_face(self, name: str, embedding: np.ndarray) -> int:
        vec = embedding.reshape(1, -1).astype(np.float32).copy()
        faiss.normalize_L2(vec)
        face_id = self.next_id
        self.index.add_with_ids(vec, np.array([face_id], dtype=np.int64))
        self.id_to_name[face_id] = name
        self.next_id += 1
        return face_id

    def add_faces_batch(self, name: str, embeddings: np.ndarray) -> list[int]:
        vecs = embeddings.astype(np.float32).copy()
        faiss.normalize_L2(vecs)
        n = vecs.shape[0]
        ids = np.arange(self.next_id, self.next_id + n, dtype=np.int64)
        self.index.add_with_ids(vecs, ids)
        assigned_ids = []
        for i in range(n):
            fid = int(ids[i])
            self.id_to_name[fid] = name
            assigned_ids.append(fid)
        self.next_id += n
        return assigned_ids

    def recognize(
        self, embedding: np.ndarray, threshold: float | None = None
    ) -> tuple[str, float]:
        if threshold is None:
            threshold = config.SIMILARITY_THRESHOLD
        if self.index.ntotal == 0:
            return ("Unknown", 0.0)
        vec = embedding.reshape(1, -1).astype(np.float32).copy()
        faiss.normalize_L2(vec)
        distances, ids = self.index.search(vec, config.TOP_K)
        best_dist = float(distances[0][0])
        best_id = int(ids[0][0])
        if best_id == -1:
            return ("Unknown", 0.0)
        if best_dist >= threshold:
            return (self.id_to_name.get(best_id, "Unknown"), best_dist)
        return ("Unknown", best_dist)

    def save(self):
        os.makedirs(config.DATABASE_DIR, exist_ok=True)
        faiss.write_index(self.index, config.FAISS_INDEX_PATH)
        meta = {"id_to_name": self.id_to_name, "next_id": self.next_id}
        with open(config.FAISS_META_PATH, "wb") as f:
            pickle.dump(meta, f)

    def load(self) -> bool:
        if not os.path.exists(config.FAISS_INDEX_PATH):
            return False
        if not os.path.exists(config.FAISS_META_PATH):
            return False
        self.index = faiss.read_index(config.FAISS_INDEX_PATH)
        with open(config.FAISS_META_PATH, "rb") as f:
            meta = pickle.load(f)
        self.id_to_name = meta["id_to_name"]
        self.next_id = meta["next_id"]
        return True

    def total_faces(self) -> int:
        return self.index.ntotal
