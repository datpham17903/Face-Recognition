import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people
import config
from face_engine import FaceEngine
from face_database import FaceDatabase


def main():
    print("Loading LFW dataset...")
    lfw = fetch_lfw_people(
        min_faces_per_person=20, resize=1.0, data_home=config.LFW_DATA_DIR
    )
    n_images = lfw.images.shape[0]
    n_people = len(lfw.target_names)
    print(f"Dataset: {n_images} images, {n_people} people")

    engine = FaceEngine()
    db = FaceDatabase()
    db.load()

    success = 0
    fail = 0

    for i in range(n_images):
        img_gray = lfw.images[i]
        # LFW from sklearn: float64 [0,1] grayscale -> uint8 [0,255] BGR
        gray_uint8 = (img_gray * 255).astype(np.uint8)
        bgr = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)

        results = engine.extract_embeddings(bgr)
        if results:
            bbox, embedding, score = results[0]
            person_name = lfw.target_names[lfw.target[i]]
            db.add_face(person_name, embedding)
            success += 1
        else:
            fail += 1

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{n_images} images...")

    db.save()
    print(f"\nDone! Added {success} faces, {fail} failed detections")
    print(f"Database size: {db.total_faces()} faces")


if __name__ == "__main__":
    main()
