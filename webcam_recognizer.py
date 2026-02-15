import time
import cv2
import numpy as np
import config
from face_engine import FaceEngine
from face_database import FaceDatabase


def main():
    engine = FaceEngine()
    db = FaceDatabase()

    if not db.load() or db.total_faces() == 0:
        print("Warning: Database is empty! Run build_database.py or register_face.py first.")

    print(f"Database loaded: {db.total_faces()} faces")

    cap = cv2.VideoCapture(config.WEBCAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_count = 0
    last_results: list[tuple[np.ndarray, str, float]] = []
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % config.FRAME_SKIP == 0:
            detections = engine.extract_embeddings(frame)
            last_results = []
            for bbox, embedding, det_score in detections:
                name, similarity = db.recognize(embedding)
                last_results.append((bbox, name, similarity))

        for bbox, name, similarity in last_results:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{name} ({similarity:.2f})"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1
            )
            cv2.putText(
                frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )

        cv2.imshow("Face Recognition", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
