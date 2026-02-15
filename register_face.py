import argparse
import cv2
import config
from face_engine import FaceEngine
from face_database import FaceDatabase


def register_from_image(name: str, image_path: str, engine: FaceEngine, db: FaceDatabase):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}'")
        return False
    results = engine.extract_embeddings(img)
    if not results:
        print("No face detected in the image.")
        return False
    _, embedding, score = results[0]
    face_id = db.add_face(name, embedding)
    db.save()
    print(f"Registered '{name}' (ID: {face_id}, score: {score:.3f})")
    return True


def register_from_webcam(name: str, engine: FaceEngine, db: FaceDatabase):
    cap = cv2.VideoCapture(config.WEBCAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False

    print("Press SPACE to capture, ESC to quit.")
    registered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(
            display, "Press SPACE to capture, ESC to quit",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.imshow("Register Face", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:
            results = engine.extract_embeddings(frame)
            if not results:
                print("No face detected. Try again.")
                continue
            _, embedding, score = results[0]
            face_id = db.add_face(name, embedding)
            db.save()
            print(f"Registered '{name}' (ID: {face_id}, score: {score:.3f})")
            registered = True
            break

    cap.release()
    cv2.destroyAllWindows()
    return registered


def main():
    parser = argparse.ArgumentParser(description="Register a face into the database")
    parser.add_argument("--name", required=True, help="Name of the person")
    parser.add_argument("--image", help="Path to image file (omit for webcam)")
    args = parser.parse_args()

    engine = FaceEngine()
    db = FaceDatabase()
    db.load()

    print(f"Database has {db.total_faces()} faces.")

    if args.image:
        register_from_image(args.name, args.image, engine, db)
    else:
        register_from_webcam(args.name, engine, db)

    print(f"Database now has {db.total_faces()} faces.")


if __name__ == "__main__":
    main()
