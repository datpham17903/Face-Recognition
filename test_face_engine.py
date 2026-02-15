"""
Test script for FaceEngine with real webcam capture
"""
import cv2
import numpy as np
from face_engine import FaceEngine
import config

def test_face_engine():
    print("=== FaceEngine Test ===\n")
    
    # Initialize engine
    print("1. Initializing FaceEngine...")
    engine = FaceEngine()
    print(f"   Model: {engine.model_name}")
    print(f"   Detector: {engine.detector}")
    print("   [OK] Initialized\n")
    
    # Capture from webcam
    print("2. Capturing image from webcam...")
    print("   Press SPACE to capture, ESC to exit")
    
    cap = cv2.VideoCapture(config.WEBCAM_ID)
    if not cap.isOpened():
        print("   [ERROR] Failed to open webcam")
        return
    
    img_bgr = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("   [ERROR] Failed to read frame")
            break
        
        # Display instructions
        cv2.putText(frame, "Press SPACE to capture, ESC to exit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Capture Face", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("   [CANCELLED] Cancelled by user")
            break
        elif key == 32:  # SPACE
            img_bgr = frame.copy()
            print("   [OK] Image captured\n")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if img_bgr is None:
        print("No image captured. Exiting.")
        return
    
    # Test face detection
    print("3. Testing face detection...")
    faces = engine.detect_faces(img_bgr)
    print(f"   Found {len(faces)} face(s)")
    for i, bbox in enumerate(faces):
        print(f"   Face {i+1}: bbox={bbox}")
    
    if len(faces) == 0:
        print("   [WARNING] No faces detected. Try again with better lighting or positioning.\n")
        return
    
    print("   [OK] Detection successful\n")
    
    # Test embedding extraction
    print("4. Testing embedding extraction...")
    results = engine.extract_embeddings(img_bgr)
    print(f"   Extracted {len(results)} embedding(s)")
    
    for i, (bbox, embedding, confidence) in enumerate(results):
        print(f"   Face {i+1}:")
        print(f"      BBox: {bbox}")
        print(f"      Embedding shape: {embedding.shape}")
        print(f"      Embedding dimension: {len(embedding)}")
        print(f"      Confidence: {confidence:.4f}")
        print(f"      First 5 values: {embedding[:5]}")
        
        # Verify dimension
        if len(embedding) == 512:
            print(f"      [OK] Embedding dimension is correct (512)")
        else:
            print(f"      [ERROR] Expected 512 dimensions, got {len(embedding)}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_face_engine()
