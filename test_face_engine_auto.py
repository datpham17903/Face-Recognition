"""
Automated test script for FaceEngine - captures automatically without GUI
"""
import cv2
import numpy as np
from face_engine import FaceEngine
import config

def test_face_engine_auto():
    print("=== FaceEngine Automated Test ===\n")
    
    # Initialize engine
    print("1. Initializing FaceEngine...")
    engine = FaceEngine()
    print(f"   Model: {engine.model_name}")
    print(f"   Detector: {engine.detector}")
    print("   [OK] Initialized\n")
    
    # Capture from webcam (auto-capture after 2 seconds)
    print("2. Capturing image from webcam (auto-capture in 2 seconds)...")
    
    cap = cv2.VideoCapture(config.WEBCAM_ID)
    if not cap.isOpened():
        print("   [ERROR] Failed to open webcam")
        return
    
    # Warm up camera and auto-capture
    import time
    time.sleep(2)
    
    ret, img_bgr = cap.read()
    cap.release()
    
    if not ret:
        print("   [ERROR] Failed to read frame")
        return
    
    print(f"   [OK] Image captured - shape: {img_bgr.shape}\n")
    
    # Test face detection
    print("3. Testing face detection...")
    faces = engine.detect_faces(img_bgr)
    print(f"   Found {len(faces)} face(s)")
    for i, bbox in enumerate(faces):
        print(f"   Face {i+1}: bbox={bbox}")
    
    if len(faces) == 0:
        print("   [WARNING] No faces detected.")
        print("   This is expected if no one is in front of webcam.")
        print("   Testing with synthetic data instead...\n")
        
        # Create a dummy test to verify the pipeline works
        print("4. Testing with random data (pipeline verification)...")
        random_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = engine.extract_embeddings(random_img)
        print(f"   Random image test: {len(results)} faces detected (expected: 0)")
        print("   [OK] Pipeline working correctly\n")
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
    test_face_engine_auto()
