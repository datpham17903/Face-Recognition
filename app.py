import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from face_engine import FaceEngine
from face_database import FaceDatabase
from camera_utils import list_available_cameras, get_camera_name, test_camera_connection
import config


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("900x700")
        
        self.engine = FaceEngine()
        self.db = FaceDatabase()
        self.db.load()
        
        self.cap = None
        self.is_running = False
        self.current_frame = None
        
        # Camera selection
        self.available_cameras = list_available_cameras()
        self.selected_camera_register = self.available_cameras[0] if self.available_cameras else 0
        self.selected_camera_recognize = self.available_cameras[0] if self.available_cameras else 0
        
        self.create_widgets()
        
    def create_widgets(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.tab_register = tk.Frame(notebook)
        self.tab_recognize = tk.Frame(notebook)
        self.tab_database = tk.Frame(notebook)
        
        notebook.add(self.tab_register, text="Register Face")
        notebook.add(self.tab_recognize, text="Live Recognition")
        notebook.add(self.tab_database, text="Database")
        
        self.setup_register_tab()
        self.setup_recognize_tab()
        self.setup_database_tab()
        
    def setup_register_tab(self):
        frame_top = tk.Frame(self.tab_register)
        frame_top.pack(pady=10)
        
        tk.Label(frame_top, text="Name:", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        self.entry_name = tk.Entry(frame_top, font=("Arial", 12), width=20)
        self.entry_name.pack(side=tk.LEFT, padx=5)
        
        frame_camera = tk.Frame(self.tab_register)
        frame_camera.pack(pady=5)
        tk.Label(frame_camera, text="Select Camera:", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)
        
        camera_options = [f"Camera {cam_id}" for cam_id in self.available_cameras] if self.available_cameras else ["Camera 0 (Default)"]
        self.combo_camera_register = ttk.Combobox(frame_camera, values=camera_options, state="readonly", width=15)
        self.combo_camera_register.current(0)
        self.combo_camera_register.pack(side=tk.LEFT, padx=5)
        self.combo_camera_register.bind("<<ComboboxSelected>>", lambda e: self.on_camera_select_register())
        
        self.label_register = tk.Label(self.tab_register, bg="black", text="Camera Preview", fg="white", font=("Arial", 20))
        self.label_register.pack(pady=10)
        
        placeholder = Image.new('RGB', (640, 480), color=(0, 0, 0))
        self.placeholder_imgtk = ImageTk.PhotoImage(placeholder)
        self.label_register.configure(image=self.placeholder_imgtk)
        
        frame_btns = tk.Frame(self.tab_register)
        frame_btns.pack(pady=10)
        
        self.btn_start_register = tk.Button(frame_btns, text="Start Camera", 
                                           command=self.start_register_camera, 
                                           font=("Arial", 12), bg="#4CAF50", fg="white", width=15)
        self.btn_start_register.pack(side=tk.LEFT, padx=5)
        
        self.btn_capture = tk.Button(frame_btns, text="Capture & Register", 
                                     command=self.capture_register, 
                                     font=("Arial", 12), bg="#2196F3", fg="white", 
                                     width=15, state=tk.DISABLED)
        self.btn_capture.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop_register = tk.Button(frame_btns, text="Stop Camera", 
                                          command=self.stop_register_camera, 
                                          font=("Arial", 12), bg="#f44336", fg="white", 
                                          width=15, state=tk.DISABLED)
        self.btn_stop_register.pack(side=tk.LEFT, padx=5)
        
        self.btn_load_image = tk.Button(frame_btns, text="Load from Image", 
                                        command=self.register_from_file, 
                                        font=("Arial", 12), bg="#FF9800", fg="white", width=15)
        self.btn_load_image.pack(side=tk.LEFT, padx=5)
        
    def on_camera_select_register(self):
        selected_idx = self.combo_camera_register.current()
        self.selected_camera_register = self.available_cameras[selected_idx] if self.available_cameras else 0
        
    def setup_recognize_tab(self):
        frame_camera = tk.Frame(self.tab_recognize)
        frame_camera.pack(pady=5)
        tk.Label(frame_camera, text="Select Camera:", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)
        
        camera_options = [f"Camera {cam_id}" for cam_id in self.available_cameras] if self.available_cameras else ["Camera 0 (Default)"]
        self.combo_camera_recognize = ttk.Combobox(frame_camera, values=camera_options, state="readonly", width=15)
        self.combo_camera_recognize.current(0)
        self.combo_camera_recognize.pack(side=tk.LEFT, padx=5)
        self.combo_camera_recognize.bind("<<ComboboxSelected>>", lambda e: self.on_camera_select_recognize())
        
        self.label_recognize = tk.Label(self.tab_recognize, bg="black", text="Recognition Preview", fg="white", font=("Arial", 20))
        self.label_recognize.pack(pady=10)
        
        placeholder = Image.new('RGB', (640, 480), color=(0, 0, 0))
        self.placeholder_imgtk2 = ImageTk.PhotoImage(placeholder)
        self.label_recognize.configure(image=self.placeholder_imgtk2)
        
        frame_btns = tk.Frame(self.tab_recognize)
        frame_btns.pack(pady=10)
        
        self.btn_start_recognize = tk.Button(frame_btns, text="Start Recognition", 
                                            command=self.start_recognition, 
                                            font=("Arial", 12), bg="#4CAF50", fg="white", width=20)
        self.btn_start_recognize.pack(side=tk.LEFT, padx=10)
        
        self.btn_stop_recognize = tk.Button(frame_btns, text="Stop Recognition", 
                                           command=self.stop_recognition, 
                                           font=("Arial", 12), bg="#f44336", fg="white", 
                                           width=20, state=tk.DISABLED)
        self.btn_stop_recognize.pack(side=tk.LEFT, padx=10)
        
        self.label_fps = tk.Label(self.tab_recognize, text="FPS: 0.0", font=("Arial", 14))
        self.label_fps.pack()
        
    def on_camera_select_recognize(self):
        selected_idx = self.combo_camera_recognize.current()
        self.selected_camera_recognize = self.available_cameras[selected_idx] if self.available_cameras else 0
        
    def setup_database_tab(self):
        frame_info = tk.Frame(self.tab_database)
        frame_info.pack(pady=10)
        
        self.label_db_count = tk.Label(frame_info, text=f"Total Faces: {self.db.total_faces()}", 
                                       font=("Arial", 14, "bold"))
        self.label_db_count.pack()
        
        frame_btns = tk.Frame(self.tab_database)
        frame_btns.pack(pady=10)
        
        tk.Button(frame_btns, text="Build from LFW Dataset", 
                 command=self.build_from_lfw, 
                 font=("Arial", 12), bg="#2196F3", fg="white", width=20).pack(pady=5)
        
        tk.Button(frame_btns, text="Build from LFW Folder", 
                 command=self.build_from_lfw_folder, 
                 font=("Arial", 12), bg="#9C27B0", fg="white", width=20).pack(pady=5)
        
        tk.Button(frame_btns, text="Test Camera", 
                 command=self.test_all_cameras, 
                 font=("Arial", 12), bg="#FF5722", fg="white", width=20).pack(pady=5)
        
        tk.Button(frame_btns, text="Clear Database", 
                 command=self.clear_database, 
                 font=("Arial", 12), bg="#f44336", fg="white", width=20).pack(pady=5)
        
        tk.Button(frame_btns, text="Refresh", 
                 command=self.refresh_database, 
                 font=("Arial", 12), bg="#4CAF50", fg="white", width=20).pack(pady=5)
        
        frame_list = tk.Frame(self.tab_database)
        frame_list.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        tk.Label(frame_list, text="Registered People:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        self.listbox = tk.Listbox(frame_list, font=("Arial", 11), height=15)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        
        self.update_person_list()
        
    def start_register_camera(self):
        print(f"Starting camera {self.selected_camera_register}...")
        self.cap = cv2.VideoCapture(self.selected_camera_register)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)
        
        ret, frame = self.cap.read()
        print(f"Read frame: ret={ret}, frame={frame.shape if frame is not None else None}")
        if not ret or frame is None:
            messagebox.showerror("Error", "Cannot read from camera! Try another camera.")
            self.cap.release()
            return
            
        self.is_running = True
        self.btn_start_register.config(state=tk.DISABLED)
        self.btn_stop_register.config(state=tk.NORMAL)
        self.btn_capture.config(state=tk.NORMAL)
        self.update_register_frame()
        
    def update_register_frame(self):
        if not self.is_running or not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if ret and frame is not None:
            print(f"Updating frame: {frame.shape}")
            self.current_frame = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.label_register.imgtk = imgtk
            self.label_register.configure(image=imgtk, text="")
        else:
            print("Failed to read frame")
        
        self.root.after(30, self.update_register_frame)
            
    def stop_register_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.btn_start_register.config(state=tk.NORMAL)
        self.btn_stop_register.config(state=tk.DISABLED)
        self.btn_capture.config(state=tk.DISABLED)
        
    def capture_register(self):
        name = self.entry_name.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name!")
            return
            
        if self.current_frame is None:
            messagebox.showerror("Error", "No frame captured!")
            return
            
        results = self.engine.extract_embeddings(self.current_frame)
        if not results:
            messagebox.showerror("Error", "No face detected! Try again.")
            return
            
        _, embedding, score = results[0]
        face_id = self.db.add_face(name, embedding)
        self.db.save()
        messagebox.showinfo("Success", f"Registered '{name}' (ID: {face_id}, score: {score:.3f})")
        self.entry_name.delete(0, tk.END)
        self.update_person_list()
        self.label_db_count.config(text=f"Total Faces: {self.db.total_faces()}")
        
    def register_from_file(self):
        name = self.entry_name.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name!")
            return
            
        filepath = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not filepath:
            return
            
        img = cv2.imread(filepath)
        if img is None:
            messagebox.showerror("Error", "Could not load image!")
            return
            
        results = self.engine.extract_embeddings(img)
        if not results:
            messagebox.showerror("Error", "No face detected in the image!")
            return
            
        _, embedding, score = results[0]
        face_id = self.db.add_face(name, embedding)
        self.db.save()
        messagebox.showinfo("Success", f"Registered '{name}' (ID: {face_id}, score: {score:.3f})")
        self.entry_name.delete(0, tk.END)
        self.update_person_list()
        self.label_db_count.config(text=f"Total Faces: {self.db.total_faces()}")
        
    def start_recognition(self):
        if self.db.total_faces() == 0:
            messagebox.showwarning("Warning", "Database is empty! Register faces first.")
            return
        
        try:
            self.cap = cv2.VideoCapture(self.selected_camera_recognize)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                messagebox.showerror("Error", "Cannot read from camera!")
                self.cap.release()
                return
            
            self.frame_count = 0
            self.last_results = []
            self.is_running = True
            self.btn_start_recognize.config(state=tk.DISABLED)
            self.btn_stop_recognize.config(state=tk.NORMAL)
            self.update_recognize_frame()
        except Exception as e:
            import traceback
            print(f"Error starting recognition: {e}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to start: {str(e)}")
    
    def update_recognize_frame(self):
        if not self.is_running or not self.cap or not self.cap.isOpened():
            return
        
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.root.after(30, self.update_recognize_frame)
                return
            
            self.frame_count = getattr(self, 'frame_count', 0) + 1
            
            if self.frame_count % config.FRAME_SKIP == 0:
                # Pre-resize frame for faster processing
                frame_small = cv2.resize(frame, (320, 240))
                detections = self.engine.extract_embeddings(frame_small)
                self.last_results = []
                for bbox, embedding, det_score in detections:
                    # Scale bbox back to original size
                    scale_x = frame.shape[1] / 320
                    scale_y = frame.shape[0] / 240
                    bbox_scaled = np.array([
                        bbox[0] * scale_x, bbox[1] * scale_y,
                        bbox[2] * scale_x, bbox[3] * scale_y
                    ])
                    name, similarity = self.db.recognize(embedding)
                    self.last_results.append((bbox_scaled, name, similarity))
            
            last_results = getattr(self, 'last_results', [])
            
            for bbox, name, similarity in last_results:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{name} ({similarity:.2f})"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Update FPS every 10 frames to reduce CPU
            if self.frame_count % 10 == 0:
                curr_time = time.time()
                if hasattr(self, 'prev_time') and self.prev_time:
                    fps = 10.0 / (curr_time - self.prev_time + 1e-9)
                    self.label_fps.config(text=f"FPS: {fps:.1f}")
                self.prev_time = curr_time
            
            # Resize for display only (not for processing)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.label_recognize.imgtk = imgtk
            self.label_recognize.configure(image=imgtk, text="")
            
        except Exception as e:
            import traceback
            print(f"Error in recognition: {e}")
            traceback.print_exc()
        
        # Slower update rate to reduce CPU
        self.root.after(30, self.update_recognize_frame)
            
    def stop_recognition(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.btn_start_recognize.config(state=tk.NORMAL)
        self.btn_stop_recognize.config(state=tk.DISABLED)
        
    def build_from_lfw(self):
        response = messagebox.askyesno("Confirm", 
            "This will download LFW dataset (~233MB) and may take several minutes. Continue?")
        if not response:
            return
            
        threading.Thread(target=self._build_from_lfw_thread, daemon=True).start()
        
    def _build_from_lfw_thread(self):
        try:
            from sklearn.datasets import fetch_lfw_people
            
            self.root.after(0, lambda: messagebox.showinfo("Info", "Downloading LFW dataset..."))
            
            lfw = fetch_lfw_people(min_faces_per_person=20, resize=1.0, data_home=config.LFW_DATA_DIR)
            n_images = lfw.images.shape[0]
            
            success = 0
            for i in range(n_images):
                img_gray = lfw.images[i]
                gray_uint8 = (img_gray * 255).astype(np.uint8)
                bgr = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)
                
                results = self.engine.extract_embeddings(bgr)
                if results:
                    _, embedding, score = results[0]
                    person_name = lfw.target_names[lfw.target[i]]
                    self.db.add_face(person_name, embedding)
                    success += 1
                    
            self.db.save()
            self.root.after(0, lambda: messagebox.showinfo("Success", 
                f"Added {success} faces from LFW dataset!"))
            self.root.after(0, self.refresh_database)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed: {str(e)}"))
            
    def build_from_lfw_folder(self):
        import os
        lfw_folder = os.path.join(config.BASE_DIR, "data", "lfw", "lfw_home", "lfw_funneled")
        
        if not os.path.exists(lfw_folder):
            messagebox.showerror("Error", "LFW folder not found! Run 'Build from LFW Dataset' first.")
            return
        
        folders = [f for f in os.listdir(lfw_folder) if os.path.isdir(os.path.join(lfw_folder, f))]
        if not folders:
            messagebox.showerror("Error", "No person folders found in LFW folder!")
            return
            
        response = messagebox.askyesno("Confirm", 
            f"Found {len(folders)} people in LFW folder. Build database from these images?")
        if not response:
            return
            
        threading.Thread(target=self._build_from_lfw_folder_thread, args=(lfw_folder,), daemon=True).start()
        
    def _build_from_lfw_folder_thread(self, lfw_folder):
        import os
        try:
            folders = [f for f in os.listdir(lfw_folder) if os.path.isdir(os.path.join(lfw_folder, f))]
            
            self.root.after(0, lambda: messagebox.showinfo("Info", f"Processing {len(folders)} people..."))
            
            success = 0
            total = 0
            for person_name in folders:
                person_folder = os.path.join(lfw_folder, person_name)
                images = [f for f in os.listdir(person_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
                
                for img_name in images[:5]:
                    total += 1
                    img_path = os.path.join(person_folder, img_name)
                    img = cv2.imread(img_path)
                    
                    if img is None:
                        continue
                    
                    results = self.engine.extract_embeddings(img)
                    if results:
                        _, embedding, score = results[0]
                        self.db.add_face(person_name, embedding)
                        success += 1
                        
            self.db.save()
            self.root.after(0, lambda: messagebox.showinfo("Success", 
                f"Added {success}/{total} faces from LFW folder!"))
            self.root.after(0, self.refresh_database)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed: {str(e)}"))
            
    def clear_database(self):
        response = messagebox.askyesno("Confirm", "Clear all registered faces?")
        if response:
            self.db = FaceDatabase()
            self.db.save()
            messagebox.showinfo("Success", "Database cleared!")
            self.refresh_database()
            
    def refresh_database(self):
        self.db.load()
        self.label_db_count.config(text=f"Total Faces: {self.db.total_faces()}")
        self.update_person_list()
        
    def update_person_list(self):
        self.listbox.delete(0, tk.END)
        names = set(self.db.id_to_name.values())
        for name in sorted(names):
            count = list(self.db.id_to_name.values()).count(name)
            self.listbox.insert(tk.END, f"{name} ({count} samples)")
            
    def test_all_cameras(self):
        from camera_utils import test_camera_connection
        results = []
        for cam_id in self.available_cameras:
            success, msg = test_camera_connection(cam_id)
            results.append(f"Camera {cam_id}: {msg}")
        
        if results:
            messagebox.showinfo("Camera Test", "\n".join(results))
        else:
            messagebox.showwarning("Camera Test", "No cameras found!")
    
    def on_closing(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
