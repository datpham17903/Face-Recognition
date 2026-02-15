import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from face_engine import FaceEngine
from face_database import FaceDatabase
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
        
        self.label_register = tk.Label(self.tab_register, bg="black", width=640, height=480)
        self.label_register.pack(pady=10)
        
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
        
    def setup_recognize_tab(self):
        self.label_recognize = tk.Label(self.tab_recognize, bg="black", width=640, height=480)
        self.label_recognize.pack(pady=10)
        
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
        self.cap = cv2.VideoCapture(config.WEBCAM_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)
        self.is_running = True
        self.btn_start_register.config(state=tk.DISABLED)
        self.btn_stop_register.config(state=tk.NORMAL)
        self.btn_capture.config(state=tk.NORMAL)
        threading.Thread(target=self.update_register_frame, daemon=True).start()
        
    def update_register_frame(self):
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480))
                imgtk = ImageTk.PhotoImage(image=img)
                self.label_register.imgtk = imgtk
                self.label_register.configure(image=imgtk)
            time.sleep(0.03)
            
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
            
        self.cap = cv2.VideoCapture(config.WEBCAM_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)
        self.is_running = True
        self.btn_start_recognize.config(state=tk.DISABLED)
        self.btn_stop_recognize.config(state=tk.NORMAL)
        threading.Thread(target=self.update_recognize_frame, daemon=True).start()
        
    def update_recognize_frame(self):
        frame_count = 0
        last_results = []
        prev_time = time.time()
        
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            if frame_count % config.FRAME_SKIP == 0:
                detections = self.engine.extract_embeddings(frame)
                last_results = []
                for bbox, embedding, det_score in detections:
                    name, similarity = self.db.recognize(embedding)
                    last_results.append((bbox, name, similarity))
                    
            for bbox, name, similarity in last_results:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{name} ({similarity:.2f})"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                           
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-9)
            prev_time = curr_time
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.label_recognize.imgtk = imgtk
            self.label_recognize.configure(image=imgtk)
            self.label_fps.config(text=f"FPS: {fps:.1f}")
            
            time.sleep(0.01)
            
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
