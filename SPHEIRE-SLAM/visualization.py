import tkinter as tk
from tkinter import ttk
import cv2
import time
from PIL import Image, ImageTk
import numpy as np

class Visualizer:
    def __init__(self, reset_callback=None, shared_state=None):
        self.reset_callback = reset_callback
        self.shared_state = shared_state

        self.frame = None
        self.trajectory = []
        self.position = np.array([0.0, 0.0, 0.0])
        self.matches_img = None
        self.ransac_img = None
        self.matches_window = None

        self.root = tk.Tk()
        self.root.title("VisualOdometry_v1")
        self.root.geometry("800x800")
        self.root.minsize(640, 480)

        # Vue caméra
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=5)

        # Position
        self.position_label = ttk.Label(self.root, text="Position: x=0.00, y=0.00, z=0.00", font=("Arial", 14))
        self.position_label.pack(pady=5)

        # FPS
        self.fps_label = ttk.Label(self.root, text="FPS: 0.00", font=("Arial", 12))
        self.fps_label.pack(pady=5)
        self.last_time = None
        self.current_fps = 0.0

        # Sliders
        self.ratio = tk.DoubleVar(value=0.8)
        self.nfeatures = tk.IntVar(value=2000)
        self.checks = tk.IntVar(value=50)

        ttk.Label(self.root, text="Ratio (0.5 - 1.0)").pack()
        tk.Scale(self.root, from_=0.5, to=1.0, resolution=0.01, orient='horizontal',
                 variable=self.ratio, command=lambda e: self._on_slider_change()).pack()

        ttk.Label(self.root, text="ORB nfeatures").pack()
        tk.Scale(self.root, from_=500, to=3000, resolution=100, orient='horizontal',
                 variable=self.nfeatures, command=lambda e: self._on_slider_change()).pack()

        ttk.Label(self.root, text="FLANN checks").pack()
        tk.Scale(self.root, from_=10, to=100, resolution=10, orient='horizontal',
                 variable=self.checks, command=lambda e: self._on_slider_change()).pack()

        # Boutons
        self.reset_button = ttk.Button(self.root, text="Reset Camera", command=self.reset_pose)
        self.reset_button.pack(pady=5)

        self.match_button = ttk.Button(self.root, text="Afficher Matches", command=self.open_matches_window)
        self.match_button.pack(pady=5)

        self.root.after(1, self.update_tk)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.map_window = None
        self.show_map_button = ttk.Button(self.root, text="Afficher Carte 3D", command=self.open_map_window)
        self.show_map_button.pack(pady=5)

    def reset_pose(self):
        if self.reset_callback:
            self.reset_callback()

    def set_param_callback(self, callback):
        self.param_callback = callback

    def _on_slider_change(self):
        if hasattr(self, "param_callback"):
            self.param_callback(self.ratio.get(), self.nfeatures.get(), self.checks.get())

    def update_view(self, frame, trajectory, position, matches_img=None, ransac_img=None):
        self.frame = frame
        self.trajectory = trajectory
        self.position = position
        self.matches_img = matches_img
        self.ransac_img = ransac_img

    def update_tk(self):
        if self.shared_state:
            self.frame = self.shared_state.get("frame", None)
            self.trajectory = self.shared_state.get("trajectory", [])
            self.position = self.shared_state.get("position", np.array([0.0, 0.0, 0.0]))
            self.matches_img = self.shared_state.get("matches_img", None)
            self.ransac_img = self.shared_state.get("ransac_img", None)

        if self.frame is not None:
            img_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.image_label.imgtk = img_tk
            self.image_label.configure(image=img_tk)

            x, y, z = self.position
            self.position_label.config(text=f"Position: x={x:.2f}, y={y:.2f}, z={z:.2f}")

            now = time.time()
            if self.last_time is not None:
                self.current_fps = 1.0 / (now - self.last_time)
            self.last_time = now
            self.fps_label.config(text=f"FPS: {self.current_fps:.2f}")

        self.update_matches_window()
        self.update_map_window()
        self.root.after(15, self.update_tk)

    def open_matches_window(self):
        if self.matches_window is None or not tk.Toplevel.winfo_exists(self.matches_window):
            self.matches_window = tk.Toplevel(self.root)
            self.matches_window.title("Matches & RANSAC")
            self.matches_window.geometry("1200x600")

            self.matches_label = tk.Label(self.matches_window)
            self.matches_label.pack(side=tk.LEFT, padx=5, pady=5)

            self.ransac_label = tk.Label(self.matches_window)
            self.ransac_label.pack(side=tk.RIGHT, padx=5, pady=5)

    def open_map_window(self):
        if self.map_window is None or not tk.Toplevel.winfo_exists(self.map_window):
            self.map_window = tk.Toplevel(self.root)
            self.map_window.title("Carte 3D (projection XY)")
            self.map_window.geometry("600x600")

            self.map_canvas = tk.Label(self.map_window)
            self.map_canvas.pack(padx=5, pady=5)

    def update_matches_window(self):
        if self.matches_window is not None and tk.Toplevel.winfo_exists(self.matches_window):
            if self.matches_img is not None:
                img_rgb = cv2.cvtColor(self.matches_img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                self.matches_label.imgtk = img_tk
                self.matches_label.configure(image=img_tk)
                self.matches_label.update_idletasks()

            if self.ransac_img is not None:
                img_rgb = cv2.cvtColor(self.ransac_img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                self.ransac_label.imgtk = img_tk
                self.ransac_label.configure(image=img_tk)
                self.ransac_label.update_idletasks()

    def update_map_window(self):
        if self.map_window is not None and tk.Toplevel.winfo_exists(self.map_window):
            map_points = self.shared_state.get("map_points", [])
            pose_graph = self.shared_state.get("pose_graph", {"nodes": [], "edges": []})

            if len(map_points) > 0 or len(pose_graph["edges"]) > 0:
                img = np.zeros((600, 600, 3), dtype=np.uint8)

                for pt in map_points:
                    try:
                        x, y, z = pt
                        u = int(300 + x * 30)
                        v = int(300 - y * 30)
                        if 0 <= u < 600 and 0 <= v < 600:
                            cv2.circle(img, (u, v), 1, (0, 255, 0), -1)
                    except:
                        pass

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                self.map_canvas.imgtk = img_tk
                self.map_canvas.configure(image=img_tk)
        
    
    def draw_map_points(image, map_points):
        for pt in map_points:
            x, y, z = pt
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)
        return image   

    def run(self):
        self.root.mainloop()

    def close(self):
        self.root.quit()

    def on_close(self):
        self.close()
