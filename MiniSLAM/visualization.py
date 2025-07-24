import tkinter as tk
from tkinter import ttk
import cv2
import time
from PIL import Image, ImageTk
import numpy as np

class Visualizer:
    def __init__(self, reset_callback=None):
        self.reset_callback = reset_callback

        self.matches_img = None
        self.ransac_img = None

        # Création de la fenêtre tkinter
        self.root = tk.Tk()
        self.root.title("MiniSLAM")
        self.root.geometry("800x600")  # taille initiale
        self.root.minsize(640, 480)    # taille minimale

        # Vue caméra
        self.image_label = tk.Label(self.root)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Coordonnées X, Y, Z
        self.position_label = ttk.Label(self.root, text="Position: x=0.00, y=0.00, z=0.00", font=("Arial", 14))
        self.position_label.pack(pady=10)

        # FPS label
        self.fps_label = ttk.Label(self.root, text="FPS: 0.00", font=("Arial", 12))
        self.fps_label.pack(pady=5)
        self.last_time = None
        self.current_fps = 0.0

        # Bouton Reset
        self.reset_button = ttk.Button(self.root, text="Reset Camera", command=self.reset_pose)
        self.reset_button.pack(pady=5)

        # Sliders de paramètres
        self.ratio = tk.DoubleVar(value=0.8)
        self.nfeatures = tk.IntVar(value=2000)
        self.checks = tk.IntVar(value=50)

        ttk.Label(self.root, text="Ratio (0.5 - 1.0)").pack()
        ttk.Scale(self.root, from_=0.5, to=1.0, resolution=0.01, orient='horizontal', variable=self.ratio, command=lambda e: self._on_slider_change()).pack()

        ttk.Label(self.root, text="ORB nfeatures").pack()
        ttk.Scale(self.root, from_=500, to=3000, resolution=100, orient='horizontal', variable=self.nfeatures, command=lambda e: self._on_slider_change()).pack()

        ttk.Label(self.root, text="FLANN checks").pack()
        ttk.Scale(self.root, from_=10, to=100, resolution=10, orient='horizontal', variable=self.checks, command=lambda e: self._on_slider_change()).pack()

        # Gestion asynchrone
        self.frame = None
        self.trajectory = []
        self.position = np.array([0.0, 0.0, 0.0])

        # Démarrer loop
        self.root.after(1, self.update_tk)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def reset_pose(self):
        if self.reset_callback:
            self.reset_callback()

    def update_view(self, frame, trajectory, position):
        self.frame = frame
        self.trajectory = trajectory
        self.position = position

    def update_tk(self):
        if self.frame is not None:
            # Conversion BGR -> RGB -> PhotoImage
            img_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            self.image_label.imgtk = img_tk
            self.image_label.configure(image=img_tk)

            x, y, z = self.position
            self.position_label.config(text=f"Position: x={x:.2f}, y={y:.2f}, z={z:.2f}")

            # FPS calculation
            now = time.time()
            if self.last_time is not None:
                self.current_fps = 1.0 / (now - self.last_time)
            self.last_time = now
            self.fps_label.config(text=f"FPS: {self.current_fps:.2f}")

            # Forcer le rafraîchissement de l'UI
            self.root.update_idletasks()

        self.root.after(10, self.update_tk)

    def run(self):
        self.root.mainloop()

    def close(self):
        self.root.quit()

    def on_close(self):
        self.close()
