import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np

class Visualizer:
    def __init__(self, reset_callback=None):
        self.reset_callback = reset_callback

        # Création de la fenêtre tkinter
        self.root = tk.Tk()
        self.root.title("MiniSLAM")

        # Vue caméra
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        # Coordonnées X, Y, Z
        self.position_label = ttk.Label(self.root, text="Position: x=0.00, y=0.00, z=0.00", font=("Arial", 14))
        self.position_label.pack(pady=10)

        # Bouton Reset
        self.reset_button = ttk.Button(self.root, text="Reset Camera", command=self.reset_pose)
        self.reset_button.pack(pady=5)

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

            # Affichage position
            x, y, z = self.position
            self.position_label.config(text=f"Position: x={x:.2f}, y={y:.2f}, z={z:.2f}")

        self.root.after(10, self.update_tk)

    def run(self):
        self.root.mainloop()

    def close(self):
        self.root.quit()

    def on_close(self):
        self.close()
