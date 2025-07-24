import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Visualizer:
    def __init__(self, reset_callback):
        self.root = tk.Tk()
        self.root.title("Mini Visual Odometry UI")

        # Position label
        self.position_var = tk.StringVar()
        self.position_label = ttk.Label(self.root, textvariable=self.position_var, font=("Helvetica", 14))
        self.position_label.pack(pady=10)

        # Reset button
        self.reset_button = ttk.Button(self.root, text="Reset Camera Pose", command=reset_callback)
        self.reset_button.pack(pady=5)

        # Matplotlib 3D plot for trajectory
        self.fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title("Camera Trajectory")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

        # Video window (OpenCV in another thread)
        self.frame = None
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

    def _video_loop(self):
        while True:
            if self.frame is not None:
                cv2.imshow("Live Camera View", self.frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        cv2.destroyAllWindows()

    def update_view(self, frame, trajectory, position):
        # Update video frame
        self.frame = frame

        # Update position text
        x, y, z = position
        self.position_var.set(f"Position → X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}")

        # Update trajectory plot
        self.ax.cla()
        self.ax.set_title("Camera Trajectory")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        if len(trajectory) > 0:
            traj = np.array(trajectory)
            self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-')

        self.canvas.draw()

    def start(self):
        self.root.mainloop()
    