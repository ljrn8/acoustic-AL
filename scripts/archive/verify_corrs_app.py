##
#   Messy false positive checker for correlations (not for use)
#

import os
import sys
import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk
from pathlib import Path

# where to write the false positive indexes
FP_FILE = Path("..") / "scripts" / "false_positives" / sys.argv[1]


class App:

    def __init__(self, root):
        self.root = root
        self.root.title("Image Slideshow")

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.action_button = tk.Button(
            self.root, text="Mark FP", command=self.on_button_click
        )
        self.action_button.pack()

        self.prev_button = tk.Button(
            self.root, text="Previous", command=self.show_prev_image
        )
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(
            self.root, text="Next", command=self.show_next_image
        )
        self.next_button.pack(side=tk.RIGHT)

        self.open_button = tk.Button(
            self.root, text="!! Open Image Folder", command=self.open_folder
        )
        self.open_button.pack()

        self.image_files = []
        self.current_image_index = 0

        self.root.bind("<Right>", self.show_next_image_event)

    def show_next_image_event(self, event):
        self.show_next_image()

    def open_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(("png", "jpg", "jpeg", "gif", "bmp"))
            ]
            if self.image_files:
                self.current_image_index = 0
                self.show_image()

    def show_image(self):
        if self.image_files:
            image_path = self.image_files[self.current_image_index]
            image = Image.open(image_path)
            image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

    def show_prev_image(self):
        if self.image_files:
            self.current_image_index = (self.current_image_index - 1) % len(
                self.image_files
            )
            self.show_image()

    def show_next_image(self):
        if self.image_files:
            self.current_image_index = (self.current_image_index + 1) % len(
                self.image_files
            )
            self.show_image()

    def on_button_click(self):
        if self.image_files:
            print(f"Button clicked for image: {self.current_image_index}")
            with open(FP_FILE, "a") as f:
                f.write(str(self.current_image_index + 1) + "\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
