import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw

from sklearn.preprocessing import MinMaxScaler

import pickle
from config import *
import h5py
import numpy as np


def update_display(index):
    if 0 <= index < len(spectrograms):

        spectrogram, Y = spectrograms[index]

        scaler = MinMaxScaler((1, 255))
        scaler.fit(spectrogram)
        spectrogram = scaler.transform(spectrogram)
        spectrogram = (spectrogram * 255).astype(np.int)

        img = Image.fromarray(spectrogram)

        for i, y_col in enumerate(Y):
            if y_col.any():
                draw_red_dot(img, (i, 100), 5)
                print(i)

        img.thumbnail((400, 400))  
        img_tk = ImageTk.PhotoImage(img)
        spectrogram_label.config(image=img_tk)
        spectrogram_label.image = img_tk
        
        # Display the selected annotation
        # annotation_text = annotations[index] if index < len(annotations) else "No annotation"
        # annotation_label.config(text=annotation_text)
       

def draw_red_dot(image, position, radius=5):
    draw = ImageDraw.Draw(image)
    x, y = position
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius), 
        fill='red', 
        outline='red'
    )

    return image

def on_select(event):
    selection = listbox.curselection()
    if selection:
        update_display(selection[0])


with open("./objects/spec_test.pkl", 'rb') as f:
    spectrograms = pickle.load(f)



root = tk.Tk()
root.title("Spectrogram Viewer")

listbox = tk.Listbox(root)
listbox.pack(side=tk.LEFT, fill=tk.Y)

spectrogram_label = tk.Label(root)
spectrogram_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

annotation_label = tk.Label(root, text="Select a spectrogram to see annotations.")
annotation_label.pack(side=tk.BOTTOM, fill=tk.X)

for i in range(len(spectrograms)):
    listbox.insert(tk.END, f"Spectrogram {i+1}")

listbox.bind('<<ListboxSelect>>', on_select)

root.mainloop()
