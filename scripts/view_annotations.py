import tkinter as tk
from PIL import Image, ImageTk, ImageDraw


import pickle
from config import *
import numpy as np
import librosa

def update_display(index):
    if 0 <= index < len(spectrograms):
        S, Y = spectrograms[index]
        S = S.astype(float)

        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        img = Image.fromarray(S_db.astype(np.int8), mode='L')

        for i, col in enumerate(Y):
            diff = np.diff(col)
            label_starts = np.where(diff == 1)[0]
            for start in label_starts:
                # draw_annotation(img, start, 250) # just do 250hz for now
                draw = ImageDraw.Draw(img)   
                draw.line([(start, 0), (start, S.shape[0])], fill="red", width = 1) 

        img = img.transpose(Image.FLIP_TOP_BOTTOM)  # Flip the image vertically
        # img.thumbnail((400, 400))  
        img_tk = ImageTk.PhotoImage(img)
        spectrogram_label.config(image=img_tk)
        spectrogram_label.image = img_tk
       

def draw_annotation(img, x, y, w=100, l=50):
    draw = ImageDraw.Draw(img)
    draw.rectangle(((x, y), (x+w, y+l)), outline='red', width=2)
    return img


def on_select(event):
    selection = listbox.curselection()
    if selection:
        update_display(selection[0])


# with open("./objects/spec_test.pkl", 'rb') as f:
#    spectrograms = pickle.load(f)


# load annotated sample from train set
import h5py
with h5py.File(INTERMEDIATE / 'train.h5py', 'r') as f:
    rec = list(f)[2]
    s = np.array(f[rec]['X'])
    Y = np.array(f[rec]['Y'])

print("check ->> ", s.shape, Y.shape)

sr = 16_000
chunk = sr * 10
chunks = [
        (s[:, start, start + chunk], Y[:, start, start + chunk]) 
        for start in range(0, sr * 500)
        ]

root = tk.Tk()
root.title("Spectrogram Viewer")

listbox = tk.Listbox(root)
listbox.pack(side=tk.LEFT, fill=tk.Y)

spectrogram_label = tk.Label(root)
spectrogram_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

annotation_label = tk.Label(root, text="Select a spectrogram to see annotations.")
annotation_label.pack(side=tk.BOTTOM, fill=tk.X)

for i in range(len(chunks)):
    listbox.insert(tk.END, f"chunk {i+1}")

listbox.bind('<<ListboxSelect>>', on_select)

root.mainloop()
