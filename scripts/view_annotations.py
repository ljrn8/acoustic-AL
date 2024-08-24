import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw

from sklearn.preprocessing import MinMaxScaler


def update_display(index):
    """Update the spectrogram and annotation display based on the selected index."""
    if 0 <= index < len(spectrograms):
        # Display the selected spectrogram
        spectrogram, Y = spectrograms[index]
        scaler = MinMaxScaler((1, 255))
        scaler.fit(spectrogram)
        spectrogram = scaler.transform(spectrogram)
        spectrogram = (spectrogram * 255).astype(np.uint8)  # Scale to [0, 255] and convert to 8-bit

        img = Image.fromarray(spectrogram)

        for i, y_col in enumerate(Y):
            if y_col.any():
                draw_red_dot(img, (i, 100), 5)
                print(i)

        img.thumbnail((400, 400))  # Resize image for display
        img_tk = ImageTk.PhotoImage(img)
        spectrogram_label.config(image=img_tk)
        spectrogram_label.image = img_tk
        
        # Display the selected annotation
        # annotation_text = annotations[index] if index < len(annotations) else "No annotation"
        # annotation_label.config(text=annotation_text)
        
def draw_red_dot(image, position, radius=5):
    """
    Draw a red dot on the given image at the specified position.

    :param image: The PIL Image object on which to draw the dot.
    :param position: A tuple (x, y) representing the position of the dot.
    :param radius: The radius of the dot.
    :return: The PIL Image object with the red dot drawn on it.
    """
    draw = ImageDraw.Draw(image)
    x, y = position

    # Draw a red dot
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius), 
        fill='red', 
        outline='red'
    )

    return image

def on_select(event):
    """Handle listbox selection event."""
    selection = listbox.curselection()
    if selection:
        update_display(selection[0])

from config import *
import h5py
import numpy as np


spectrograms = []
f = h5py.File(INTERMEDIATE / 'train.hdf5', 'r')

S = np.array(f[ANNOTATED_RECORDING]['X'])
Y = np.array(f[ANNOTATED_RECORDING]['Y'])


spectrograms += [
        (S[:, start:min(start+500, S.shape[1]-1)], 
        Y[:, start:min(start+500, S.shape[1]-1)] )
        for start in range(1, S.shape[1]-1, 500)
]

f.close()


import pickle
with open("./objects/spec_test.pkl", 'wb') as f:
    pickle.dump(spectrograms, f)
exit()

# Create the main window
root = tk.Tk()
root.title("Spectrogram Viewer")

# Create and pack the widgets
listbox = tk.Listbox(root)
listbox.pack(side=tk.LEFT, fill=tk.Y)

spectrogram_label = tk.Label(root)
spectrogram_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

annotation_label = tk.Label(root, text="Select a spectrogram to see annotations.")
annotation_label.pack(side=tk.BOTTOM, fill=tk.X)

# Populate the listbox with spectrogram names
for i in range(len(spectrograms)):
    listbox.insert(tk.END, f"Spectrogram {i+1}")

listbox.bind('<<ListboxSelect>>', on_select)

# Start the Tkinter event loop
root.mainloop()
