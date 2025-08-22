import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
import os

# Suppress unnecessary TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load model
model = load_model("rice_classifier_mobilenetv2.h5")

# Class labels — ensure this matches your training order
class_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Open file dialog to select image
root = Tk()
root.withdraw()  # Hide the root window
file_path = filedialog.askopenfilename(title="Select a rice image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

if not file_path:
    print("❌ No file selected.")
else:
    # Load and preprocess image
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    print(f"\n✅ Selected file: {file_path}")
    print(f"🔍 Predicted Rice Type: {predicted_class}")
