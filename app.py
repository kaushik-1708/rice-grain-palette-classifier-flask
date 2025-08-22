from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
model = tf.keras.models.load_model("rice_classifier_mobilenetv2.h5")
class_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            img_path = os.path.join("static", file.filename)
            file.save(img_path)

            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            pred = model.predict(img_array)
            prediction = class_labels[np.argmax(pred)]

    return render_template("index.html", prediction=prediction, img_path=img_path)
if __name__ == "__main__":
    app.run(debug=True)
