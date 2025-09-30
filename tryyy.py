import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, Label
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageTk

# Load your model
model = load_model("waste_classifier.h5")

# BONUS: Print model input shape for debugging
print("✅ Model Input Shape:", model.input_shape)

# Match this order to how your training folders were loaded
class_names = ['BIODEGRADABLE','NON-BIODEGRADABLE','OTHER']

def classify_image(file_path):
    # Adjust image size to match model input
    img = image.load_img(file_path, target_size=(154, 154))  # match model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0  # Normalize

    predictions = model.predict(img_array)

    probabilities = tf.nn.softmax(predictions[0]).numpy()
    pred_class_index = np.argmax(probabilities)
    confidence = probabilities[pred_class_index] * 100
    pred_class = class_names[pred_class_index]

    return pred_class,confidence


def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        pred_class, confidence = classify_image(file_path)
        
        # Display image
        img = Image.open(file_path)
        img = img.resize((200, 200))
        tk_img = ImageTk.PhotoImage(img)
        image_label.config(image=tk_img)
        image_label.image = tk_img
        
        result_label.config(text=f"Prediction: {pred_class}\nConfidence: {confidence:.2f}%")

# GUI Setup
app = tk.Tk()
app.title("Waste Classifier")
app.geometry("400x400")

button = tk.Button(app, text="Pick an Image", command=open_image)
button.pack(pady=10)

image_label = Label(app)
image_label.pack()

result_label = Label(app, text="", font=("Arial", 14))
result_label.pack(pady=20)

app.mainloop()