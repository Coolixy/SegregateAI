from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

# Allow requests from any origin (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Image Classification Model ---
image_model = tf.keras.models.load_model("waste_classifier_model.h5")
image_class_names = ["Biodegradable", "Non-Biodegradable", "Other"]

# --- Load Text Classification Model ---
text_model = tf.keras.models.load_model("waste_classifier_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 20
text_class_names = ["Biodegradable", "Non-Biodegradable", "Other"]

# --- Image Classification Endpoint ---
@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB").resize((154, 154))
        img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

        prediction = image_model.predict(img_array)
        predicted_class = image_class_names[np.argmax(prediction)]

        return {"category": predicted_class}
    except Exception as e:
        return {"error": str(e)}

# --- Text Classification Endpoint ---
@app.post("/classify-text/")
async def classify_text(payload: dict):
    description = payload.get("description", "")
    try:
        sequence = tokenizer.texts_to_sequences([description])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
        prediction = text_model.predict(padded)
        predicted_class = text_class_names[int(np.argmax(prediction[0]))]
        confidence = float(np.max(prediction[0]))
        return {
            "category": predicted_class,
            "confidence": round(confidence, 3)
        }
    except Exception as e:
        return {"error": str(e)}