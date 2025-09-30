import tensorflow as tf
import os

# --- CONFIG ---
MODEL_PATH = "waste_classifier_model.keras"  # or "waste_classifier_model.h5"
TFLITE_PATH = "waste_classifier.tflite"

# --- LOAD MODEL ---
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# --- CONVERT TO TFLITE ---
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# --- SAVE TFLITE FILE ---
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved as: {TFLITE_PATH}")