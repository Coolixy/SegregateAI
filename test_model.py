import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model
model = tf.keras.models.load_model("waste_classifier_model.keras")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Constants from training
MAX_LEN = 20
CLASS_NAMES = ["Biodegradable", "Non-Biodegradable", "Other"]

# Preprocessing function
def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded

# Test input
description = "plastic wrapper"
processed = preprocess_text(description)
prediction = model.predict(processed)

# Output result
predicted_class = CLASS_NAMES[int(np.argmax(prediction[0]))]
confidence = float(np.max(prediction[0]))

print(f"Input: {description}")
print(f"Predicted Category: {predicted_class}")
print(f"Confidence: {confidence:.2f}")