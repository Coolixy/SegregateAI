# train_model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os

# === CONFIG ===
CSV_PATH = 'data.csv'  # Make sure this exists
MODEL_SAVE_PATH = 'waste_classifier.keras'
MAX_TOKENS = 1000
SEQUENCE_LENGTH = 100
EPOCHS = 50
BATCH_SIZE = 16

# === LOAD DATA ===
df = pd.read_csv(CSV_PATH)
if df.shape[1] < 2:
    raise ValueError("CSV must have at least 2 columns: [description, label]")

df.columns = ['description', 'label']
df.dropna(inplace=True)

# === ENCODE LABELS ===
label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
df['label_int'] = df['label'].map(label_mapping)

# === TRAIN / VAL / TEST SPLIT ===
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label_int'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label_int'], random_state=42)

# === TEXT VECTORIZATION ===
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)
vectorize_layer.adapt(train_df['description'].astype(str))

# === CLASS WEIGHTS ===
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(train_df['label_int']),
                                     y=train_df['label_int'])
class_weight_dict = dict(enumerate(class_weights))

# === MODEL ===
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,), dtype=tf.string),
    vectorize_layer,
    tf.keras.layers.Embedding(input_dim=MAX_TOKENS + 1, output_dim=64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(label_mapping), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === TRAIN ===
history = model.fit(
    train_df['description'].astype(str).values,
    train_df['label_int'].astype(np.int32).values,
    validation_data=(
        val_df['description'].astype(str).values,
        val_df['label_int'].astype(np.int32).values
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    verbose=2
)

# === EVALUATE ===
test_loss, test_acc = model.evaluate(
    test_df['description'].astype(str).values,
    test_df['label_int'].astype(np.int32).values,
    verbose=2
)

# === SAVE MODEL & LABEL MAP ===
model.save(MODEL_SAVE_PATH)
with open("label_mapping.txt", "w") as f:
    for k, v in label_mapping.items():
        f.write(f"{v},{k}\n")
print(f"\nModel saved to {MODEL_SAVE_PATH}")
