import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

# 1) Paths
base_dir = r'C:\Users\Saatv\Desktop\proo\DATASET MODEL'
train_dir = os.path.join(base_dir, 'TRAIN')
test_dir  = os.path.join(base_dir, 'TEST')

# 2) Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 3) Create iterators
batch_size = 32
img_size = (150, 150)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# 4) Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(*img_size, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')  # 3 classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 5) Train
epochs = 10
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# 💾 Save the trained model
model.save('waste_classifier.h5')
print("Model saved successfully as 'waste_classifier.h5'")

# 6) Evaluate on TEST
test_loss, test_acc = model.evaluate(test_generator)
print(f'\nTest accuracy: {test_acc:.2%}')

# 7) Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, epochs+1)

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Train Acc')
plt.plot(epochs_range, val_acc, label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.title('Loss')
plt.legend()

plt.show()

# 8) Predict a single image
# ----------------------------------------
img_path = r'C:\Users\Saatv\Desktop\proo\tryy.jpg'  # 🔁 Replace with your image path

# Load and preprocess
img = load_img(img_path, target_size=img_size)
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)[0]  # softmax output
predicted_index = np.argmax(prediction)
confidence = prediction[predicted_index]

# Map prediction to label
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}
predicted_class = class_labels[predicted_index]

# Threshold for "not identified"
threshold = 0.60

if confidence >= threshold:
    print(f"\nPredicted class: {predicted_class} ({confidence:.2%} confidence)")
else:
    print(f"\n❌ Not identified (confidence too low: {confidence:.2%})")
# ----------------------------------------