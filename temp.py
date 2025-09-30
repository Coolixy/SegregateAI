import tensorflow as tf

# Load the .h5 model
model = tf.keras.models.load_model("waste_classifier_model.h5", safe_mode=False)

# Save it in SavedModel format
model.save("waste_classifier_model")  # This creates a folder
