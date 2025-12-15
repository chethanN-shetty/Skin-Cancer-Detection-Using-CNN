from tensorflow import keras

# 1️⃣ Load your old HDF5 model
model = keras.models.load_model("skin_cancer_cnn_model.h5")

# 2️⃣ Save it in the new Keras format
model.save("skin_cancerCNN_model.keras")

print("✅ Conversion complete! Saved as skin_cancer_model_converted.keras")
