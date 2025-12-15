import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ============================
# 1️⃣  CONFIGURATION
# ============================
MODEL_PATH = r"C:\Users\Sudarshan\PycharmProjects\SkinCancerdatasetISIC2020\.venv\skin_cancer_cnn_model.h5"  # your saved model path
IMAGE_PATH = r"C:\Users\Sudarshan\Desktop\SkinCancerDetectionDataset\archive (1)\ISIC_2019_Training_Input\ISIC_2019_Training_Input\ISIC_0000075.jpg"  # test image
IMG_SIZE = (224, 224)  # same as training

# ============================
# 2️⃣  LOAD MODEL
# ============================
model = keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ============================
# 3️⃣  CLASS LABELS
# ============================
# Update with your dataset's class folders
CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'UNK', 'VASC']

# ============================
# 4️⃣  PREPROCESS TEST IMAGE
# ============================
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # make batch of 1
    img_array = img_array / 255.0  # rescale same as training
    return img_array

# ============================
# 5️⃣  PREDICT
# ============================
img_array = preprocess_image(IMAGE_PATH)
predictions = model.predict(img_array)

# Get predicted class
predicted_index = np.argmax(predictions[0])
predicted_label = CLASS_NAMES[predicted_index]
confidence = np.max(predictions[0]) * 100

print(f"🔍 Predicted Class: {predicted_label} ({confidence:.2f}% confidence)")
# ============================
# 6️⃣  DISPLAY IMAGE + RESULT
# ============================
img_display = Image.open(IMAGE_PATH)

plt.figure(figsize=(5, 5))
plt.imshow(img_display)
plt.axis('off')
plt.title(f"Prediction: {predicted_label}\nConfidence: {confidence:.2f}%", fontsize=12, color='blue')
plt.show()