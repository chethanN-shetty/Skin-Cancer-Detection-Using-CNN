import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
from tensorflow import keras

# ========== CONFIG ==========
# CHANGE this path if your model is somewhere else
MODEL_PATH = r"C:\Users\Sudarshan\PycharmProjects\SkinCancerdatasetISIC2020\.venv\skin_cancerCNN_model.keras"

# Optional: path to a test image you mentioned earlier
TEST_IMAGE_PATH = r"C:\Users\Sudarshan\Downloads\Psoriasis-dermoscopy.jpg"

IMG_SIZE = (224, 224)  # adjust if model expects different
CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'UNK', 'VASC']

# ========== APP INIT ==========
app = Flask(__name__, static_folder="static", static_url_path="/")
CORS(app)  # allow cross-origin for dev

# ========== LOAD MODEL ==========
print("Loading model from:", MODEL_PATH)
model = keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

def preprocess_pil_image(pil_img):
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(pil_img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/sample-image")
def sample_image():
    if not os.path.exists(TEST_IMAGE_PATH):
        return jsonify({"error": f"Test image not found at {TEST_IMAGE_PATH}"}), 404
    return send_file(TEST_IMAGE_PATH, mimetype="image/png")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' in request"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        pil_img = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": "Could not open image: " + str(e)}), 400

    try:
        x = preprocess_pil_image(pil_img)
        preds = model.predict(x)
        preds = preds.flatten()
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100.0
        label = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else "Unknown"
        return jsonify({"prediction": label, "confidence": round(confidence, 2)})
    except Exception as e:
        return jsonify({"error": "Prediction error: " + str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
