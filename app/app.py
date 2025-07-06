from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import joblib
import numpy as np
from werkzeug.utils import secure_filename

# ------------------------
# Initialize Flask app
# ------------------------
app = Flask(__name__)
CORS(app)  # Allow CORS for frontend access (like React)

# Create temp folder if not exist
os.makedirs("temp", exist_ok=True)

# Load the trained model
model = joblib.load("../model/svm_model.pkl")

# Initialize HOG descriptor
hog = cv2.HOGDescriptor()
winSize = (64, 128)

# ------------------------
# Prediction Route
# ------------------------
@app.route("/predict", methods=["POST"])
def predict():
    print("Received request...")

    if 'file' not in request.files:
        print("No file in request")
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']

    if file.filename == '':
        print("Empty filename")
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join("temp", filename)
    file.save(filepath)
    print(f"Saved file to {filepath}")

    img = cv2.imread(filepath)
    if img is None:
        print("Failed to read image")
        return jsonify({"error": "Cannot read image"}), 400

    try:
        img = cv2.resize(img, (64, 128))
        features = hog.compute(img).reshape(1, -1)
        prediction = model.predict(features)
        label = "Human" if prediction[0] == 1 else "Not Human"
        print("Prediction:", label)
    except Exception as e:
        print("Exception occurred:", str(e))
        return jsonify({"error": "Prediction failed"}), 500

    os.remove(filepath)
    return jsonify({"prediction": label})
