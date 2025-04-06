from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load the trained model (Random Forest + StandardScaler pipeline)
model = joblib.load("model.joblib")  # Make sure this file is in the same folder

# Class labels (these must match your label encoding)
folders = ['Autumn', 'Spring', 'Summer', 'Winter']

# Feature extraction function: color histogram
def extract_features(image, bins=(29, 29, 29)):
    image = image.resize((224, 224)).convert("RGB")
    image = np.array(image)

    hist = []
    for i in range(3):  # For R, G, B channels
        channel_hist = np.histogram(image[:, :, i], bins=bins[0], range=(0, 256))[0]
        hist.extend(channel_hist)

    return np.array(hist).reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))

        print("✅ Image received. Processing...")

        features = extract_features(image)
        print(f"✅ Features extracted: {features.shape}")

        prediction = model.predict(features)[0]
        class_name = folders[prediction]

        print(f"✅ Prediction: {class_name} ({prediction})")

        return jsonify({'prediction': int(prediction), 'label': class_name})

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Allow access from other devices on the network
    app.run(host='0.0.0.0', port=5000)
