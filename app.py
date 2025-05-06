from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import tensorflow as tf

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Load Keras model
model = tf.keras.models.load_model("skin_undertone_classifier_final.h5")

# Class labels (same order as training folders)
labels = ['Autumn', 'Spring', 'Summer', 'Winter']

# Preprocessing (must match your training)
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        print("Image received")

        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0]
        predicted_class = int(np.argmax(prediction))
        predicted_label = labels[predicted_class]

        print(f"Prediction: {predicted_label} ({predicted_class})")

        return jsonify({'prediction': predicted_class, 'label': predicted_label})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
