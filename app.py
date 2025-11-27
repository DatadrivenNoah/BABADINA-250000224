# app.py
import os
import logging
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import sqlite3
from datetime import datetime
from PIL import Image
import io
import requests
import atexit

# Configure logging for Render
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Initialize Flask app
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Paths using relative paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DB_PATH = os.path.join(BASE_DIR, "database.db")

os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "face_emotionModel.h5")

# -------------------------------
# Global variables
# -------------------------------
model = None
face_cascade = None

# -------------------------------
# Download model if not exists
# -------------------------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1ylpZo85D-t8G1tBa8PwjCi8M1G9XZ_bm"

def download_model():
    """Download model with better error handling"""
    try:
        if not os.path.exists(MODEL_PATH):
            logger.info("Downloading model...")
            r = requests.get(MODEL_URL, timeout=30)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
            logger.info("Model downloaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        return False

# -------------------------------
# Initialize components
# -------------------------------
def init_app():
    global model, face_cascade
    
    # Download model
    if not download_model():
        logger.error("Failed to download model. App will not function properly.")
        return False
    
    # Load model
    try:
        logger.info("Loading TensorFlow model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False
    
    # Initialize face cascade
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        logger.info("Face cascade loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load face cascade: {e}")
    
    # Initialize database
    init_db()
    return True

# -------------------------------
# Initialize database
# -------------------------------
def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      emotion TEXT,
                      description TEXT,
                      image BLOB)''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully!")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

# -------------------------------
# Emotion labels and descriptions
# -------------------------------
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
description_labels = {
    'Angry': "Take a deep breath. Calm is power.",
    'Disgust': "Stay positive. Focus on what inspires you.",
    'Fear': "Be brave. Face challenges head-on.",
    'Happy': "Keep smiling! Share your joy today.",
    'Sad': "It's okay to feel down. Tomorrow is brighter.",
    'Surprise': "Expect the unexpected! Embrace new experiences.",
    'Neutral': "Stay balanced. Enjoy the moment."
}

# -------------------------------
# Initialize on startup
# -------------------------------
if not init_app():
    logger.error("App initialization failed!")

# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'opencv_version': cv2.__version__,
        'tensorflow_version': tf.__version__
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        file = request.files['image']
        if file is None:
            return jsonify({'error': 'No image provided'}), 400
        
        # Process image
        img = Image.open(file.stream).convert('L')  # Grayscale
        img = img.resize((48, 48))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 48, 48, 1)

        # Predict emotion
        prediction = model.predict(img_array, verbose=0)
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]
        description = description_labels[emotion]

        # Save to database
        img_byte_arr = io.BytesIO()
        Image.fromarray((img_array[0] * 255).astype(np.uint8).reshape(48,48)).save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO predictions (timestamp, emotion, description, image) VALUES (?, ?, ?, ?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), emotion, description, img_bytes))
        conn.commit()
        conn.close()

        logger.info(f"Prediction made: {emotion}")
        
        return jsonify({
            'emotion': emotion,
            'description': description,
            'confidence': float(prediction[0][emotion_idx])
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

# -------------------------------
# Run with Gunicorn for Render
# -------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)