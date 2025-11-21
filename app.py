# app.py
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import sqlite3
import os
from datetime import datetime
from PIL import Image
import base64
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('face_emotionModel.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize database
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  emotion TEXT,
                  confidence REAL,
                  image BLOB)''')
    conn.commit()
    conn.close()

init_db()

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        img = Image.open(file.stream).convert('L')  # Grayscale
        img = img.resize((48, 48))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 48, 48, 1)

        prediction = model.predict(img_array)
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]
        confidence = float(prediction[0][emotion_idx])

        # Save to database
        img_byte_arr = io.BytesIO()
        Image.fromarray((img_array[0] * 255).astype(np.uint8).reshape(48,48)).save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("INSERT INTO predictions (timestamp, emotion, confidence, image) VALUES (?, ?, ?, ?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), emotion, confidence, img_bytes))
        conn.commit()
        conn.close()

        return jsonify({
            'emotion': emotion,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
    