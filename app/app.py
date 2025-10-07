import sys
import os
import cv2
import numpy as np
import io
from flask import Flask, request, render_template, jsonify
import tensorflow as tf

# Add the parent directory (the project root) to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import our detection functions (Laplacian for blur, CNN for authenticity)
from utils.blur_detector import is_blurry 
from utils.authenticity_classifier import load_authenticity_model, predict_authenticity

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define paths to your trained models
AUTHENTICITY_MODEL_PATH = os.path.join(parent_dir, 'models', 'authenticity_model.h5')

# Load the trained model into memory when the app starts
try:
    authenticity_model = load_authenticity_model(AUTHENTICITY_MODEL_PATH)
except Exception as e:
    authenticity_model = None
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        img_stream = io.BytesIO(file.read())
        img_array = np.frombuffer(img_stream.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        resized_img = cv2.resize(img, (256, 256))
        
        # --- BLUR DETECTION ---
        try:
            # Assumes blur_detector.py has the simple Laplacian function
            blurry_status, blur_score = is_blurry(resized_img)
        except Exception as e:
            # Fallback if there is an unexpected error in detection
            print(f"Blur Detection Error: {e}")
            blur_score = 0
            blurry_status = True 
        
        # --- AUTHENTICITY DETECTION ---
        authenticity_status = "Model not loaded"
        if authenticity_model:
            authenticity_status = predict_authenticity(resized_img, authenticity_model)
        
        # --- PREPARE RESULT (CRITICAL FIX) ---
        result = {
            "is_blurry": str(blurry_status),
            "blur_score": float(round(blur_score, 2)), # CONVERSION FIX: This prevents the crash on Render
            "clarity_status": "Blurry" if blurry_status else "Sharp",
            "authenticity_status": authenticity_status
        }
        
        return jsonify(result)

if __name__ == '__main__':
    # We set debug=False for proper Render deployment behavior
    app.run(debug=False)