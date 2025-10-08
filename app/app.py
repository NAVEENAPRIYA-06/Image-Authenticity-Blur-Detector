import sys
import os
import cv2
import numpy as np
import io
import uuid
import base64
from flask import Flask, request, render_template, jsonify, redirect, url_for

# Ensure project root paths are accessible for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import detection functions
from utils.blur_detector import is_blurry 
from utils.authenticity_classifier import load_authenticity_model, predict_authenticity

# We define the static folder where uploads will go
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'uploads') 

app = Flask(__name__)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define paths to your trained models
AUTHENTICITY_MODEL_PATH = os.path.join(parent_dir, 'models', 'authenticity_model.h5')

# Global variables for models
authenticity_model = None
MAX_CLARITY_SCORE = 1500

# Load the trained model into memory when the app starts
try:
    authenticity_model = load_authenticity_model(AUTHENTICITY_MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        return redirect(url_for('analysis_choice', filename=filename))

@app.route('/choice/<filename>')
def analysis_choice(filename):
    return render_template('choice.html', filename=filename)


@app.route('/analyze/<filename>/<analysis_type>')
def analyze_image(filename, analysis_type):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return redirect(url_for('index')) 

    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(img, (256, 256))
    
    context = {'filename': filename}

    if analysis_type == 'blur':
        # --- BLUR DETECTION & IMAGE PROCESSING ---
        # 1. Grayscale Conversion and Encoding (for visual comparison)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, buffer = cv2.imencode('.png', gray_img)
        gray_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 2. Analysis
        blurry_status, blur_score = is_blurry(resized_img)
        
        clarity_status = "Blurry" if blurry_status else "Sharp"
        clarity_percentage = min(100, (blur_score / MAX_CLARITY_SCORE) * 100)
        
        if clarity_percentage > 75:
            blur_color = '#28a745'
        elif clarity_percentage > 40:
            blur_color = '#ffc107'
        else:
            blur_color = '#dc3545'

        context.update({
            'blur_status': clarity_status,
            'blur_score': float(round(blur_score, 2)),
            'blur_percentage': round(clarity_percentage, 2),
            'blur_color': blur_color,
            'grayscale_image': gray_image_base64 # <-- CRITICAL: Passed to template
        })

    elif analysis_type == 'authenticity':
        # --- AUTHENTICITY DETECTION ---
        authenticity_status = "Model not loaded"
        if authenticity_model:
            authenticity_status = predict_authenticity(resized_img, authenticity_model)
        
        context['authenticity_status'] = authenticity_status
    
    context['analysis_type'] = analysis_type 
    return render_template('result.html', **context)

if __name__ == '__main__':
    app.run(debug=True)