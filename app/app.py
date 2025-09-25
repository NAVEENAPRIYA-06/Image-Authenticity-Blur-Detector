import sys
import os
import cv2
import numpy as np
import io
from flask import Flask, request, render_template, jsonify, send_from_directory
import tensorflow as tf

# Add the parent directory (the project root) to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import our detection functions
from utils.blur_detector import is_blurry
from utils.authenticity_classifier import load_authenticity_model, predict_authenticity

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the path to your trained model and examples folder
MODEL_PATH = os.path.join(parent_dir, 'models', 'authenticity_model.h5')
# This is the corrected line to define the examples folder
EXAMPLES_FOLDER = os.path.join(app.root_path, '..', 'examples')
# Print the path to confirm it is correct
print(f"Flask is serving examples from: {EXAMPLES_FOLDER}")

# Load the trained model into memory when the app starts
try:
    authenticity_model = load_authenticity_model(MODEL_PATH)
except Exception as e:
    authenticity_model = None
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

# New route to serve example images
@app.route('/examples/<filename>')
def serve_example_image(filename):
    return send_from_directory(EXAMPLES_FOLDER, filename)

# Route for analyzing images from the examples folder
@app.route('/analyze_example/<filename>', methods=['GET'])
def analyze_example(filename):
    image_path = os.path.join(EXAMPLES_FOLDER, filename)
    if not os.path.exists(image_path):
        return jsonify({"error": "Image not found"}), 404

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Resize the image to a consistent size before analysis
    resized_img = cv2.resize(img, (256, 256))
    
    # Perform blur detection on the resized image
    blurry_status, blur_score = is_blurry(resized_img)
    
    # Perform authenticity detection on the resized image
    authenticity_status = "Model not loaded"
    if authenticity_model:
        authenticity_status = predict_authenticity(resized_img, authenticity_model)
    
    result = {
        "is_blurry": str(blurry_status),
        "blur_score": round(blur_score, 2),
        "clarity_status": "Blurry" if blurry_status else "Sharp",
        "authenticity_status": authenticity_status
    }
    
    print(f"Analysis Result for {filename}: {result}")
    
    return jsonify(result)

# Original upload route remains the same
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
        
        blurry_status, blur_score = is_blurry(resized_img)
        
        authenticity_status = "Model not loaded"
        if authenticity_model:
            authenticity_status = predict_authenticity(resized_img, authenticity_model)
        
        result = {
            "is_blurry": str(blurry_status),
            "blur_score": round(blur_score, 2),
            "clarity_status": "Blurry" if blurry_status else "Sharp",
            "authenticity_status": authenticity_status
        }
        
        print(f"Analysis Result: {result}")
        
        return jsonify(result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)