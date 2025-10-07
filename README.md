Image Authenticity & Blur Detector

This is a web-based tool developed using Python and Flask that analyzes any uploaded image to determine its authenticity (Is it real or fake?) and clarity (Is it sharp or blurry?).

The project uses advanced machine learning models and computer vision techniques to provide instant analysis.

🌐 Live Application
The application is currently deployed and live on Render:

[LAUNCH APP HERE] (https://image-authenticity-blur-detector.onrender.com)

Key Features:

Authenticity Detection (CNN): Uses a fine-tuned Convolutional Neural Network (CNN) model to classify images as "Genuine" or "Fake" by detecting subtle artifacts and manipulation patterns.

Blur Detection (Laplacian): Utilizes the highly effective Laplacian Variance algorithm to calculate a numerical clarity score and determine the sharpness of the image.

Responsive UI: Designed to work seamlessly on both mobile devices and desktop computers.

Local Setup (For Developers)
Follow these steps to run the project locally on your machine.

1. Clone the Repository
git clone [https://github.com/NAVEENAPRIYA-06/Image-Authenticity-Blur-Detector.git](https://github.com/NAVEENAPRIYA-06/Image-Authenticity-Blur-Detector.git)
cd Image-Authenticity-Blur-Detector

2. Install Dependencies
Set up your virtual environment and install all necessary Python packages.

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install all libraries, including TensorFlow, OpenCV, and Flask
pip install -r requirements.txt

3. Run the Application
Navigate to the project's root folder and run the Flask application.

python app/app.py

Open your web browser and navigate to http://127.0.0.1:5000.