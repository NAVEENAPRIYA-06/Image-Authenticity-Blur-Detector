### Image Authenticity & Blur Detector 

This is a web-based tool developed in Python and Flask that can analyze any uploaded image to determine its authenticity and clarity. The project uses machine learning and computer vision techniques to provide instant analysis.

### Features

  * **Blur Detection:** Analyzes an image for blur and provides a clarity score. The detection is performed by a Convolutional Neural Network (CNN) model for high accuracy.It follows the algorithm of Laplacian Variance.
  * **Authenticity Detection:** Classifies an image as "Genuine" or "Fake" based on forensic analysis performed by a fine-tuned CNN model.
  * **User-Friendly Interface:** An attractive and easy-to-use website for uploading images and viewing results.

### How to Run the Project

Follow these steps to get a copy of the project running on your local machine.

#### 1\. Clone the Repository

```bash
git clone [YOUR_REPOSITORY_URL]
cd Image-Authenticity-Blur-Detector
```

#### 2\. Install Dependencies

It's recommended to use a virtual environment.

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

#### 3\. Run the Application

Navigate to the project's root folder and run the Flask application.

```bash
python app/app.py
```

After running this command, open your web browser and go to `http://127.0.0.1:5000` to see the website.

-----

### Technologies Used

  * **Python:** The core programming language.
  * **Flask:** The web framework for the backend.
  * **TensorFlow/Keras:** Used for building and training the CNN models.
  * **OpenCV:** Used for image processing.
  * **HTML, CSS, JavaScript:** Used for the front-end user interface.