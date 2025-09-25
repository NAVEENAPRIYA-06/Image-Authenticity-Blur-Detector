import tensorflow as tf
import numpy as np

# A function to build the CNN model for training
def build_cnn_model(input_shape):
    """
    Builds a simple CNN model for image authenticity classification.
    """
    model = tf.keras.models.Sequential([
        # First Convolutional Block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Second Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Third Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Flatten the feature maps to feed into the dense layers
        tf.keras.layers.Flatten(),

        # Dense layers for classification
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# A function to load the trained model
def load_authenticity_model(model_path):
    """
    Loads a pre-trained CNN model from a file.
    """
    return tf.keras.models.load_model(model_path)

# A function to make a prediction
def predict_authenticity(image_to_predict, loaded_model):
    """
    Predicts if an image is real or fake.
    """
    # Preprocess the image to match the model's input size
    image_to_predict = tf.image.resize(image_to_predict, (128, 128))
    
    # The model expects a batch of images, so we add a new dimension
    input_array = np.expand_dims(image_to_predict, axis=0)
    
    # Scale the image data to be between 0 and 1
    input_array = input_array / 255.0

    # Make a prediction
    prediction = loaded_model.predict(input_array)[0][0]
    
    # Interpret the prediction
    if prediction > 0.5:
        return "Genuine"
    else:
        return "Fake"

# This is a test block to check the functions
if __name__ == "__main__":
    # Ensure you are at the project root for this test to work
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'authenticity_model.h5')
    test_image_path = "path/to/your/test/image.jpg" # Replace this with a real image path

    try:
        model = load_authenticity_model(model_path)
        test_image = tf.io.read_file(test_image_path)
        test_image = tf.image.decode_image(test_image, channels=3)
        
        result = predict_authenticity(test_image, model)
        print(f"The image is predicted as: {result}")

    except FileNotFoundError:
        print("Error: Make sure you provide the correct paths for the model and test image.")