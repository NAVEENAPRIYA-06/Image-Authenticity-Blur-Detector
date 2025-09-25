import cv2

def is_blurry(image, threshold=400.0):
    """
    Checks if an image is blurry using the Laplacian Variance method.

    Args:
        image: The image to analyze (loaded by OpenCV).
        threshold: The variance score below which an image is considered blurry.

    Returns:
        A tuple containing:
        - True/False: If the image is blurry or not.
        - The variance score of the image.
    """
    # 1. Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply the Laplacian operator to the grayscale image
    laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 3. Check if the variance is below the threshold
    is_image_blurry = laplacian_variance < threshold
    
    # Return the boolean result and the numerical score
    return is_image_blurry, laplacian_variance

# Optional: A simple example to test the function
if __name__ == "__main__":
    # Replace 'path/to/your/image.jpg' with the path to a test image
    test_image_path = 'path/to/your/image.jpg' 
    image_to_check = cv2.imread(test_image_path)
    
    if image_to_check is None:
        print("Error: Could not read the image.")
    else:
        blurry, score = is_blurry(image_to_check)
        print(f"Image is blurry: {blurry}")
        print(f"Laplacian Variance Score: {score:.2f}")