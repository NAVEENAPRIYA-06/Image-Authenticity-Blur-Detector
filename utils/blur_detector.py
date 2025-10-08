import cv2

def is_blurry(image, threshold=100.0):
    """
    Checks if an image is blurry using the Laplacian Variance method.
    """
    # 1. Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply the Laplacian operator and compute variance
    laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 3. Check if the variance is below the threshold
    is_image_blurry = laplacian_variance < threshold
    
    # Return the boolean result and the numerical score
    return is_image_blurry, laplacian_variance