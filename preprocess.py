import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

# Set Tesseract-OCR path (Windows only, remove this line for Linux/Mac)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path):
    """Preprocess historical document for better OCR accuracy."""
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("❌ Error: Could not load image.")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Bilateral Filtering to remove noise while keeping edges sharp
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Enhance contrast using Histogram Equalization
    equalized = cv2.equalizeHist(denoised)

    # Apply Adaptive Thresholding to create a binary image
    binary = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Apply Morphological Operations to remove small dots & stains
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Show processed images step by step
    images = [gray, denoised, equalized, binary, cleaned]
    titles = ["Grayscale", "Denoised", "Contrast Enhanced", "Thresholded", "Cleaned"]

    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.show()

    return cleaned

def extract_text(image_path):
    """Extract text from a preprocessed historical document."""
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return ""

    # Perform OCR on the cleaned image
    extracted_text = pytesseract.image_to_string(processed_image)

    return extracted_text.strip()

# Change this to the actual path of your historical document image
image_path = "histo2.jpg"

# Extract text and print result
text = extract_text(image_path)
print("\n📝 Extracted Text:\n", text)
