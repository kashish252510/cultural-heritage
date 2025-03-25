import cv2
import pytesseract
import matplotlib.pyplot as plt

# Set path to Tesseract-OCR (Update if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(image_path):
    """Extracts text from an image using Tesseract OCR."""
    image = cv2.imread(image_path)

    if image is None:
        print("❌ Error: Could not load image.")
        return ""

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Bilateral Filtering to remove noise
    # Apply Bilateral Filtering to remove noise
    smooth = cv2.bilateralFilter(gray, 9, 75, 75) #bilateral filter to remove noice
    # Increase contrast using Histogram Equalization
    equalized = cv2.equalizeHist(gray)
    # Simple Binary Thresholding
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)





    # Apply adaptive thresholding for better OCR
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

    # Show processed image
    plt.imshow(processed, cmap='gray')
    plt.axis("off")
    plt.title("Processed Image for OCR")
    plt.show()

    # Extract text using Tesseract
    extracted_text = pytesseract.image_to_string(processed)

    return extracted_text.strip()

# Set the image path
image_path = r"C:\Users\kashi\Desktop\pro\text\histo2.jpg"  # Change if needed

# Run OCR and print the extracted text
text = extract_text(image_path)
print("\n📝 Extracted Text:\n", text)