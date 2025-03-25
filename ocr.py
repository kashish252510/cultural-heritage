import cv2
import pytesseract

# Set the path to Tesseract-OCR (only for Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(image_path):
    """Extract text from an image using Tesseract OCR."""
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding for better text recognition
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

    # Extract text using Tesseract OCR
    extracted_text = pytesseract.image_to_string(processed)

    return extracted_text

# Specify your image file
image_path = r"C:\Users\kashi\Desktop\pro\text\histo2.jpg"  # Update if needed

# Extract and print the text
text = extract_text(image_path)
print("📝 Extracted Text:\n", text)
