import cv2
import pytesseract
from transformers import pipeline

# Set Tesseract path (Change if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load an image
image = cv2.imread("histo2.jpg")

# Check if image is loaded correctly
if image is None:
    print("Error: Image not found or cannot be read.")
else:
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to improve contrast
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Perform OCR on the preprocessed image
    extracted_text = pytesseract.image_to_string(thresh)
    print("Extracted Text:\n", extracted_text)

    # Check if extracted text is not empty
    if len(extracted_text.strip()) == 0:
        print("Error: No text detected.")
    else:
        # Load BERT fill-mask model
        fill_mask = pipeline("fill-mask", model="bert-base-uncased")

        # Split text into smaller chunks (max 512 characters)
        max_length = 500  # Keep it safe under 512
        text_chunks = [extracted_text[i:i + max_length] for i in range(0, len(extracted_text), max_length)]

        # Process each chunk separately
        for idx, chunk in enumerate(text_chunks):
            print(f"\nProcessing Chunk {idx+1}:\n")
            
            # Replace missing letters (underscores) with [MASK]
            masked_text = chunk.replace("_", "[MASK]")

            # Ensure at least one [MASK] token exists
            if "[MASK]" in masked_text:
                predictions = fill_mask(masked_text)

                print("\nPredicted Restored Text:")
                for pred in predictions[:5]:  # Show top 5 results
                    print(f"{pred['sequence']} (Confidence: {pred['score']:.4f})")
            else:
                print("No missing letters detected in this chunk.")
