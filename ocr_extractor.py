import pytesseract                   # OCR engine for reading text from images
from PIL import Image                # PIL to open images

def extract_image_text(file_path: str) -> str:
    """
    Uses Tesseract OCR to read text from image files (JPG, PNG).
    """

    image = Image.open(file_path)              # Open the image using PIL
    text = pytesseract.image_to_string(image)  # Perform OCR to extract text
    return text                                # Return extracted text
