from docx import Document   # python-docx library for reading .docx files

def extract_docx_text(file_path: str) -> str:
    """
    Extracts text from a Microsoft Word (.docx) file.
    """

    doc = Document(file_path)                 # Load the DOCX file
    paragraphs = []                           # Will store text from each paragraph

    for para in doc.paragraphs:               # Loop through every paragraph in the file
        paragraphs.append(para.text)          # Append raw text from the paragraph

    return "\n".join(paragraphs)              # Join all paragraphs with newlines
