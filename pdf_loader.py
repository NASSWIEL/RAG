import requests
from llama_index.readers.file import PDFReader
from pathlib import Path
import tempfile
import os

def load_pdf_from_url(url):
    print(f"Downloading PDF from {url}...")
    response = requests.get(url)
    
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_rag_document.pdf")
    
    with open(temp_path, 'wb') as f:
        f.write(response.content)
    
    return temp_path

def load_documents_from_pdf(pdf_path):
    loader = PDFReader()
    documents = loader.load_data(file=Path(pdf_path))
    return documents
