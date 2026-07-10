import pdfplumber
from typing import List, Dict


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, object]]:

    """
    Extracts text from a PDF, page by page.

    Returns a list like:
    [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]

    We keep page numbers attached here — not later — because once text
    gets merged into one big string, page boundaries are gone for good.
    """

    extracted_pages: List[Dict[str, object]] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                
                if page_text:
                    extracted_pages.append({
                        "page": page_number,
                        "text": page_text
                    })

    except FileNotFoundError:
        raise FileNotFoundError(f"No PDF found at path: {pdf_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")

    return extracted_pages