from pypdf import PdfReader


def extract_pages(path):
    """Return [(page_number, text)] for a text-based PDF. Scanned PDFs raise ValueError."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    if not pages:
        raise ValueError("No extractable text found (scanned PDF?). OCR is not supported.")
    return pages
