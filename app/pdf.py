from pypdf import PdfReader


def extract_pages(path):
    """Return [(page_number, text)] for a text-based PDF. Raises ValueError (sent back as HTTP 422, which the Moodle
    indexer treats as final) for a damaged, password-protected or scanned PDF."""
    pages = []
    try:
        reader = PdfReader(path)
        locked = reader.is_encrypted and not reader.decrypt("")  # many "protected" PDFs only restrict editing
        if not locked:
            for i, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append((i, text))
    except Exception as e:  # noqa: BLE001 - pypdf raises many types for a broken file; all mean "cannot read it"
        raise ValueError(f"This file could not be read as a PDF (damaged or not a real PDF): {type(e).__name__}") from e
    if locked:
        raise ValueError("This PDF is password-protected. Remove the password and upload it again.")
    if not pages:
        raise ValueError("No extractable text found (scanned PDF?). OCR is not supported.")
    return pages
