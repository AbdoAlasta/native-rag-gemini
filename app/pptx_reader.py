from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE


def _shape_text(shape):
    out = []
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for s in shape.shapes:
            out += _shape_text(s)
    elif getattr(shape, "has_table", False) and shape.has_table:
        for row in shape.table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                out.append(" | ".join(cells))
    elif getattr(shape, "has_text_frame", False) and shape.has_text_frame:
        for para in shape.text_frame.paragraphs:
            t = "".join(r.text for r in para.runs).strip()
            if t:
                out.append(t)
    return out


def slide_texts(path):
    """[(slide_number, text)] for a .pptx: titles/text boxes/tables in reading order, then speaker notes.
    Pictures are ignored (no OCR), so slides that are only images are skipped. Raises ValueError (HTTP 422, final for
    the Moodle indexer) for a damaged or password-protected file, or one without text."""
    slides = []
    try:
        prs = Presentation(path)
        for i, slide in enumerate(prs.slides, start=1):
            lines = []
            for shape in slide.shapes:
                lines += _shape_text(shape)
            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip() if slide.notes_slide.notes_text_frame else ""
                if notes:
                    lines.append("Notes: " + notes)
            text = "\n".join(lines).strip()
            if text:
                slides.append((i, text))
    except Exception as e:  # noqa: BLE001 - python-pptx / zipfile / lxml errors all mean "cannot read it"
        raise ValueError("This file could not be read as a PowerPoint .pptx (damaged, password-protected, or not a "
                         f"real .pptx): {type(e).__name__}") from e
    if not slides:
        raise ValueError("No extractable text found in this PowerPoint (only images?). OCR is not supported.")
    return slides
