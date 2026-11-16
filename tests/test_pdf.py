"""PDF extraction failures become clear ValueErrors (HTTP 422), not crashes (no database needed)."""
import pytest
from pypdf import PdfWriter

from app import pdf


def write(path, data):
    path.write_bytes(data)
    return str(path)


def blank_pdf(path, user_password=None, owner_password=None):
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    if user_password is not None:
        w.encrypt(user_password=user_password, owner_password=owner_password or "owner")
    with open(path, "wb") as f:
        w.write(f)
    return str(path)


@pytest.mark.parametrize("data", [b"Software quality assurance notes, saved with a .pdf name.",
                                  b"%PDF-1.7\n1 0 obj << /Type /Catalog garbage", b""])
def test_not_a_real_pdf(tmp_path, data):
    with pytest.raises(ValueError, match="could not be read as a PDF"):
        pdf.extract_pages(write(tmp_path / "x.pdf", data))


def test_password_protected_pdf(tmp_path):
    with pytest.raises(ValueError, match="password-protected"):
        pdf.extract_pages(blank_pdf(tmp_path / "locked.pdf", user_password="secret"))


def test_pdf_locked_only_for_editing_is_opened(tmp_path):
    # Empty user password: anyone may open it, so it gets past the lock (and then has no text).
    with pytest.raises(ValueError, match="No extractable text"):
        pdf.extract_pages(blank_pdf(tmp_path / "editlock.pdf", user_password=""))
