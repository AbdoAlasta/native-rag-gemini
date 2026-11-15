import re

# Sentence/paragraph boundaries, including the Arabic question mark and full stop variants.
_BOUNDARY = re.compile(r"(?<=[.!?؟۔…])\s+|\n+")


def _units(text, max_len):
    out = []
    for u in _BOUNDARY.split(text):
        u = u.strip()
        while len(u) > max_len:  # hard-split very long unbroken text
            cut = u.rfind(" ", 0, max_len)
            cut = cut if cut > max_len // 2 else max_len
            out.append(u[:cut].strip())
            u = u[cut:].strip()
        if u:
            out.append(u)
    return out


def _tail(text, overlap):
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    t = text[-overlap:]
    sp = t.find(" ")
    return t[sp + 1:] if 0 <= sp < len(t) - 1 else t


def chunk_pages(pages, size=1000, overlap=150):
    """pages: [(page_number|None, text)] -> [(page_number|None, chunk_text)]. Chunks never cross pages,
    so every chunk has an exact page for citation."""
    chunks = []
    for page, text in pages:
        cur = ""
        for u in _units(text, max(size - overlap, 100)):
            if cur and len(cur) + 1 + len(u) > size:
                chunks.append((page, cur))
                cur = (_tail(cur, overlap) + " " + u).strip()
            else:
                cur = (cur + " " + u).strip()
        if cur:
            chunks.append((page, cur))
    return chunks
