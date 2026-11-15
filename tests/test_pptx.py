"""PowerPoint extraction (no database needed)."""
import os

import pytest
from pptx import Presentation
from pptx.util import Inches

from app import pptx_reader


def make_deck(path):
    prs = Presentation()
    s1 = prs.slides.add_slide(prs.slide_layouts[1])  # title + content
    s1.shapes.title.text = "Software Crisis"
    s1.placeholders[1].text_frame.text = "Conference in 1968"
    s1.placeholders[1].text_frame.add_paragraph().text = "Projects late and over budget"
    s1.notes_slide.notes_text_frame.text = "Mention NATO conference"
    s2 = prs.slides.add_slide(prs.slide_layouts[6])  # blank: table + grouped text
    tbl = s2.shapes.add_table(2, 2, Inches(1), Inches(1), Inches(4), Inches(1)).table
    tbl.cell(0, 0).text, tbl.cell(0, 1).text, tbl.cell(1, 0).text, tbl.cell(1, 1).text = "Model", "Phase", "Waterfall", "Design"
    grp = s2.shapes.add_group_shape()
    grp.shapes.add_textbox(Inches(1), Inches(3), Inches(3), Inches(1)).text_frame.text = "Grouped note"
    prs.slides.add_slide(prs.slide_layouts[6])  # empty slide: must be skipped
    prs.save(path)


def test_extracts_text_tables_groups_notes_and_skips_empty(tmp_path):
    f = tmp_path / "deck.pptx"
    make_deck(f)
    slides = pptx_reader.slide_texts(str(f))
    assert [n for n, _ in slides] == [1, 2]  # slide 3 is empty
    t1, t2 = slides[0][1], slides[1][1]
    assert "Software Crisis" in t1 and "Conference in 1968" in t1 and "Projects late and over budget" in t1
    assert "Notes: Mention NATO conference" in t1
    assert "Model | Phase" in t2 and "Waterfall | Design" in t2 and "Grouped note" in t2


def test_deck_without_text_is_rejected(tmp_path):
    f = tmp_path / "empty.pptx"
    prs = Presentation()
    prs.slides.add_slide(prs.slide_layouts[6])
    prs.save(f)
    with pytest.raises(ValueError, match="No extractable text"):
        pptx_reader.slide_texts(str(f))


from tests.test_native import client  # noqa: E402,F401  (reuse the DB-backed client fixture)


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")
def test_native_upload_pptx_end_to_end(tmp_path, client):
    f = tmp_path / "lecture.pptx"
    make_deck(f)
    with open(f, "rb") as fh:
        r = client.post("/index/file", headers={"X-API-Key": "k"}, files={"file": ("lecture.pptx", fh)},
                        data={"course_id": "c1", "source": "lecture.pptx"})
    assert r.status_code == 200 and r.json()["status"] == "indexed", r.text
    q = client.post("/query", headers={"X-API-Key": "k"}, json={"question": "software crisis conference", "course_id": "c1"})
    refs = q.json()["references"]
    assert refs and refs[0]["source"] == "lecture.pptx" and refs[0]["page"] in (1, 2)
    bad = client.post("/index/file", headers={"X-API-Key": "k"}, files={"file": ("old.ppt", b"x")},
                      data={"course_id": "c1", "source": "old.ppt"})
    assert bad.status_code == 415
