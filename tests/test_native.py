"""Runs against a REAL PostgreSQL + pgvector (set TEST_DATABASE_URL); only the embedding model and
Gemini are replaced by stand-ins. Skipped when TEST_DATABASE_URL is not set."""
import hashlib
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

TEST_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_URL, reason="TEST_DATABASE_URL not set")

DIM = 768


def fake_embed(texts, task_type="RETRIEVAL_DOCUMENT", titles=None):
    out = []
    for t in texts:
        v = np.zeros(DIM, dtype=np.float32)
        for w in t.lower().replace(".", " ").split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        out.append(v / (np.linalg.norm(v) or 1.0))
    return np.stack(out)


@pytest.fixture()
def client(monkeypatch):
    from app import config, db, embedding, llm, main
    monkeypatch.setattr(config, "DATABASE_URL", TEST_URL)
    monkeypatch.setattr(config, "SERVICE_API_KEY", "k")
    monkeypatch.setattr(embedding, "embed_texts", fake_embed)
    monkeypatch.setattr(config, "PRELOAD_EMBEDDING", False)
    monkeypatch.setattr(llm, "generate", lambda prompt: "Mitochondria produce ATP [1].")
    with TestClient(main.app) as c:
        with db.pool.connection() as conn:
            conn.execute("TRUNCATE chunks, documents CASCADE")
        yield c


H = {"X-API-Key": "k"}


def up(c, text, course="c1", source="bio.txt", doc_id=None):
    data = {"course_id": course, "source": source}
    if doc_id:
        data["doc_id"] = doc_id
    return c.post("/index/file", headers=H, files={"file": (source, text.encode())}, data=data)


def test_auth(client):
    assert client.post("/query", json={"question": "x", "course_id": "c1"}).status_code == 401


def test_index_query_reference(client):
    r = up(client, "Mitochondria are the organelles that produce ATP for the cell.")
    assert r.json()["status"] == "indexed"
    q = client.post("/query", headers=H, json={"question": "Mitochondria produce ATP", "course_id": "c1"}).json()
    assert q["references"][0]["source"] == "bio.txt" and q["references"][0]["cited"] is True


def test_duplicate_unchanged_replace_delete(client):
    t1 = "Mitochondria are the organelles that produce ATP for the cell."
    d = up(client, t1, doc_id="r1").json()["doc_id"]
    assert up(client, t1, doc_id="r1").json()["status"] == "unchanged"
    assert up(client, t1, source="copy.txt", doc_id="r2").json()["status"] == "duplicate"
    assert up(client, "Ribosomes build proteins from amino acids in the cell.", doc_id="r1").json()["status"] == "replaced"
    q = client.post("/query", headers=H, json={"question": "Mitochondria produce ATP", "course_id": "c1"}).json()
    assert "Mitochondria" not in q["references"][0]["snippet"]  # old text is gone after replace
    assert client.delete("/documents/r1", headers=H, params={"course_id": "c1"}).status_code == 200
    q = client.post("/query", headers=H, json={"question": "Ribosomes build proteins", "course_id": "c1"}).json()
    assert q["references"] == []


def test_course_isolation(client):
    up(client, "Secret exam answers: mitochondria produce ATP.", course="c2", source="exam.txt")
    q = client.post("/query", headers=H, json={"question": "mitochondria produce ATP", "course_id": "c1"}).json()
    assert q["references"] == []
    q2 = client.post("/query", headers=H, json={"question": "mitochondria produce ATP", "course_id": "c2"}).json()
    assert q2["references"][0]["source"] == "exam.txt"
    assert up(client, "x", course="c1", doc_id=client.get("/documents", headers=H, params={"course_id": "c2"}).json()[0]["doc_id"]).status_code == 409


def test_chunking_pages_and_arabic():
    from app.chunking import chunk_pages
    ar = "الميتوكوندريا هي مصدر الطاقة في الخلية. " * 60
    ch = chunk_pages([(1, ar), (2, "Short page.")], size=500, overlap=80)
    assert all(len(t) <= 600 for _, t in ch)
    assert {p for p, _ in ch} == {1, 2} and len(ch) >= 4


def test_follow_up_is_rewritten_and_instructions_stay_out_of_the_search(client, monkeypatch):
    from app import embedding, llm
    up(client, "Mitochondria are the organelles that produce ATP for the cell.", source="bio.txt")
    up(client, "Photosynthesis in chloroplasts turns light into sugar.", source="plants.txt")
    prompts, searched = [], []

    def fake_generate(prompt, system=None):
        prompts.append((system, prompt))
        if system == llm.REWRITE_SYSTEM:
            assert "Q: What do mitochondria do?" in prompt and "New question: explain more" in prompt
            return "Standalone question: Explain in more detail how mitochondria produce ATP"
        return "More detail [1]."
    real_embed_query = embedding.embed_query
    monkeypatch.setattr(llm, "generate", fake_generate)
    monkeypatch.setattr(embedding, "embed_query", lambda text: searched.append(text) or real_embed_query(text))

    q = client.post("/query", headers=H, json={
        "question": "explain more", "course_id": "c1", "instructions": "Answer as a JSON quiz.",
        "history": [{"question": "What do mitochondria do?", "answer": "They produce ATP [1]."}]}).json()
    assert q["rewritten"] == "Explain in more detail how mitochondria produce ATP"
    assert searched == ["Explain in more detail how mitochondria produce ATP"]  # no instructions in the search
    assert q["references"][0]["source"] == "bio.txt"
    answer_prompt = prompts[-1][1]
    assert prompts[-1][0] == llm.SYSTEM + llm.CONVERSATION_RULE and "Question: Explain in more detail" in answer_prompt
    assert "(The student typed: explain more)" in answer_prompt
    assert answer_prompt.endswith("Instructions for the answer: Answer as a JSON quiz.")


def test_questions_about_the_conversation_see_the_conversation(client, monkeypatch):
    from app import llm
    up(client, "Mitochondria are the organelles that produce ATP for the cell.")
    answer_prompts = []

    def fake_generate(prompt, system=None):
        if system == llm.REWRITE_SYSTEM:
            return "what did I ask you before?"  # questions about the conversation stay as they are
        answer_prompts.append((system, prompt))
        return "You asked: What do mitochondria do?"
    monkeypatch.setattr(llm, "generate", fake_generate)
    q = client.post("/query", headers=H, json={"question": "what did I ask you before?", "course_id": "c1",
        "history": [{"question": "What do mitochondria do?", "answer": "They produce ATP [1]."}]}).json()
    system, prompt = answer_prompts[0]
    assert q["rewritten"] is None and system.endswith(llm.CONVERSATION_RULE)
    assert prompt.startswith("Earlier conversation with this student (oldest first):\n\nQ: What do mitochondria do?")
    assert "Question: what did I ask you before?" in prompt


def test_no_history_means_no_rewrite_call_and_a_failed_rewrite_falls_back(client, monkeypatch):
    from app import llm
    up(client, "Mitochondria are the organelles that produce ATP for the cell.")
    calls = []

    def fake_generate(prompt, system=None):
        calls.append(system)
        if system == llm.REWRITE_SYSTEM:
            raise RuntimeError("model down")
        return "ATP [1]."
    monkeypatch.setattr(llm, "generate", fake_generate)
    q = client.post("/query", headers=H, json={"question": "Mitochondria produce ATP", "course_id": "c1"}).json()
    assert calls == [None] and q["rewritten"] is None
    q = client.post("/query", headers=H, json={"question": "Mitochondria produce ATP", "course_id": "c1",
                                               "history": [{"question": "a", "answer": "b"}]}).json()
    assert q["rewritten"] is None and q["references"][0]["cited"] is True


def test_grouped_citations_are_marked_cited(client, monkeypatch):
    from app import llm
    up(client, "Mitochondria are the organelles that produce ATP for the cell.", source="a.txt")
    up(client, "Mitochondria have their own DNA and produce ATP.", source="b.txt")
    monkeypatch.setattr(llm, "generate", lambda prompt, system=None: "They make ATP [1, 2].")
    q = client.post("/query", headers=H, json={"question": "Mitochondria produce ATP", "course_id": "c1"}).json()
    assert [r["cited"] for r in q["references"][:2]] == [True, True]
