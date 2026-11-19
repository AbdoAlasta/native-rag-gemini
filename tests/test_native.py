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
    monkeypatch.setattr(config, "MIN_SCORE", 0.0)  # fake word-count vectors score lower than Gemini; tests below set it
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


def test_off_topic_question_is_answered_without_a_model_call(client, monkeypatch):
    from app import config, llm, service
    up(client, "Mitochondria are the organelles that produce ATP for the cell.")
    monkeypatch.setattr(config, "MIN_SCORE", 0.3)
    calls = []
    monkeypatch.setattr(llm, "generate", lambda prompt, system=None: calls.append(prompt) or "x [1].")
    q = client.post("/query", headers=H, json={"question": "How do I bake chocolate cake?", "course_id": "c1"}).json()
    assert calls == [] and q["references"] == [] and q["answer"] in service.NOT_FOUND["en"]
    q = client.post("/query", headers=H, json={"question": "ما هي عاصمة مصر؟", "course_id": "c1"}).json()
    assert calls == [] and q["answer"] in service.NOT_FOUND["ar"]
    q = client.post("/query", headers=H, json={"question": "Mitochondria produce ATP", "course_id": "c1"}).json()
    assert len(calls) == 1 and q["references"][0]["source"] == "bio.txt"


def test_style_reaches_the_model_but_keeps_the_cut_off(client, monkeypatch):
    from app import config, embedding, llm, service
    up(client, "Mitochondria are the organelles that produce ATP for the cell.")
    monkeypatch.setattr(config, "MIN_SCORE", 0.3)
    searched, prompts = [], []
    real_embed_query = embedding.embed_query
    monkeypatch.setattr(embedding, "embed_query", lambda text: searched.append(text) or real_embed_query(text))
    monkeypatch.setattr(llm, "generate", lambda prompt, system=None: prompts.append(prompt) or "A hint [1].")
    style = "HINT MODE. This is hint 1 of 3."
    q = client.post("/query", headers=H, json={"question": "How do I bake chocolate cake?", "course_id": "c1",
                                               "style": style}).json()
    assert prompts == [] and q["answer"] in service.NOT_FOUND["en"]  # off-topic: still no model call
    q = client.post("/query", headers=H, json={"question": "Mitochondria produce ATP", "course_id": "c1",
                                               "style": style}).json()
    assert len(prompts) == 1 and prompts[0].endswith("Answer style: " + style)
    assert all("HINT" not in s for s in searched)  # the style is never searched


def test_actions_skip_the_cut_off_and_history_still_reaches_the_model(client, monkeypatch):
    from app import config, llm
    up(client, "Mitochondria are the organelles that produce ATP for the cell.")
    monkeypatch.setattr(config, "MIN_SCORE", 0.3)
    prompts = []

    def fake_generate(prompt, system=None):
        if system == llm.REWRITE_SYSTEM:
            return "what did I ask before?"
        prompts.append(prompt)
        return "{}"
    monkeypatch.setattr(llm, "generate", fake_generate)
    q = client.post("/query", headers=H, json={"question": "Make me a study plan", "course_id": "c1",
                                               "instructions": "Respond with ONLY valid JSON"}).json()
    assert len(prompts) == 1 and q["references"][0]["source"] == "bio.txt"  # generic action wording is not cut off
    q = client.post("/query", headers=H, json={"question": "what did I ask before?", "course_id": "c1",
        "history": [{"question": "What do mitochondria do?", "answer": "They produce ATP."}]}).json()
    assert len(prompts) == 2 and q["references"] == []
    assert "(No course passage matched this question.)" in prompts[-1] and "Q: What do mitochondria do?" in prompts[-1]


def test_near_duplicate_passages_are_given_once(client):
    up(client, "Software quality assurance checks that the process and product meet the standards.", source="ch1.txt")
    up(client, "Software quality assurance checks that the process and product meet the standards today.",
       source="recap.txt")
    up(client, "Unit testing checks one software module on its own.", source="ch2.txt")
    q = client.post("/query", headers=H, json={"question": "software quality assurance", "course_id": "c1"}).json()
    assert [r["source"] for r in q["references"]] in (["ch1.txt", "ch2.txt"], ["recap.txt", "ch2.txt"])


def test_request_limits(client, monkeypatch):
    from app import config
    assert client.post("/query", headers=H, json={"question": "", "course_id": "c1"}).status_code == 422
    long = "x" * (config.MAX_QUESTION_CHARS + 1)
    assert client.post("/query", headers=H, json={"question": long, "course_id": "c1"}).status_code == 422
    turn = {"question": "a", "answer": long}
    assert client.post("/query", headers=H, json={"question": "q", "course_id": "c1", "history": [turn]}).status_code == 422
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 1)
    r = up(client, "a" * (1024 * 1024 + 1))
    assert r.status_code == 413 and client.get("/documents", headers=H, params={"course_id": "c1"}).json() == []
    assert up(client, "small file").status_code == 200


def test_model_failure_is_a_503(client, monkeypatch):
    from app import llm
    up(client, "Mitochondria are the organelles that produce ATP for the cell.")

    def down(prompt, system=None):
        raise llm.ModelUnavailable("Gemini answer failed: 503")
    monkeypatch.setattr(llm, "generate", down)
    r = client.post("/query", headers=H, json={"question": "Mitochondria produce ATP", "course_id": "c1"})
    assert r.status_code == 503 and "try again" in r.json()["detail"]


def test_summary_and_recent_turns_reach_the_model_and_the_rewrite_sees_only_the_last_two(client, monkeypatch):
    from app import config, llm
    up(client, "Mitochondria are the organelles that produce ATP for the cell.")
    seen = {}

    def fake_generate(prompt, system=None):
        seen["rewrite" if system == llm.REWRITE_SYSTEM else "answer"] = prompt
        return "Explain how mitochondria make ATP" if system == llm.REWRITE_SYSTEM else "ATP [1]."
    monkeypatch.setattr(llm, "generate", fake_generate)
    monkeypatch.setattr(config, "REWRITE_TURNS", 2)
    history = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(1, 5)]
    q = client.post("/query", headers=H, json={"question": "explain more", "course_id": "c1", "history": history,
                                               "summary": "The student asked about cell organelles."}).json()
    assert q["rewritten"] == "Explain how mitochondria make ATP"
    assert "The student asked about cell organelles." in seen["rewrite"]
    assert "Q: q3" in seen["rewrite"] and "Q: q4" in seen["rewrite"] and "Q: q2" not in seen["rewrite"]
    answer = seen["answer"]
    assert answer.startswith(llm.SUMMARY_LABEL + "\nThe student asked about cell organelles.")
    assert all(f"Q: q{i}" in answer for i in range(1, 5))


def test_a_summary_alone_still_lets_the_model_answer_about_the_chat(client, monkeypatch):
    from app import config, llm
    up(client, "Mitochondria are the organelles that produce ATP for the cell.")
    monkeypatch.setattr(config, "MIN_SCORE", 0.3)
    prompts = []

    def fake_generate(prompt, system=None):
        prompts.append((system, prompt))
        return "what did I ask first?" if system == llm.REWRITE_SYSTEM else "You first asked about mitochondria."
    monkeypatch.setattr(llm, "generate", fake_generate)
    q = client.post("/query", headers=H, json={"question": "what did I ask first?", "course_id": "c1",
                                               "summary": "First the student asked what mitochondria do."}).json()
    assert q["answer"] == "You first asked about mitochondria." and q["references"] == []
    assert "First the student asked what mitochondria do." in prompts[-1][1]


def test_summarize_endpoint(client, monkeypatch):
    from app import config, llm
    got = {}

    def fake_generate(prompt, system=None):
        got["prompt"], got["system"] = prompt, system
        return "Updated summary: " + "s" * 5000
    monkeypatch.setattr(llm, "generate", fake_generate)
    r = client.post("/summarize", headers=H, json={"summary": "Earlier: organelles.",
                    "turns": [{"question": "What is ATP?", "answer": "Energy currency [1]."}]})
    assert r.status_code == 200 and r.json()["summary"] == "s" * config.SUMMARY_MAX_CHARS
    assert "Summary so far:\nEarlier: organelles." in got["prompt"] and "Q: What is ATP?" in got["prompt"]
    assert got["system"].startswith("You keep a short running summary")
    assert client.post("/summarize", headers=H, json={"summary": "", "turns": []}).status_code == 422
    assert client.post("/summarize", json={"turns": [{"question": "a", "answer": "b"}]}).status_code == 401

    def down(prompt, system=None):
        raise llm.ModelUnavailable("Gemini answer failed: 429")
    monkeypatch.setattr(llm, "generate", down)
    r = client.post("/summarize", headers=H, json={"turns": [{"question": "a", "answer": "b"}]})
    assert r.status_code == 503


def test_broken_upload_is_a_422_with_a_reason(client):
    r = up(client, "plain text saved as a pdf", source="fake.pdf")
    assert r.status_code == 422 and "could not be read as a PDF" in r.json()["detail"]
    r = up(client, "plain text saved as a pptx", source="fake.pptx")
    assert r.status_code == 422 and "could not be read as a PowerPoint" in r.json()["detail"]


def test_wrong_key_and_no_public_api_docs(client):
    assert client.post("/query", headers={"X-API-Key": "kk"}, json={"question": "x", "course_id": "c1"}).status_code == 401
    assert client.post("/query", headers={"X-API-Key": "é".encode("latin-1")}, json={"question": "x", "course_id": "c1"}).status_code == 401
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404
    assert client.get("/health").status_code == 200
