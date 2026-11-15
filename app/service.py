"""Core logic, independent of FastAPI so it is easy to test."""
import hashlib
import logging
import re
import time

from . import chunking, config, db, embedding, llm

log = logging.getLogger(__name__)


def content_hash(pages):
    h = hashlib.sha256()
    for page, text in pages:
        h.update(f"{page}\n{text}\n".encode("utf-8"))
    return h.hexdigest()


def index_document(course_id, source, pages, doc_id=None):
    t0 = time.time()
    chash = content_hash(pages)
    doc_id = doc_id or "doc-" + chash[:32]

    existing = db.find_by_id(doc_id)
    if existing and existing[1] != course_id:
        raise PermissionError("doc_id already belongs to another course")
    if existing and existing[3] == chash:
        return {"status": "unchanged", "doc_id": doc_id, "chunks": existing[4], "seconds": 0.0}
    same = db.find_by_hash(course_id, chash)
    if same and same[0] != doc_id:
        return {"status": "duplicate", "doc_id": same[0], "chunks": 0, "seconds": 0.0}

    chunks = chunking.chunk_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    embeddings = embedding.embed_texts([c[1] for c in chunks], titles=[source] * len(chunks))  # before the DB write: old data stays until ready
    db.store_document(doc_id, course_id, source, chash, chunks, embeddings)
    return {
        "status": "replaced" if existing else "indexed",
        "doc_id": doc_id,
        "chunks": len(chunks),
        "seconds": round(time.time() - t0, 2),
    }


def answer(course_id, question, history=None, instructions=""):
    """history = [(earlier_question, earlier_answer)], oldest first: when given, one small extra model call rewrites a
    follow-up ("explain more") so it stands alone before searching. instructions (answer format, tone...) go to the
    answer model only, never into the search. result['rewritten'] is the searched question when it differs."""
    t0 = time.time()
    searched = question
    if history:
        try:
            searched = llm.rewrite_question(question, history) or question
        except Exception:  # the rewrite is a help, never a reason to fail: use the question as typed
            log.exception("could not rewrite the question")
    rewritten = searched if " ".join(searched.split()).lower() != " ".join(question.split()).lower() else None
    qvec = embedding.embed_query(searched)
    rows = db.search(course_id, qvec, config.TOP_K)
    if not rows:
        return {"answer": "No course materials are indexed for this course yet.", "references": [],
                "rewritten": rewritten, "seconds": round(time.time() - t0, 2)}
    context = "\n\n".join(
        f"[{i}] (source: {src}{', page ' + str(page) if page else ''})\n{content}"
        for i, (src, _doc, page, content, _score) in enumerate(rows, start=1)
    )
    prompt = f"Context passages:\n\n{context}\n\nQuestion: {searched}"
    if rewritten:
        prompt += f"\n(The student typed: {question})"
    if instructions.strip():
        prompt += f"\n\nInstructions for the answer: {instructions.strip()}"
    if history:
        # The conversation is given too, so questions about it ("what did I ask before?") can be answered.
        turns = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in history)
        prompt = f"Earlier conversation with this student (oldest first):\n\n{turns}\n\n{prompt}"
        text = llm.generate(prompt, system=llm.SYSTEM + llm.CONVERSATION_RULE)
    else:
        text = llm.generate(prompt)
    # [1], and the grouped form models also write: [1, 2] / [1,2,4]
    cited = {int(n) for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text) for n in re.findall(r"\d+", group)}
    refs = [
        {"ref": i, "source": src, "doc_id": doc, "page": page, "score": round(float(score), 4),
         "cited": i in cited, "snippet": content[:300]}
        for i, (src, doc, page, content, score) in enumerate(rows, start=1)
    ]
    return {"answer": text, "references": refs, "rewritten": rewritten, "seconds": round(time.time() - t0, 2)}
