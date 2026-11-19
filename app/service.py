"""Core logic, independent of FastAPI so it is easy to test."""
import hashlib
import logging
import random
import re
import time

from . import chunking, config, db, embedding, llm

log = logging.getLogger(__name__)


def content_hash(pages):
    h = hashlib.sha256()
    for page, text in pages:
        h.update(f"{page}\n{text}\n".encode("utf-8"))
    return h.hexdigest()


_ARABIC = re.compile(r"[؀-ۿ]")


# Replies to off-topic questions (no model call), in a few wordings so the tutor does not repeat one sentence.
NOT_FOUND = {
    "en": [
        "That one isn't covered in this course, so I can't answer it reliably. Try asking about a topic from the lectures.",
        "I don't have anything on that in this course. If it's related to one of the lectures, try rephrasing it.",
        "That's outside what this course covers. Ask me about something from the lectures and I'll help.",
        "I couldn't find that topic in this course. Could you rephrase it, or ask about something from the lectures?",
    ],
    "ar": [
        "هذا الموضوع غير مغطى في هذا المقرر، لذلك لا أستطيع الإجابة عنه بدقة. جرّب السؤال عن موضوع من المحاضرات.",
        "لا يوجد لدي شيء عن هذا في هذا المقرر. إن كان مرتبطاً بإحدى المحاضرات، جرّب صياغته بطريقة أخرى.",
        "هذا خارج ما يغطيه المقرر. اسألني عن أي موضوع من المحاضرات وسأساعدك.",
        "لم أجد هذا الموضوع في المقرر. هل يمكنك إعادة صياغة سؤالك، أو السؤال عن شيء من المحاضرات؟",
    ],
}


def not_found_message(question):
    return random.choice(NOT_FOUND["ar" if _ARABIC.search(question) else "en"])


def drop_near_duplicates(rows, threshold=0.8):
    """rows come best first; a row whose words overlap an earlier kept row by >= threshold (Jaccard) is dropped, so
    text repeated across pages (recaps, repeated slides) does not fill the context several times."""
    kept, seen = [], []
    for row in rows:
        words = set(re.findall(r"\w+", row[3].lower()))
        if any(len(words & s) / (len(words | s) or 1) >= threshold for s in seen):
            continue
        kept.append(row)
        seen.append(words)
    return kept


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


def answer(course_id, question, history=None, instructions="", summary="", style=""):
    """history = [(earlier_question, earlier_answer)], oldest first, and summary = the older part of the chat in short:
    when given, one small extra model call rewrites a follow-up ("explain more") so it stands alone before searching
    (it only sees the summary and the last config.REWRITE_TURNS exchanges). instructions (answer format, tone...) go
    to the answer model only, never into the search. style (hint mode...) goes to the answer model too but, unlike
    instructions, keeps the relevance cut-off, because it comes with real questions. result['rewritten'] is the searched question when it differs."""
    t0 = time.time()
    history = history or []
    summary = (summary or "").strip()
    searched = question
    if history or summary:
        try:
            recent = history[-config.REWRITE_TURNS:] if config.REWRITE_TURNS > 0 else []
            searched = llm.rewrite_question(question, recent, summary) or question
        except Exception:  # the rewrite is a help, never a reason to fail: use the question as typed
            log.exception("could not rewrite the question")
    rewritten = searched if " ".join(searched.split()).lower() != " ".join(question.split()).lower() else None
    try:
        qvec = embedding.embed_query(searched)
    except Exception as e:  # noqa: BLE001 - reported to the caller as "try again", details in the log
        log.exception("could not embed the question")
        raise llm.ModelUnavailable("Gemini embedding failed") from e
    rows = db.search(course_id, qvec, config.TOP_K * 2)  # extra candidates: some are dropped below
    if not rows:
        return {"answer": "No course materials are indexed for this course yet.", "references": [],
                "rewritten": rewritten, "seconds": round(time.time() - t0, 2)}
    rows = drop_near_duplicates(rows)
    # Actions (quiz, study plan, topics...) carry instructions and ask about the course as a whole, so their generic
    # wording scores low on purpose: the cut-off is only for questions.
    if config.MIN_SCORE > 0 and not instructions.strip():
        rows = [r for r in rows if r[4] >= config.MIN_SCORE]
    rows = rows[:config.TOP_K]
    if not rows and not history and not summary:  # nothing relevant: answer without spending a model call
        return {"answer": not_found_message(question), "references": [], "rewritten": rewritten,
                "seconds": round(time.time() - t0, 2)}
    # With history but no relevant passage, the model still sees the conversation (for "what did I ask before?").
    context = "\n\n".join(
        f"[{i}] (source: {src}{', page ' + str(page) if page else ''})\n{content}"
        for i, (src, _doc, page, content, _score) in enumerate(rows, start=1)
    ) or "(No course passage matched this question.)"
    prompt = f"Context passages:\n\n{context}\n\nQuestion: {searched}"
    if rewritten:
        prompt += f"\n(The student typed: {question})"
    if instructions.strip():
        prompt += f"\n\nInstructions for the answer: {instructions.strip()}"
    if (style or "").strip():
        prompt += f"\n\nAnswer style: {style.strip()}"
    if history or summary:
        # The conversation is given too, so questions about it ("what did I ask before?") can be answered.
        earlier = f"{llm.SUMMARY_LABEL}\n{summary}\n\n" if summary else ""
        if history:
            turns = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in history)
            label = ("Most recent exchanges, after the summary (oldest first)" if summary
                     else "Earlier conversation with this student (oldest first)")
            earlier += f"{label}:\n\n{turns}\n\n"
        prompt = earlier + prompt
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
