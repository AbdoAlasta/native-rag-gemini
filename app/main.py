import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import config, db, embedding, llm, pdf, pptx_reader, service

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    if config.PRELOAD_EMBEDDING:
        embedding.warmup()
    yield
    db.close()


app = FastAPI(title="Native RAG (pgvector, Gemini embeddings)", lifespan=lifespan)


def require_key(x_api_key: str = Header(default="")):
    if not config.SERVICE_API_KEY or x_api_key != config.SERVICE_API_KEY:
        raise HTTPException(401, "Invalid or missing X-API-Key")


class Turn(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str
    course_id: str
    mode: str | None = None  # accepted for API parity with the LightRAG service; ignored here
    history: list[Turn] = Field(default=[], max_length=10)  # earlier exchanges, oldest first (follow-up questions)
    instructions: str = Field(default="", max_length=4000)  # answer format/tone, not used for the search


@app.get("/health")
def health():
    return {"status": "ok", "engine": "native-pgvector-gemini-embeddings", "embedding": embedding.info()}


@app.get("/stats", dependencies=[Depends(require_key)])
def stats():
    return llm.stats.snapshot()


@app.post("/index/file", dependencies=[Depends(require_key)])
def index_file(
    file: UploadFile = File(...),
    course_id: str = Form(...),
    source: str = Form(..., description="Citation label, e.g. 'lecture1.pdf'"),
    doc_id: str | None = Form(None),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".pptx", ".txt", ".md"}:
        raise HTTPException(415, "Only .pdf, .pptx, .txt and .md are supported (save old .ppt files as .pptx)")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name
    try:
        if suffix == ".pdf":
            pages = pdf.extract_pages(path)
        elif suffix == ".pptx":
            pages = pptx_reader.slide_texts(path)  # the "page" of a reference is the slide number
        else:
            pages = [(None, Path(path).read_text(encoding="utf-8", errors="replace"))]
        return {**service.index_document(course_id, source, pages, doc_id), "source": source}
    except ValueError as e:
        raise HTTPException(422, str(e))
    except PermissionError as e:
        raise HTTPException(409, str(e))
    finally:
        Path(path).unlink(missing_ok=True)


@app.post("/query", dependencies=[Depends(require_key)])
def query(req: AskRequest):
    history = [(t.question, t.answer) for t in req.history]
    return service.answer(req.course_id, req.question, history, req.instructions)


@app.delete("/documents/{doc_id}", dependencies=[Depends(require_key)])
def delete_document(doc_id: str, course_id: str):
    if not db.delete_document(doc_id, course_id):
        raise HTTPException(404, "No such document in this course")
    return {"status": "deleted", "doc_id": doc_id}


@app.get("/documents", dependencies=[Depends(require_key)])
def list_documents(course_id: str):
    return [{"doc_id": r[0], "source": r[1], "chunks": r[2], "updated_at": r[3].isoformat()}
            for r in db.list_documents(course_id)]
