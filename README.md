# native_rag_gemini

A small RAG (retrieval-augmented generation) service for university course materials. It indexes PDF, PPTX, TXT and
MD files per course, stores the chunks in PostgreSQL with pgvector, and answers questions with Google Gemini, citing
the file and page each fact came from.

It is the backend for two Moodle plugins:

- [moodle-local_ragindex](https://github.com/AbdoAlasta/moodle-local_ragindex): sends each course's files to this service.
- [moodle-local_aitutor](https://github.com/AbdoAlasta/moodle-local_aitutor): the student-facing "AI Tutor" chat.

![System overview](docs/diagrams/native_rag_gemini_storage.png)

## How it works

- **Indexing** (`POST /index/file`): text is extracted per page or slide, split into ~1000-character chunks that never
  cross a page, embedded with `gemini-embedding-2`, and stored atomically. Unchanged and duplicate files are skipped.
- **Answering** (`POST /query`): the question is embedded, the 8 nearest chunks **of that course only** are retrieved,
  and Gemini answers from them with `[n]` citations. With `history`, a follow-up like "explain more" is first rewritten
  into a standalone question.
- **Chat memory**: a client sends the chat's running `summary` plus its last few exchanges. `POST /summarize` folds
  older exchanges into that summary, so a long chat costs about the same as a short one.
- **Guards**: questions whose best passage scores below `MIN_SCORE` get a "not found" reply without a model call;
  near-duplicate passages are sent once; Gemini answer calls retry on 429/5xx and fail as HTTP 503.
- Courses are isolated by a `course_id` filter inside the SQL query.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Status and embedding settings (no key needed) |
| `POST /index/file` | Multipart: `file`, `course_id`, `source` (citation label), optional `doc_id` |
| `POST /query` | JSON: `question`, `course_id`, optional `history`, `instructions`, `summary` |
| `POST /summarize` | JSON: `summary`, `turns`. Folds older chat exchanges into a running summary (chat memory) |
| `GET /documents?course_id=` | List indexed documents |
| `DELETE /documents/{doc_id}?course_id=` | Remove a document |
| `GET /stats` | Gemini call and token counters |

Every endpoint except `/health` requires the header `X-API-Key: <SERVICE_API_KEY>`.

Table layout: [docs/diagrams/native_rag_gemini_tables.png](docs/diagrams/native_rag_gemini_tables.png)

## Installation

### Requirements

- Python 3.10 or newer
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension (0.8+ recommended); Docker is the easiest way
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/apikey))

### 1. Start PostgreSQL with pgvector

```bash
docker run -d --name rags-postgres --restart unless-stopped -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=ragpass -e POSTGRES_DB=native_rag_gemini -p 5433:5432 pgvector/pgvector:pg17
```

Use a stronger password than `ragpass` on any shared machine. The service creates the `vector` extension, tables and
indexes itself on first start.

**Without Docker:** any PostgreSQL 13+ server works once pgvector is installed into it. On Linux, install the package
(e.g. `postgresql-17-pgvector`). On Windows, build or download pgvector for your PostgreSQL version
([instructions](https://github.com/pgvector/pgvector#windows)). Then create a database and user, and point
`DATABASE_URL` at it. Using a separate database from Moodle's is recommended.

### 2. Install the service

```bash
git clone https://github.com/AbdoAlasta/native-rag-gemini.git
```

```bash
cd native-rag-gemini
```

```bash
python -m venv .venv
```

Activate the virtual environment (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on Linux/macOS), then:

```bash
pip install -r requirements.txt
```

### 3. Configure

Copy `.env.example` to `.env` and fill in at least:

| Variable | Meaning |
| --- | --- |
| `DATABASE_URL` | e.g. `postgresql://rag:ragpass@localhost:5433/native_rag_gemini` |
| `GEMINI_API_KEY` | Your Gemini API key |
| `SERVICE_API_KEY` | A long random string; the Moodle plugins send it as `X-API-Key` |
| `GEMINI_MODEL` | Answer model (e.g. `gemini-2.5-flash`) |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `gemini-embedding-2` / `768` (or `1536`) |

`EMBEDDING_DIM` is fixed when the tables are first created. To change it later, use a new, empty database.
Never commit `.env`.

### 4. Run

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8003
```

On Windows, `run.bat` does the same. Check it with:

```bash
curl http://127.0.0.1:8003/health
```

At startup the service embeds one test word, so a wrong API key, model or dimension fails immediately.

**Optional, Windows:** `scripts/autostart.ps1` waits for the database, starts the service and restarts it if it
stops, logging to `logs/service.log`. Register it as a Task Scheduler task that runs at logon:
`powershell -ExecutionPolicy Bypass -File <path>\scripts\autostart.ps1`.

### 5. Try it

```bash
curl -H "X-API-Key: <SERVICE_API_KEY>" -F file=@lecture1.pdf -F course_id=2 -F source=lecture1.pdf http://127.0.0.1:8003/index/file
```

```bash
python scripts/ask.py --course 2
```

`scripts/ask.py` reads the key from `.env` and lets you type questions interactively.

### 6. Connect Moodle

Install [local_ragindex](https://github.com/AbdoAlasta/moodle-local_ragindex) and
[local_aitutor](https://github.com/AbdoAlasta/moodle-local_aitutor), and give both the same service URL
(`http://127.0.0.1:8003`) and `SERVICE_API_KEY`.

## Tests

```bash
pytest
```

The embedding and PPTX tests run without network access. The API tests in `tests/test_native.py` need a real,
empty pgvector database and are skipped unless `TEST_DATABASE_URL` is set (Gemini is replaced by stand-ins):

```bash
TEST_DATABASE_URL=postgresql://rag:ragpass@localhost:5433/native_rag_test pytest
```

## Limitations

- No OCR: scanned PDFs and image-only slides cannot be indexed.
- `.ppt` (old PowerPoint) is not supported; save as `.pptx`.
- The service listens on `127.0.0.1` over plain HTTP. Put it behind HTTPS before exposing it to other machines.
