import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from . import config

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id        text PRIMARY KEY,
    course_id     text NOT NULL,
    source        text NOT NULL,
    content_hash  text NOT NULL,
    n_chunks      int  NOT NULL DEFAULT 0,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);
-- Identical content inside one course is stored once (the same file in another course is allowed).
CREATE UNIQUE INDEX IF NOT EXISTS documents_course_hash ON documents (course_id, content_hash);

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    doc_id       text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    course_id    text NOT NULL,
    chunk_index  int  NOT NULL,
    page         int,
    content      text NOT NULL,
    embedding    vector({config.EMBEDDING_DIM}) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_course ON chunks (course_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
"""

pool = None


def init():
    global pool
    with psycopg.connect(config.DATABASE_URL, autocommit=True) as conn:
        conn.execute(SCHEMA)
    pool = ConnectionPool(config.DATABASE_URL, min_size=1, max_size=5, configure=register_vector, open=True)


def close():
    if pool:
        pool.close()


def find_by_id(doc_id):
    with pool.connection() as conn:
        return conn.execute(
            "SELECT doc_id, course_id, source, content_hash, n_chunks FROM documents WHERE doc_id=%s", (doc_id,)
        ).fetchone()


def find_by_hash(course_id, content_hash):
    with pool.connection() as conn:
        return conn.execute(
            "SELECT doc_id FROM documents WHERE course_id=%s AND content_hash=%s", (course_id, content_hash)
        ).fetchone()


def store_document(doc_id, course_id, source, content_hash, chunks, embeddings):
    """Atomically replace everything stored for doc_id. chunks: [(page, text)]."""
    with pool.connection() as conn:  # one transaction: commit on exit, rollback on error
        conn.execute("DELETE FROM documents WHERE doc_id=%s", (doc_id,))  # cascades to chunks
        conn.execute(
            "INSERT INTO documents (doc_id, course_id, source, content_hash, n_chunks) VALUES (%s,%s,%s,%s,%s)",
            (doc_id, course_id, source, content_hash, len(chunks)),
        )
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (doc_id, course_id, chunk_index, page, content, embedding) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                [(doc_id, course_id, i, page, text, emb) for i, ((page, text), emb) in enumerate(zip(chunks, embeddings))],
            )


def delete_document(doc_id, course_id):
    with pool.connection() as conn:
        cur = conn.execute("DELETE FROM documents WHERE doc_id=%s AND course_id=%s", (doc_id, course_id))
        return cur.rowcount > 0


def list_documents(course_id):
    with pool.connection() as conn:
        return conn.execute(
            "SELECT doc_id, source, n_chunks, updated_at FROM documents WHERE course_id=%s ORDER BY source",
            (course_id,),
        ).fetchall()


def search(course_id, query_vec, k):
    """Nearest chunks inside ONE course. The course filter is part of the query, so chunks from other
    courses can never be returned."""
    with pool.connection() as conn:
        try:  # pgvector >= 0.8: keep scanning the HNSW index until k filtered rows are found
            conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
        except psycopg.Error:
            conn.rollback()
        return conn.execute(
            "SELECT d.source, d.doc_id, c.page, c.content, 1 - (c.embedding <=> %s) AS score "
            "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
            "WHERE c.course_id = %s ORDER BY c.embedding <=> %s LIMIT %s",
            (query_vec, course_id, query_vec, k),
        ).fetchall()
