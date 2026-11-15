"""Gemini embeddings instead of a local model. Works with gemini-embedding-2 (default) and gemini-embedding-001.

* gemini-embedding-2: no task_type; questions are sent as "task: search result | query: ..." and chunks as
  "title: <file> | text: ...". Every text goes in its OWN Content object, otherwise the API returns a single
  averaged vector for the whole list.
* gemini-embedding-001: task types RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY.
* Vectors are L2-normalised, so cosine == dot product. Requests are batched (<= EMBEDDING_BATCH_SIZE texts),
  retried with back-off on 429/5xx, and a batch rejected with 400 (e.g. too many tokens) is split in half.
"""
import logging
import time

import numpy as np

from . import config

log = logging.getLogger("embedding")
_client = None
_RETRY_CODES = {408, 429, 500, 502, 503, 504}


def _get_client():
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set (needed for Gemini embeddings)")
        from google import genai

        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _normalise(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _is_v2():
    return config.EMBEDDING_MODEL.startswith("gemini-embedding-2")


def format_query(text):
    """gemini-embedding-2 has no task_type: the task is stated in the text itself."""
    return f"task: search result | query: {text}"


def format_document(text, title=None):
    return f"title: {title or 'none'} | text: {text}"


def _stats_add(**kw):
    from . import llm  # imported lazily: keeps this module importable without the Gemini LLM client

    llm.stats.add(**kw)


def _embed_batch(batch, task_type):
    from google.genai import types

    if _is_v2():
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in batch]  # one vector per text
        cfg = types.EmbedContentConfig(output_dimensionality=config.EMBEDDING_DIM)
    else:
        contents = batch
        cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=config.EMBEDDING_DIM)
    for attempt in range(config.EMBEDDING_MAX_RETRIES + 1):
        try:
            resp = _get_client().models.embed_content(model=config.EMBEDDING_MODEL, contents=contents, config=cfg)
            vecs = np.asarray([e.values for e in resp.embeddings], dtype=np.float32)
            if len(vecs) != len(batch):
                raise RuntimeError(f"Gemini returned {len(vecs)} embeddings for {len(batch)} texts")
            _stats_add(embed_calls=1, embed_chars=sum(len(t) for t in batch))
            return vecs
        except Exception as e:  # noqa: BLE001 - decide below whether it is worth retrying
            code = getattr(e, "code", None)
            retryable = code in _RETRY_CODES or isinstance(e, (ConnectionError, TimeoutError)) or (
                code is None and type(e).__module__.startswith(("httpx", "httpcore"))
            )
            if not retryable or attempt >= config.EMBEDDING_MAX_RETRIES:
                raise
            wait = min(60, 2 ** attempt)
            log.warning("Gemini embedding error (%s), retry %d/%d in %ss", code or type(e).__name__,
                        attempt + 1, config.EMBEDDING_MAX_RETRIES, wait)
            time.sleep(wait)


def _embed_adaptive(batch, task_type):
    try:
        return _embed_batch(batch, task_type)
    except Exception as e:  # noqa: BLE001
        if getattr(e, "code", None) == 400 and len(batch) > 1:  # e.g. too many tokens in one request
            mid = len(batch) // 2
            log.warning("Gemini rejected a batch of %d texts (400); splitting it", len(batch))
            return np.vstack([_embed_adaptive(batch[:mid], task_type), _embed_adaptive(batch[mid:], task_type)])
        raise


def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", titles=None):
    texts = [t if t.strip() else " " for t in texts]  # the API rejects empty strings
    if _is_v2():
        if task_type == "RETRIEVAL_QUERY":
            texts = [format_query(t) for t in texts]
        else:
            texts = [format_document(t, titles[i] if titles else None) for i, t in enumerate(texts)]
    out = []
    for i in range(0, len(texts), config.EMBEDDING_BATCH_SIZE):
        out.append(_embed_adaptive(texts[i:i + config.EMBEDDING_BATCH_SIZE], task_type))
    if not out:
        return np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)
    vecs = np.vstack(out)
    if vecs.shape[1] != config.EMBEDDING_DIM:
        raise RuntimeError(f"Gemini returned {vecs.shape[1]}-dim vectors but EMBEDDING_DIM={config.EMBEDDING_DIM}")
    return _normalise(vecs)


def embed_query(text):
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]


def warmup():
    """Embed one tiny text at startup: fails fast on a wrong key / model / dimension."""
    embed_texts(["warmup"])


def info():
    return {"provider": "gemini", "model": config.EMBEDDING_MODEL, "dim": config.EMBEDDING_DIM,
            "batch_size": config.EMBEDDING_BATCH_SIZE, "max_retries": config.EMBEDDING_MAX_RETRIES,
            "api_key_set": bool(config.GEMINI_API_KEY)}
