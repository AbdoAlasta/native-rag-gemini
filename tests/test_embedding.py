"""Unit tests for the Gemini embedding wrapper. No network: a fake client stands in for google-genai."""
import types

import numpy as np
import pytest


class FakeAPIError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


class FakeClient:
    def __init__(self, dim=768, fail_codes=(), aggregate=False, max_batch=None):
        self.dim, self.fail_codes, self.calls = dim, list(fail_codes), []
        self.aggregate, self.max_batch = aggregate, max_batch
        self.models = types.SimpleNamespace(embed_content=self._embed)

    def _embed(self, model, contents, config):
        texts = [c.parts[0].text if hasattr(c, "parts") else c for c in contents]
        self.calls.append({"model": model, "n": len(contents), "task": config.task_type,
                           "dim": config.output_dimensionality, "texts": texts,
                           "wrapped": all(hasattr(c, "parts") for c in contents)})
        if self.max_batch and len(contents) > self.max_batch:
            raise FakeAPIError(400)
        if self.fail_codes:
            raise FakeAPIError(self.fail_codes.pop(0))
        rng = np.random.default_rng(len(self.calls))
        n = 1 if self.aggregate else len(contents)  # gemini-embedding-2 merges a plain list into ONE vector
        vecs = rng.normal(size=(n, self.dim)) * 3  # deliberately NOT unit length
        return types.SimpleNamespace(embeddings=[types.SimpleNamespace(values=v.tolist()) for v in vecs])


@pytest.fixture()
def emb(monkeypatch):
    from app import config, embedding
    monkeypatch.setattr(config, "GEMINI_API_KEY", "x")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setattr(config, "EMBEDDING_DIM", 768)
    monkeypatch.setattr(config, "EMBEDDING_BATCH_SIZE", 100)
    monkeypatch.setattr(config, "EMBEDDING_MAX_RETRIES", 3)
    monkeypatch.setattr(embedding.time, "sleep", lambda s: None)
    return embedding


def use(monkeypatch, emb, client):
    monkeypatch.setattr(emb, "_get_client", lambda: client)
    return client


def test_batches_normalises_and_uses_document_task(monkeypatch, emb):
    c = use(monkeypatch, emb, FakeClient())
    out = emb.embed_texts([f"chunk {i}" for i in range(250)])
    assert [x["n"] for x in c.calls] == [100, 100, 50]
    assert {x["task"] for x in c.calls} == {"RETRIEVAL_DOCUMENT"}
    assert {x["dim"] for x in c.calls} == {768}
    assert out.shape == (250, 768)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_query_uses_query_task(monkeypatch, emb):
    c = use(monkeypatch, emb, FakeClient())
    v = emb.embed_query("what is x?")
    assert v.shape == (768,) and c.calls[0]["task"] == "RETRIEVAL_QUERY"


def test_empty_text_is_replaced_not_rejected(monkeypatch, emb):
    c = use(monkeypatch, emb, FakeClient())
    assert emb.embed_texts(["", "  ", "ok"]).shape == (3, 768) and c.calls[0]["n"] == 3


def test_retries_rate_limit_then_succeeds(monkeypatch, emb):
    c = use(monkeypatch, emb, FakeClient(fail_codes=[429, 503]))
    assert emb.embed_texts(["a"]).shape == (1, 768) and len(c.calls) == 3


def test_does_not_retry_bad_request(monkeypatch, emb):
    c = use(monkeypatch, emb, FakeClient(fail_codes=[400]))
    with pytest.raises(FakeAPIError):
        emb.embed_texts(["a"])
    assert len(c.calls) == 1


def test_gives_up_after_max_retries(monkeypatch, emb):
    c = use(monkeypatch, emb, FakeClient(fail_codes=[429] * 10))
    with pytest.raises(FakeAPIError):
        emb.embed_texts(["a"])
    assert len(c.calls) == 4  # first try + 3 retries


def test_dimension_mismatch_is_reported(monkeypatch, emb):
    use(monkeypatch, emb, FakeClient(dim=3072))
    with pytest.raises(RuntimeError, match="EMBEDDING_DIM"):
        emb.embed_texts(["a"])


def test_missing_key_is_reported(monkeypatch):
    from app import config, embedding
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(embedding, "_client", None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        embedding.embed_texts(["a"])


# ---------------- gemini-embedding-2 ----------------

@pytest.fixture()
def emb2(emb, monkeypatch):
    from app import config
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "gemini-embedding-2")
    monkeypatch.setattr(config, "EMBEDDING_BATCH_SIZE", 16)
    return emb


def test_v2_wraps_each_text_and_sends_no_task_type(monkeypatch, emb2):
    c = use(monkeypatch, emb2, FakeClient())
    out = emb2.embed_texts(["alpha", "beta", "gamma"])
    assert out.shape == (3, 768)
    assert c.calls[0]["wrapped"] and c.calls[0]["task"] is None
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_v2_document_and_query_prompt_formats(monkeypatch, emb2):
    c = use(monkeypatch, emb2, FakeClient())
    emb2.embed_texts(["some chunk"])
    emb2.embed_texts(["a question"], "RETRIEVAL_QUERY")
    assert c.calls[0]["texts"] == ["title: none | text: some chunk"]
    assert c.calls[1]["texts"] == ["task: search result | query: a question"]


def test_v2_guard_against_single_aggregated_vector(monkeypatch, emb2):
    use(monkeypatch, emb2, FakeClient(aggregate=True))
    with pytest.raises(RuntimeError, match="embeddings for"):
        emb2.embed_texts(["a", "b", "c"])


def test_v2_splits_a_rejected_batch(monkeypatch, emb2):
    c = use(monkeypatch, emb2, FakeClient(max_batch=2))
    out = emb2.embed_texts([f"t{i}" for i in range(5)])
    assert out.shape == (5, 768)
    assert max(x["n"] for x in c.calls if x["n"] <= 2) == 2  # eventually sent in pieces of <= 2


def test_v2_document_title_is_the_source_name(monkeypatch, emb2):
    c = use(monkeypatch, emb2, FakeClient())
    emb2.embed_texts(["chunk one", "chunk two"], titles=["lecture1.pdf", "lecture1.pdf"])
    assert c.calls[0]["texts"][0] == "title: lecture1.pdf | text: chunk one"


def test_embed_query_uses_query_prompt_for_v2(monkeypatch, emb2):
    c = use(monkeypatch, emb2, FakeClient())
    assert emb2.embed_query("what is x?").shape == (768,)
    assert c.calls[0]["texts"] == ["task: search result | query: what is x?"]
