"""Answer-model calls: retries, timeouts and empty replies (Gemini replaced by a stand-in, no network)."""
from types import SimpleNamespace

import pytest

from app import config, llm, service


class Err(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def generate_content(self, **kw):
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(text=out, usage_metadata=None, prompt_feedback=None, candidates=[])


@pytest.fixture()
def fake(monkeypatch):
    def install(*outcomes):
        models = FakeModels(outcomes)
        monkeypatch.setattr(llm, "_client", SimpleNamespace(models=models))
        return models
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 2)
    return install


def test_rate_limit_is_retried_then_answers(fake):
    models = fake(Err(429), Err(503), "ATP [1].")
    assert llm.generate("q") == "ATP [1]." and models.calls == 3


def test_gives_up_after_the_retries(fake):
    models = fake(Err(429), Err(429), Err(429))
    with pytest.raises(llm.ModelUnavailable, match="429"):
        llm.generate("q")
    assert models.calls == 3


def test_bad_request_and_timeout_are_not_retried(fake):
    models = fake(Err(400))
    with pytest.raises(llm.ModelUnavailable):
        llm.generate("q")
    assert models.calls == 1
    ReadTimeout = type("ReadTimeout", (Exception,), {})
    models = fake(ReadTimeout("slow"))
    with pytest.raises(llm.ModelUnavailable, match="ReadTimeout"):
        llm.generate("q")
    assert models.calls == 1


def test_empty_or_blocked_reply_is_unavailable(fake):
    fake("")
    with pytest.raises(llm.ModelUnavailable, match="no text"):
        llm.generate("q")


def test_drop_near_duplicates_keeps_the_best_copy():
    rows = [("a.pdf", "d1", 1, "Quality assurance checks the process and the product.", 0.8),
            ("b.pdf", "d2", 9, "Quality assurance checks the process and the product!", 0.7),
            ("c.pdf", "d3", 2, "Unit testing checks one module.", 0.6)]
    assert [r[0] for r in service.drop_near_duplicates(rows)] == ["a.pdf", "c.pdf"]
