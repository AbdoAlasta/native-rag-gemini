"""Startup refuses to run without a database URL instead of falling back to built-in credentials."""
import pytest

from app import config, db


def test_missing_database_url_is_a_clear_startup_error(monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", "")
    with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
        db.init()
