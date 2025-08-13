import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import importlib
import logging
import os
import config


def test_env_helpers(monkeypatch):
    monkeypatch.delenv("TEST_BOOL", raising=False)
    assert config._env_bool("TEST_BOOL", True) is True
    monkeypatch.setenv("TEST_BOOL", "off")
    assert config._env_bool("TEST_BOOL", True) is False

    monkeypatch.setenv("TEST_INT", "not-int")
    assert config._env_int("TEST_INT", 5) == 5


def test_logger_handler_added(monkeypatch):
    monkeypatch.setattr(logging.Logger, "hasHandlers", lambda self: False)
    importlib.reload(config)
    assert config.logger.handlers
