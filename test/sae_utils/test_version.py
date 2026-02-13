import importlib.metadata
from importlib.metadata import PackageNotFoundError
from typing import NoReturn

import pytest


def test_version_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda pkg: "1.2.3")
    # Reload the module to apply monkeypatch
    mod = importlib.reload(importlib.import_module("sae_utils._version"))
    assert mod.__version__ == "1.2.3"


def test_version_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def version(_pkg: str) -> NoReturn:
        raise PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", version)
    mod = importlib.reload(importlib.import_module("sae_utils._version"))
    assert mod.__version__ == "0.0.0"
