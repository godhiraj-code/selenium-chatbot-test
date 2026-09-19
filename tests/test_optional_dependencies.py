import subprocess
import sys

import pytest

from selenium_chatbot_test.assertions import _ModelLoader


def test_core_import_succeeds_when_semantic_packages_are_blocked():
    code = """
import sys

class BlockSemanticPackages:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numpy' or fullname.startswith('sentence_transformers'):
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockSemanticPackages())
from selenium_chatbot_test import LatencyMonitor, SemanticAssert, StreamWaiter
assert LatencyMonitor and SemanticAssert and StreamWaiter
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


def test_semantic_use_without_extra_has_actionable_install_hint(monkeypatch):
    original_import = __import__

    def reject_sentence_transformers(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", reject_sentence_transformers)

    with pytest.raises(ImportError, match=r"selenium-chatbot-test\[semantic\]"):
        _ModelLoader()._load_model("test-model")
