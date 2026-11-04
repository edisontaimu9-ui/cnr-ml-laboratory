"""Shared test environment. Must run before `app` is imported anywhere."""
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATASET_STORAGE_DIR", f"{_tmp}/datasets")
os.environ.setdefault("PSEUDONYM_KEY", "p" * 40)
