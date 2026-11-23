"""Resumable long computations: a tiny key-value checkpoint store.

Usage:
    ck = Checkpoint("runs/spin_h1.json")          # loads existing progress if present
    for n in nodes:
        if ck.has(n):  value = ck.get_mpf(n)
        else:          value = expensive(n); ck.set(n, value)   # auto-saves every `every` seconds
    ck.save()

Values are stored as strings (mpmath numbers at full working precision, ints exactly), written
atomically (temp file + rename), so an interrupted run never leaves a corrupt file.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from mpmath import mp, mpf


class Checkpoint:
    def __init__(self, path, every: float = 30.0, meta: dict | None = None):
        self.path = Path(path)
        self.every = every
        self._last = time.monotonic()
        self.data: dict[str, str] = {}
        self.meta = meta or {}
        if self.path.exists():
            blob = json.loads(self.path.read_text())
            self.data = blob.get("data", {})
            old_meta = blob.get("meta", {})
            if meta and old_meta and {k: old_meta.get(k) for k in meta} != meta:
                raise ValueError(f"checkpoint {self.path} was made with different parameters: {old_meta} vs {meta}")
            self.meta = old_meta or self.meta

    @staticmethod
    def _key(k) -> str:
        return k if isinstance(k, str) else repr(k)

    def has(self, key) -> bool:
        return self._key(key) in self.data

    def get(self, key, default=None):
        return self.data.get(self._key(key), default)

    def get_mpf(self, key):
        return mpf(self.data[self._key(key)])

    def get_int(self, key) -> int:
        return int(self.data[self._key(key)])

    def set(self, key, value):
        if isinstance(value, int):
            s = str(value)
        else:
            s = mp.nstr(value, mp.dps + 10) if hasattr(value, "real") else str(value)
        self.data[self._key(key)] = s
        if time.monotonic() - self._last > self.every:
            self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"meta": self.meta, "data": self.data}))
        tmp.replace(self.path)
        self._last = time.monotonic()

    def __len__(self):
        return len(self.data)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.save()


def cached(fn, ck: Checkpoint | None, prefix: str = ""):
    """Wrap fn(n) so results are read from / written to the checkpoint."""
    if ck is None:
        return fn

    def wrapped(n):
        key = f"{prefix}{n}"
        if ck.has(key):
            return ck.get_mpf(key)
        v = fn(n)
        ck.set(key, v)
        return v

    return wrapped
