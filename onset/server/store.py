"""On-disk session store.  One directory per analysis; nothing leaves the machine.

* ``src/``  the footage as received (uploaded file, or captured frames).  Deleted as soon as the analysis ends unless
            ``keep_source`` is set: the report only needs the grayscale face plates in ``out/``.
* ``out/``  result.json and the plate images.
* ``meta.json``  who/what/when, consent time stamp and expiry.

Sessions expire after ``retention_hours`` and are purged on start-up and whenever a request arrives; ``purge_all``
removes everything.  A session that never reaches "done" (a recording nobody finished, an analysis cut short by a
crash) holds the most sensitive thing the store can hold - the footage itself - so those go much sooner: after
``abandon_minutes`` while the server runs, and at once when it starts (nothing can resume them).
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
import time
from typing import Optional

ID_RE = re.compile(r"^[a-z]-[0-9a-f]{10}$")


class Store:
    def __init__(self, root: str, retention_hours: float = 24.0, abandon_minutes: float = 60.0):
        self.root = os.path.abspath(root)
        self.sessions = os.path.join(self.root, "sessions")
        self.retention = float(retention_hours)
        self.abandon = float(abandon_minutes)
        os.makedirs(self.sessions, exist_ok=True)
        self._lock = threading.Lock()
        self._last_sweep = 0.0

    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def valid_id(sid: str) -> bool:
        return bool(ID_RE.match(sid or ""))

    def dir(self, sid: str) -> str:
        if not self.valid_id(sid):
            raise KeyError("bad id")
        return os.path.join(self.sessions, sid)

    def src_dir(self, sid: str) -> str:
        return os.path.join(self.dir(sid), "src")

    def out_dir(self, sid: str) -> str:
        return os.path.join(self.dir(sid), "out")

    def exists(self, sid: str) -> bool:
        return self.valid_id(sid) and os.path.isdir(self.dir(sid))

    # ------------------------------------------------------------------------------------------------
    def create(self, kind: str, label: str = "", provenance: Optional[dict] = None, consent: bool = False) -> str:
        sid = f"{kind[0]}-{secrets.token_hex(5)}"
        d = self.dir(sid)
        os.makedirs(os.path.join(d, "src"))
        os.makedirs(os.path.join(d, "out"))
        now = time.time()
        self.write_meta(sid, dict(id=sid, kind=kind, label=label[:80], created=now, expires=now + self.retention * 3600,
                                  consent_at=now if consent else None, provenance=provenance or {}, state="created"))
        return sid

    def meta(self, sid: str) -> dict:
        with open(os.path.join(self.dir(sid), "meta.json")) as fh:
            return json.load(fh)

    def write_meta(self, sid: str, meta: dict):
        p = os.path.join(self.dir(sid), "meta.json")
        with self._lock:
            with open(p + ".tmp", "w") as fh:
                json.dump(meta, fh)
            os.replace(p + ".tmp", p)

    def update_meta(self, sid: str, **kw) -> dict:
        m = self.meta(sid)
        m.update(kw)
        self.write_meta(sid, m)
        return m

    def drop_source(self, sid: str):
        shutil.rmtree(self.src_dir(sid), ignore_errors=True)
        os.makedirs(self.src_dir(sid), exist_ok=True)

    # ------------------------------------------------------------------------------------------------
    def list(self, limit: int = 30) -> list[dict]:
        out = []
        for sid in os.listdir(self.sessions):
            if not self.valid_id(sid):
                continue
            try:
                m = self.meta(sid)
            except Exception:
                continue
            if m.get("state") == "done":
                out.append(m)
        out.sort(key=lambda m: -m["created"])
        return out[:limit]

    def delete(self, sid: str) -> bool:
        if not self.exists(sid):
            return False
        shutil.rmtree(self.dir(sid), ignore_errors=True)
        return True

    def purge_all(self) -> int:
        n = 0
        for sid in os.listdir(self.sessions):
            if self.valid_id(sid):
                shutil.rmtree(os.path.join(self.sessions, sid), ignore_errors=True)
                n += 1
        return n

    def _sweep(self, doomed) -> int:
        """Remove every session for which ``doomed(meta_or_None, age_seconds)`` is true.  A folder with no readable
        meta is removed only once it is a minute old, so a session that is still being created is never caught."""
        now = time.time()
        n = 0
        for sid in os.listdir(self.sessions):
            if not self.valid_id(sid):
                continue
            path = os.path.join(self.sessions, sid)
            try:
                m = self.meta(sid)
            except Exception:
                m = None
            try:
                age = now - (m["created"] if m else os.path.getmtime(path))
            except (KeyError, OSError):
                age = 1e9
            if (m is None and age < 60) or not doomed(m, age):
                continue
            shutil.rmtree(path, ignore_errors=True)
            n += 1
        return n

    def purge_expired(self, force: bool = False) -> int:
        """Sessions past their expiry, plus unfinished ones older than ``abandon_minutes``.  Cheap: sweeps at most every 5 minutes."""
        now = time.time()
        if not force and now - self._last_sweep < 300:
            return 0
        self._last_sweep = now
        return self._sweep(lambda m, age: m is None or m.get("expires", 0) < now
                           or (m.get("state") != "done" and age >= self.abandon * 60))

    def purge_unfinished(self) -> int:
        """Everything that never reached "done".  Called once when the server starts: no job survives a restart."""
        return self._sweep(lambda m, age: m is None or m.get("state") != "done")
