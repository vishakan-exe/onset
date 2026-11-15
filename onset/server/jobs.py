"""Analysis jobs: one worker thread (the analysis is CPU-bound; two at once would only slow both), progress and errors
that say what the person can do about them."""
from __future__ import annotations

import secrets
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from ..analysis.pipeline import AnalysisError


@dataclass
class Job:
    id: str
    kind: str
    state: str = "queued"          # queued | running | done | error
    progress: float = 0.0
    stage: str = "Waiting for the analyser"
    result_id: Optional[str] = None
    error: Optional[str] = None
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    queue_position: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["elapsed_s"] = round((self.finished or time.time()) - (self.started or self.created), 1)
        return d


class JobManager:
    def __init__(self, workers: int = 1, max_queue: int = 6):
        self.pool = ThreadPoolExecutor(workers)
        self.jobs: dict[str, Job] = {}
        self.max_queue = max_queue
        self._lock = threading.Lock()

    def busy(self) -> int:
        with self._lock:
            return sum(1 for j in self.jobs.values() if j.state in ("queued", "running"))

    def submit(self, kind: str, fn: Callable[[Callable[[float, str], None]], str]) -> Job:
        if self.busy() >= self.max_queue:
            raise RuntimeError("busy")
        job = Job(id="j-" + secrets.token_hex(5), kind=kind)
        with self._lock:
            self.jobs[job.id] = job
            # keep the table small
            if len(self.jobs) > 60:
                for k in sorted(self.jobs, key=lambda k: self.jobs[k].created)[:20]:
                    if self.jobs[k].state in ("done", "error"):
                        del self.jobs[k]

        def progress(frac: float, stage: str = ""):
            job.progress = float(min(max(frac, 0.0), 1.0))
            if stage:
                job.stage = stage

        def run():
            job.state, job.started = "running", time.time()
            try:
                job.result_id = fn(progress)
                job.state, job.progress, job.stage = "done", 1.0, "Done"
            except AnalysisError as e:
                job.state, job.error = "error", str(e)
            except Exception as e:                                           # noqa: BLE001
                traceback.print_exc()
                job.state = "error"
                job.error = ("The analysis stopped unexpectedly. Try the clip again; if it keeps failing, try a shorter clip "
                             f"or another file. ({type(e).__name__})")
            finally:
                job.finished = time.time()

        self.pool.submit(run)
        return job

    def get(self, jid: str) -> Optional[Job]:
        j = self.jobs.get(jid)
        if j and j.state == "queued":
            order = sorted((x for x in self.jobs.values() if x.state == "queued"), key=lambda x: x.created)
            j.queue_position = order.index(j) + 1 if j in order else 0
        return j
