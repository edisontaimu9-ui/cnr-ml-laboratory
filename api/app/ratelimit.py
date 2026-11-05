"""Small in-memory sliding-window limiter. State lives in this process only, which matches the
single-instance deployment. If the API is ever scaled to several instances, move this to Redis/DB."""
import threading
import time
from collections import defaultdict, deque


class SlidingWindow:
    def __init__(self, limit: int, window_s: float):
        self.limit, self.window = limit, window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _trim(self, q: deque, now: float) -> None:
        while q and now - q[0] >= self.window:
            q.popleft()

    def over(self, key: str) -> bool:
        """True if `key` already used up its allowance (does not count a hit)."""
        with self._lock:
            q = self._hits.get(key)
            if not q:
                return False
            self._trim(q, time.monotonic())
            return len(q) >= self.limit

    def hit(self, key: str) -> bool:
        """Count one hit. Returns False (and does not count) when the allowance is used up."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            self._trim(q, now)
            if len(q) >= self.limit:
                return False
            q.append(now)
            if len(self._hits) > 5000:  # bound memory: drop idle keys
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
                    del self._hits[k]
            return True

    def retry_after(self, key: str) -> int:
        with self._lock:
            q = self._hits.get(key)
            return max(1, int(self.window - (time.monotonic() - q[0])) + 1) if q else 1
