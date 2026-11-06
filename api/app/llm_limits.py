"""Rate limits for LLM features, per API key (the key's database id), with a per-minute and a per-hour allowance.

A request is counted before it is processed, so refused and rejected requests use the allowance too: probing the safety checks is
not free. In memory and per process, like the other limiters here; move to Redis or the database before running several instances."""
from .config import get_settings
from .ratelimit import SlidingWindow

_s = get_settings()
per_minute = SlidingWindow(_s.llm_rate_limit_per_min, 60)
per_hour = SlidingWindow(_s.llm_rate_limit_per_hour, 3600)


class RateLimited(Exception):
    def __init__(self, scope: str, retry_after: int):
        self.scope, self.retry_after = scope, retry_after
        super().__init__(f"Too many requests this {scope}.")


def acquire(key_id: int) -> dict:
    """Count one request for this key, or raise RateLimited. Returns the remaining allowances."""
    k = str(key_id)
    if per_hour.over(k):
        raise RateLimited("hour", per_hour.retry_after(k))
    if not per_minute.hit(k):
        raise RateLimited("minute", per_minute.retry_after(k))
    if not per_hour.hit(k):
        raise RateLimited("hour", per_hour.retry_after(k))
    return {"minute": per_minute.remaining(k), "hour": per_hour.remaining(k)}
