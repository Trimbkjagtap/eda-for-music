"""
Rate limiter with exponential backoff for API calls.
Handles 429 Too Many Requests errors gracefully.
"""
import time
import threading
from functools import wraps
from loguru import logger


class RateLimiter:
    """Token-bucket rate limiter."""

    def __init__(self, calls_per_second: int = 10):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()


def with_retry(max_retries: int = 5, base_delay: float = 1.0):
    """Decorator: retry on exceptions with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    msg = str(e).lower()
                    # Rate limit
                    if "429" in msg or "rate" in msg:
                        wait = delay * (2 ** attempt)
                        logger.warning(f"Rate limited. Waiting {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                        time.sleep(wait)
                    # Retryable server errors
                    elif "500" in msg or "503" in msg or "timeout" in msg:
                        wait = delay * (2 ** attempt)
                        logger.warning(f"Server error: {e}. Retrying in {wait:.1f}s")
                        time.sleep(wait)
                    else:
                        raise
            raise RuntimeError(f"{func.__name__} failed after {max_retries} retries")
        return wrapper
    return decorator


# Shared global limiter
default_limiter = RateLimiter(calls_per_second=10)
