from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit_per_minute: int = 60) -> None:
        super().__init__(app)
        self.limit = limit_per_minute
        self.history: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        now = time.time()
        key = f"{request.client.host if request.client else 'local'}::{request.url.path}"
        entries = self.history[key]
        while entries and now - entries[0] > 60:
            entries.popleft()
        if len(entries) >= self.limit:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
        entries.append(now)
        return await call_next(request)

