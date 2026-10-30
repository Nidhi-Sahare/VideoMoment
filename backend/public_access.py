
"""
VideoMoment public processing safeguards.

A new upload or URL import consumes one workflow allowance.
Indexing an already-owned video does not consume another.

Important:
- Workflow quotas are process-local.
- Deploy with one API worker until shared quotas exist.
- Index routes must pass action="index".
- Index routes must separately verify video ownership.
"""

import hmac
import logging
import os
import threading
import time

from collections import defaultdict, deque
from contextlib import contextmanager

from fastapi import HTTPException, Request

from deployment_config import (
    IS_PUBLIC,
    PUBLIC_ACCESS_CODE,
    PUBLIC_MAX_CONCURRENT_JOBS,
)

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_request_history = defaultdict(deque)

_processing_slots = threading.BoundedSemaphore(
    max(1, PUBLIC_MAX_CONCURRENT_JOBS)
)

RATE_LIMIT_WINDOW_SECONDS = 3600


def _read_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))

    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer."
        ) from exc

    if value < 0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return value


ANONYMOUS_PROCESSING_ENABLED = (
    os.getenv(
        "ANONYMOUS_PROCESSING_ENABLED",
        "false",
    ).strip().lower() == "true"
)

ANONYMOUS_REQUESTS_PER_HOUR = _read_positive_int(
    "ANONYMOUS_REQUESTS_PER_HOUR",
    1,
)

AUTHORIZED_REQUESTS_PER_HOUR = _read_positive_int(
    "AUTHORIZED_REQUESTS_PER_HOUR",
    5,
)


def _client_key(request: Request) -> str:
    if request.client is None:
        return "unknown"

    return request.client.host or "unknown"


def _valid_access_code(request: Request) -> bool:
    if not PUBLIC_ACCESS_CODE:
        return False

    provided = request.headers.get(
        "X-Access-Code",
        "",
    )

    return hmac.compare_digest(
        provided.encode("utf-8"),
        PUBLIC_ACCESS_CODE.encode("utf-8"),
    )


def _access_policy(request: Request):
    if _valid_access_code(request):
        return (
            "authorized",
            AUTHORIZED_REQUESTS_PER_HOUR,
        )

    if not ANONYMOUS_PROCESSING_ENABLED:
        raise HTTPException(
            status_code=403,
            detail=(
                "An access code is required to "
                "upload or process videos."
            ),
        )

    return (
        "anonymous",
        ANONYMOUS_REQUESTS_PER_HOUR,
    )


def require_processing_access(
    request: Request,
    action: str = "new_video",
) -> str:
    """
    Check processing permissions.

    Actions:
        new_video: upload or URL import; consumes quota.
        index: index an existing video; does not consume quota.

    This function does not establish ownership.
    Call require_video_access() before allowing indexing.
    """
    if not IS_PUBLIC:
        return "local"

    if action not in {"new_video", "index"}:
        raise ValueError(
            f"Unsupported processing action: {action}"
        )

    access_type, limit = _access_policy(request)

    if limit == 0:
        raise HTTPException(
            status_code=403,
            detail="Video processing is currently disabled.",
        )

    if action == "index":
        return access_type

    client = _client_key(request)
    key = (client, access_type)
    now = time.monotonic()

    with _lock:
        history = _request_history[key]

        while (
            history
            and now - history[0]
            >= RATE_LIMIT_WINDOW_SECONDS
        ):
            history.popleft()

        if len(history) >= limit:
            remaining = max(
                1,
                int(
                    RATE_LIMIT_WINDOW_SECONDS
                    - (now - history[0])
                ),
            )

            raise HTTPException(
                status_code=429,
                detail=(
                    "New video limit reached. "
                    "Please try again later."
                ),
                headers={
                    "Retry-After": str(remaining)
                },
            )

        history.append(now)

        # Remove expired, empty histories occasionally.
        if len(_request_history) > 10000:
            for old_key in list(_request_history):
                if not _request_history[old_key]:
                    del _request_history[old_key]

    return access_type


@contextmanager
def processing_slot():
    """
    Limit concurrent expensive processing jobs.

    Must be used by the indexing worker.
    """
    if not IS_PUBLIC:
        yield
        return

    acquired = _processing_slots.acquire(
        blocking=False
    )

    if not acquired:
        raise HTTPException(
            status_code=429,
            detail=(
                "Video processing is busy. "
                "Please try again later."
            ),
            headers={"Retry-After": "60"},
        )

    try:
        yield
    finally:
        _processing_slots.release()
