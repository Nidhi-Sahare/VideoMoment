
import os
from fastapi import HTTPException
from sample_videos import SAMPLE_VIDEOS


def env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc

    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


DEPLOY_MODE = os.getenv("DEPLOY_MODE", "local").strip().lower()

if DEPLOY_MODE not in {"local", "demo", "public"}:
    raise ValueError(
        "DEPLOY_MODE must be 'local', 'demo', or 'public'."
    )

IS_DEMO = DEPLOY_MODE == "demo"
IS_PUBLIC = DEPLOY_MODE == "public"
IS_LOCAL = DEPLOY_MODE == "local"

SAMPLE_VIDEO_IDS = frozenset(
    sample["video_id"] for sample in SAMPLE_VIDEOS
)

FRONTEND_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500",
    ).split(",")
    if origin.strip()
]

SESSION_EXPIRY_SECONDS = env_int(
    "SESSION_EXPIRY_SECONDS", 7200
)

PUBLIC_MAX_UPLOAD_MB = env_int(
    "PUBLIC_MAX_UPLOAD_MB", 250
)

PUBLIC_MAX_VIDEO_SECONDS = env_int(
    "PUBLIC_MAX_VIDEO_SECONDS", 300
)

PUBLIC_MAX_CONCURRENT_JOBS = env_int(
    "PUBLIC_MAX_CONCURRENT_JOBS", 1
)

PUBLIC_ACCESS_CODE = os.getenv(
    "PUBLIC_ACCESS_CODE", ""
).strip()


def require_local_mode():
    """
    Preserve the existing demo restriction.

    Public-mode upload permissions must be enforced
    separately at the API endpoints.
    """
    if IS_DEMO:
        raise HTTPException(
            status_code=403,
            detail="This feature is unavailable in the public demo.",
        )


def require_demo_video_access(video_id: str):
    if IS_DEMO and video_id not in SAMPLE_VIDEO_IDS:
        raise HTTPException(
            status_code=404,
            detail="Video not available in the public demo.",
        )
