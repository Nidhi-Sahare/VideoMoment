import os
from sample_videos import SAMPLE_VIDEOS

DEPLOY_MODE=os.getenv("DEPLOY_MODE","local").strip().lower()

if DEPLOY_MODE not in {"local","demo"}:
    raise ValueError("DEPLOY_MODE must be 'local' or 'demo'.")

IS_DEMO=DEPLOY_MODE=="demo"
SAMPLE_VIDEO_IDS=frozenset(sample["video_id"] for sample in SAMPLE_VIDEOS)

FRONTEND_ORIGINS=[
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500"
    ).split(",")
    if origin.strip()]

def require_local_mode():
    if IS_DEMO:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=403,
            detail="This feature is unavailable in the public demo.")

def require_demo_video_access(video_id:str):
    if IS_DEMO and video_id not in SAMPLE_VIDEO_IDS:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail="Video not available in the public demo.")