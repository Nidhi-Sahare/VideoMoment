from pathlib import Path
from sqlalchemy import select
from database import SessionLocal
from models import Video
from deployment_config import SAMPLE_VIDEO_IDS

def get_sample_video_id(video_path:str|Path)->str:
    requested_path=Path(video_path).resolve()

    if not requested_path.is_file():
        raise FileNotFoundError(f"Video not found: {requested_path}")

    with SessionLocal() as db:
        videos=db.execute(
            select(Video).where(Video.id.in_(SAMPLE_VIDEO_IDS))
        ).scalars().all()

        for video in videos:
            registered_path=Path(video.file_path).resolve()
            if registered_path==requested_path:
                return video.id

    raise PermissionError(
        "This video is not registered as a sample video.")