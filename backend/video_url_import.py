import ipaddress
import logging
import shutil
import socket
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4
import yt_dlp
from fastapi import APIRouter,HTTPException
from pydantic import BaseModel,Field
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool
from database import SessionLocal
from models import Video

logger=logging.getLogger(__name__)
router=APIRouter(prefix="/videos",tags=["Video URL Import"])
BASE_DIR=Path(__file__).resolve().parent
PROJECT_ROOT=BASE_DIR.parent
UPLOAD_DIR=PROJECT_ROOT/"uploads"
UPLOAD_DIR.mkdir(parents=True,exist_ok=True)
MAX_VIDEO_BYTES=2*1024*1024*1024
MAX_DURATION_SECONDS=2*60*60
ALLOWED_HOSTS={
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com"}

class VideoURLRequest(BaseModel):
    url:str=Field(min_length=10,max_length=2048)

class VideoURLResponse(BaseModel):
    video_id:str
    filename:str
    status:str
    uploaded_at:str
    title:str
    duration:float|None=None
    message:str

class DownloadRejected(Exception):
    pass

def validate_youtube_url(raw_url:str)->str:
    url=raw_url.strip()
    try:
        parsed=urlsplit(url)
    except ValueError:
        raise DownloadRejected("Invalid video URL.")
    if parsed.scheme!="https":
        raise DownloadRejected("Only HTTPS video URLs are supported.")
    if parsed.username or parsed.password:
        raise DownloadRejected("URLs containing credentials are not supported.")
    try:
        hostname=parsed.hostname
        port=parsed.port
    except ValueError:
        raise DownloadRejected("Invalid URL hostname or port.")
    if hostname not in ALLOWED_HOSTS:
        raise DownloadRejected(
            "Only public YouTube links are supported "
            "in this first version.")
    if port not in (None,443):
        raise DownloadRejected("Custom URL ports are not supported.")
    if not parsed.path or parsed.path=="/":
        raise DownloadRejected("Please provide a YouTube video link.")
    return url

def check_duration(info:dict)->float|None:
    duration=info.get("duration")
    if duration is None:
        raise DownloadRejected("The video duration could not be verified.")
    try:
        duration=float(duration)
    except (TypeError,ValueError):
        raise DownloadRejected("The video has an invalid duration.")
    if duration<=0:
        raise DownloadRejected("The video duration is invalid.")
    if duration>MAX_DURATION_SECONDS:
        raise DownloadRejected("Videos longer than 2 hours are not supported.")
    return duration

def check_size(info:dict):
    size=info.get("filesize") or info.get("filesize_approx")
    if size is not None and size>MAX_VIDEO_BYTES:
        raise DownloadRejected("The video exceeds the 2 GB limit.")

def safe_title(value:str)->str:
    title=str(value or "YouTube Video")
    title="".join(
        character for character in title
        if character.isprintable()
        and character not in '\\/:*?"<>|')
    title=" ".join(title.split()).strip(" .")
    return title[:200] or "YouTube Video"

def download_video(url:str):
    url=validate_youtube_url(url)
    video_id=uuid4().hex
    temp_dir=Path(tempfile.mkdtemp(
        prefix="videomoment_url_",
        dir=UPLOAD_DIR))
    destination=UPLOAD_DIR/f"{video_id}.mp4"

    try:
        def progress_hook(data):
            if data.get("status")!="downloading":
                return
            downloaded=data.get("downloaded_bytes",0)
            total=data.get("total_bytes") or data.get("total_bytes_estimate")
            if downloaded>MAX_VIDEO_BYTES:
                raise DownloadRejected("Download exceeded the 2 GB limit.")
            if total and total>MAX_VIDEO_BYTES:
                raise DownloadRejected("Video exceeds the 2 GB limit.")

        common_options={
            "quiet":True,
            "no_warnings":True,
            "noplaylist":True,
            "ignoreerrors":False,
            "socket_timeout":20,
            "retries":2,
            "fragment_retries":2,
            "extract_flat":False,
            "skip_download":True,
            "ignoreconfig":True,
            "cachedir":False}

        with yt_dlp.YoutubeDL(common_options) as ydl:
            info=ydl.extract_info(url,download=False)

        if not isinstance(info,dict):
            raise DownloadRejected("Unable to retrieve video information.")

        if info.get("_type") in {"playlist","multi_video"}:
            raise DownloadRejected(
                "Please provide a single video URL, "
                "not a playlist.")

        if info.get("is_live") or info.get("live_status") in {"is_live","is_upcoming"}:
            raise DownloadRejected("Live streams are not supported.")

        duration=check_duration(info)
        check_size(info)
        title=safe_title(info.get("title"))

        download_options={
            **common_options,
            "skip_download":False,
            "format":(
                "bestvideo[ext=mp4]+bestaudio[ext=m4a]"
                "/best[ext=mp4]"),
            "merge_output_format":"mp4",
            "outtmpl":str(temp_dir/"video.%(ext)s"),
            "progress_hooks":[progress_hook],
            "overwrites":False,
            "restrictfilenames":True,
            "max_filesize":MAX_VIDEO_BYTES}

        with yt_dlp.YoutubeDL(download_options) as ydl:
            ydl.download([url])

        candidates=list(temp_dir.glob("*.mp4"))
        if not candidates:
            raise DownloadRejected("No MP4 video was produced.")

        downloaded_file=max(
            candidates,
            key=lambda path:path.stat().st_size)
        size=downloaded_file.stat().st_size

        if size==0:
            raise DownloadRejected("The downloaded video is empty.")
        if size>MAX_VIDEO_BYTES:
            raise DownloadRejected("Video exceeds the 2 GB limit.")

        downloaded_file.replace(destination)
        filename=f"{title}.mp4"

        with SessionLocal() as db:
            video=Video(
                id=video_id,
                filename=filename,
                file_path=str(destination),
                file_size=size,
                duration=duration,
                status="uploaded",
                indexed=False,
                progress=0,
                stage="uploaded",
                message=(
                    "Online video imported. "
                    "Ready for indexing."))
            db.add(video)
            db.commit()
            db.refresh(video)
            uploaded_at=video.uploaded_at

        return {
            "video_id":video_id,
            "filename":filename,
            "status":"uploaded",
            "uploaded_at":uploaded_at.isoformat(),
            "title":title,
            "duration":duration,
            "message":(
                "Video imported successfully. "
                "Start indexing to enable search.")}

    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(temp_dir,ignore_errors=True)

@router.post(
    "/import-url",
    response_model=VideoURLResponse,
    status_code=201)
async def import_video_url(request:VideoURLRequest):
    try:
        validate_youtube_url(request.url)
        result=await run_in_threadpool(
            download_video,
            request.url)
        return VideoURLResponse(**result)

    except DownloadRejected as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)) from error

    except yt_dlp.utils.DownloadError as error:
        logger.warning("Online video download failed: %s",error)
        raise HTTPException(
            status_code=422,
            detail=(
                "Unable to download this video. "
                "It may be private, restricted, "
                "unavailable, or unsupported.")) from error

    except SQLAlchemyError as error:
        logger.exception("Database registration failed.")
        raise HTTPException(
            status_code=500,
            detail=(
                "Video downloaded, but database "
                "registration failed.")) from error

    except Exception as error:
        logger.exception("Unexpected URL import failure.")
        raise HTTPException(
            status_code=500,
            detail=(
                "Video import failed. "
                "Check the backend logs.")) from error