
"""
VideoMoment YouTube video import.

Supports local and public modes, with:
- YouTube HTTPS URL validation
- Visitor authentication in public mode
- Atomic ownership assignment when saving video records
- Download size and duration limits
- FFprobe validation of downloaded files
- Temporary-file cleanup
- Per-process import concurrency limits

Public deployment must use one API worker until shared
admission controls and network egress restrictions exist.
"""

import logging
import math
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import yt_dlp

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from database import SessionLocal
from models import Video
from deployment_config import (
    IS_PUBLIC,
    PUBLIC_MAX_UPLOAD_MB,
    PUBLIC_MAX_VIDEO_SECONDS,
    PUBLIC_MAX_CONCURRENT_JOBS,
)
from public_access import require_processing_access
from visitor_auth import get_current_visitor_id


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/videos",
    tags=["Video URL Import"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024
LOCAL_MAX_DURATION_SECONDS = 7200

MAX_VIDEO_BYTES = (
    PUBLIC_MAX_UPLOAD_MB * 1024 * 1024
    if IS_PUBLIC
    else LOCAL_MAX_VIDEO_BYTES
)

MAX_DURATION_SECONDS = (
    PUBLIC_MAX_VIDEO_SECONDS
    if IS_PUBLIC
    else LOCAL_MAX_DURATION_SECONDS
)

_import_slots = threading.BoundedSemaphore(
    max(1, PUBLIC_MAX_CONCURRENT_JOBS)
)

ALLOWED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}


class VideoURLRequest(BaseModel):
    url: str = Field(min_length=10, max_length=2048)


class VideoURLResponse(BaseModel):
    video_id: str
    filename: str
    status: str
    uploaded_at: str
    title: str
    duration: float | None = None
    message: str


class DownloadRejected(Exception):
    pass


def validate_youtube_url(raw_url: str) -> str:
    url = raw_url.strip()

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise DownloadRejected(
            "Invalid video URL."
        ) from exc

    if parsed.scheme != "https":
        raise DownloadRejected(
            "Only HTTPS URLs are supported."
        )

    if parsed.username or parsed.password:
        raise DownloadRejected(
            "URL credentials are not supported."
        )

    if hostname not in ALLOWED_HOSTS:
        raise DownloadRejected(
            "Only public YouTube links are supported."
        )

    if port not in (None, 443):
        raise DownloadRejected(
            "Custom URL ports are not supported."
        )

    if not parsed.path or parsed.path == "/":
        raise DownloadRejected(
            "Please provide a YouTube video link."
        )

    return url


def check_duration(info: dict) -> float:
    try:
        duration = float(info.get("duration"))
    except (ValueError, TypeError) as exc:
        raise DownloadRejected(
            "Video duration could not be verified."
        ) from exc

    if not math.isfinite(duration) or duration <= 0:
        raise DownloadRejected(
            "Invalid video duration."
        )

    if duration > MAX_DURATION_SECONDS:
        raise DownloadRejected(
            f"Video exceeds {MAX_DURATION_SECONDS} seconds."
        )

    return duration


def check_size(info: dict) -> None:
    size = (
        info.get("filesize")
        or info.get("filesize_approx")
    )

    if size is None:
        return

    try:
        size = int(size)
    except (ValueError, TypeError) as exc:
        raise DownloadRejected(
            "Invalid video size metadata."
        ) from exc

    if size > MAX_VIDEO_BYTES:
        raise DownloadRejected(
            "Video exceeds the configured size limit."
        )


def safe_title(value) -> str:
    title = str(value or "YouTube Video")

    title = "".join(
        char
        for char in title
        if char.isprintable()
        and char not in '\\/:*?"<>|'
    )

    return (
        " ".join(title.split()).strip(" .")[:200]
        or "YouTube Video"
    )


def validate_downloaded_duration(
    video_path: Path,
) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )

        duration = float(result.stdout.strip())

    except (
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        raise DownloadRejected(
            "Unable to verify downloaded video duration."
        ) from exc

    if not math.isfinite(duration) or duration <= 0:
        raise DownloadRejected(
            "Downloaded video has an invalid duration."
        )

    if duration > MAX_DURATION_SECONDS:
        raise DownloadRejected(
            "Downloaded video exceeds the duration limit."
        )

    return duration


def download_video(
    url: str,
    visitor_id: str | None = None,
) -> dict:
    """
    Download one YouTube video and register it.

    In public mode visitor_id is mandatory.
    Ownership is saved in the same transaction
    that creates the video record.
    """
    url = validate_youtube_url(url)

    if IS_PUBLIC and not visitor_id:
        raise DownloadRejected(
            "A visitor session is required."
        )

    video_id = uuid4().hex
    destination = UPLOAD_DIR / f"{video_id}.mp4"

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="videomoment_url_",
            dir=UPLOAD_DIR,
        )
    )

    try:
        def progress_hook(data):
            if data.get("status") != "downloading":
                return

            downloaded = int(
                data.get("downloaded_bytes") or 0
            )

            total = (
                data.get("total_bytes")
                or data.get("total_bytes_estimate")
            )

            if downloaded > MAX_VIDEO_BYTES:
                raise DownloadRejected(
                    "Download exceeded the size limit."
                )

            if total and int(total) > MAX_VIDEO_BYTES:
                raise DownloadRejected(
                    "Video exceeds the size limit."
                )

        common_options = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "ignoreerrors": False,
            "socket_timeout": 20,
            "retries": 2,
            "fragment_retries": 2,
            "extract_flat": False,
            "skip_download": True,
            "ignoreconfig": True,
            "cachedir": False,
        }

        with yt_dlp.YoutubeDL(common_options) as ydl:
            info = ydl.extract_info(
                url,
                download=False,
            )

        if not isinstance(info, dict):
            raise DownloadRejected(
                "Unable to retrieve video information."
            )

        if info.get("_type") in {
            "playlist",
            "multi_video",
        }:
            raise DownloadRejected(
                "Please provide a single video URL."
            )

        if (
            info.get("is_live")
            or info.get("live_status")
            in {"is_live", "is_upcoming"}
        ):
            raise DownloadRejected(
                "Live streams are not supported."
            )

        check_duration(info)
        check_size(info)

        title = safe_title(info.get("title"))

        download_options = {
            **common_options,
            "skip_download": False,
            "format": (
                "bestvideo[ext=mp4]+bestaudio[ext=m4a]"
                "/best[ext=mp4]"
            ),
            "merge_output_format": "mp4",
            "outtmpl": str(
                temp_dir / "video.%(ext)s"
            ),
            "progress_hooks": [progress_hook],
            "overwrites": False,
            "restrictfilenames": True,
            "max_filesize": MAX_VIDEO_BYTES,
        }

        with yt_dlp.YoutubeDL(download_options) as ydl:
            ydl.download([url])

        candidates = [
            path
            for path in temp_dir.glob("*.mp4")
            if path.is_file() and not path.is_symlink()
        ]

        if not candidates:
            raise DownloadRejected(
                "No MP4 video was produced."
            )

        downloaded_file = max(
            candidates,
            key=lambda path: path.stat().st_size,
        )

        size = downloaded_file.stat().st_size

        if size <= 0:
            raise DownloadRejected(
                "The downloaded video is empty."
            )

        if size > MAX_VIDEO_BYTES:
            raise DownloadRejected(
                "Video exceeds the size limit."
            )

        duration = validate_downloaded_duration(
            downloaded_file
        )

        # Validate before moving into permanent
        # temporary-upload storage.
        downloaded_file.replace(destination)

        filename = f"{title}.mp4"

        with SessionLocal() as db:
            video = Video(
                id=video_id,
                visitor_id=visitor_id if IS_PUBLIC else None,
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
                    "Ready for indexing."
                ),
            )

            db.add(video)
            db.commit()
            db.refresh(video)

            uploaded_at = video.uploaded_at

        return {
            "video_id": video_id,
            "filename": filename,
            "status": "uploaded",
            "uploaded_at": uploaded_at.isoformat(),
            "title": title,
            "duration": duration,
            "message": (
                "Video imported successfully. "
                "Start indexing to enable search."
            ),
        }

    except Exception:
        destination.unlink(missing_ok=True)
        raise

    finally:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


@router.post(
    "/import-url",
    response_model=VideoURLResponse,
    status_code=201,
)
async def import_video_url(
    request: VideoURLRequest,
    http_request: Request,
):
    # Authenticate before performing expensive work.
    visitor_id = None

    if IS_PUBLIC:
        visitor_id = get_current_visitor_id(
            http_request
        )

    # Existing access-code and request-limit checks.
    require_processing_access(http_request)

    try:
        validated_url = validate_youtube_url(
            request.url
        )
    except DownloadRejected as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    acquired = True

    if IS_PUBLIC:
        acquired = _import_slots.acquire(
            blocking=False
        )

    if not acquired:
        raise HTTPException(
            status_code=429,
            detail=(
                "Video import is busy. "
                "Please try again later."
            ),
            headers={"Retry-After": "60"},
        )

    try:
        result = await run_in_threadpool(
            download_video,
            validated_url,
            visitor_id,
        )

        return VideoURLResponse(**result)

    except DownloadRejected as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except yt_dlp.utils.DownloadError as error:
        logger.warning(
            "YouTube download failed: %s",
            error,
        )

        raise HTTPException(
            status_code=422,
            detail=(
                "Unable to download this video. "
                "It may be private, restricted, "
                "unavailable, or unsupported."
            ),
        ) from error

    except SQLAlchemyError as error:
        logger.exception(
            "Database registration failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Video registration failed.",
        ) from error

    except HTTPException:
        raise

    except Exception as error:
        logger.exception(
            "Unexpected URL import failure."
        )

        raise HTTPException(
            status_code=500,
            detail="Video import failed.",
        ) from error

    finally:
        if IS_PUBLIC and acquired:
            _import_slots.release()
