
"""
VideoMoment temporary visitor-video cleanup.

Safety:
- Public mode only
- Dry-run by default
- Protects permanent sample videos
- Protects legacy videos without visitor ownership
- Skips active indexing jobs
- Skips videos with active lifecycle locks
- Restricts deletion to approved directories
- Keeps database records if video-file deletion fails

Requires one API worker and shared lifecycle locks.
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select

from database import SessionLocal
from models import Video
from deployment_config import (
    IS_PUBLIC,
    SAMPLE_VIDEO_IDS,
    SESSION_EXPIRY_SECONDS,
)
from video_lifecycle import (
    try_acquire_video_lock,
    release_video_lock,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPLOAD_DIR = (PROJECT_ROOT / "uploads").resolve()
CACHE_DIR = (PROJECT_ROOT / "video_cache").resolve()
METADATA_DIR = (PROJECT_ROOT / "video_metadata").resolve()

CACHE_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")
ACTIVE_STATUSES = {"queued", "indexing"}


def utc_now():
    return datetime.now(timezone.utc)


def normalize_datetime(value):
    if value is None:
        return None

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def is_inside_directory(path, directory):
    try:
        Path(path).resolve().relative_to(directory)
        return True
    except (ValueError, OSError, RuntimeError):
        return False


def safe_unlink(path, directory, dry_run=True):
    path = Path(path)

    if path.is_symlink():
        logger.warning("Refusing symlink: %s", path)
        return False

    if not is_inside_directory(path, directory):
        logger.warning(
            "Refusing deletion outside %s: %s",
            directory,
            path,
        )
        return False

    if not path.is_file():
        return False

    if dry_run:
        logger.info("[DRY RUN] Would delete %s", path)
        return True

    try:
        path.unlink()
        logger.info("Deleted %s", path)
        return True
    except OSError:
        logger.exception("Failed deleting %s", path)
        return False


def get_cache_id(video):
    summary = video.indexing_summary

    if isinstance(summary, dict):
        cache_id = summary.get("video_cache_id")

        if (
            isinstance(cache_id, str)
            and CACHE_ID_PATTERN.fullmatch(cache_id)
        ):
            return cache_id

    video_path = Path(video.file_path)

    if (
        video_path.is_symlink()
        or not video_path.is_file()
        or not is_inside_directory(video_path, UPLOAD_DIR)
    ):
        return None

    try:
        from video_index import get_video_id

        cache_id = get_video_id(str(video_path))

        if CACHE_ID_PATTERN.fullmatch(cache_id):
            return cache_id

    except (OSError, ValueError):
        logger.exception(
            "Cannot determine cache ID: %s",
            video.id,
        )

    return None


def get_cache_files(cache_id):
    if (
        not isinstance(cache_id, str)
        or not CACHE_ID_PATTERN.fullmatch(cache_id)
    ):
        return []

    return [
        path
        for path in CACHE_DIR.glob(f"{cache_id}_*")
        if path.is_file()
        and not path.is_symlink()
        and is_inside_directory(path, CACHE_DIR)
    ]


def get_active_video_ids():
    """
    Read active jobs from the running API process.

    If state is unavailable, cleanup fails closed.
    """
    try:
        import main

        with main.metadata_lock:
            return set(main.active_index_jobs)

    except (ImportError, AttributeError):
        logger.warning(
            "Active indexing state unavailable."
        )
        return None


def cleanup_expired_videos(
    dry_run=True,
    active_video_ids=None,
):
    summary = {
        "dry_run": dry_run,
        "examined": 0,
        "expired_candidates": 0,
        "skipped_samples": 0,
        "skipped_legacy": 0,
        "skipped_active": 0,
        "skipped_locked": 0,
        "skipped_unsafe": 0,
        "deleted_records": 0,
        "deleted_files": 0,
        "errors": 0,
    }

    if not IS_PUBLIC:
        summary["skipped_mode"] = True
        return summary

    if active_video_ids is None:
        active_video_ids = get_active_video_ids()

    if active_video_ids is None:
        summary["skipped_active_state_unknown"] = True
        return summary

    cutoff = utc_now() - timedelta(
        seconds=SESSION_EXPIRY_SECONDS
    )

    with SessionLocal() as db:
        videos = db.scalars(select(Video)).all()
        summary["examined"] = len(videos)

        protected_cache_ids = set()
        candidates = []

        for video in videos:
            cache_id = get_cache_id(video)

            if video.id in SAMPLE_VIDEO_IDS:
                summary["skipped_samples"] += 1

            elif video.visitor_id is None:
                summary["skipped_legacy"] += 1

            elif (
                video.id in active_video_ids
                or video.status in ACTIVE_STATUSES
            ):
                summary["skipped_active"] += 1

            elif (
                normalize_datetime(video.uploaded_at) is None
                or normalize_datetime(video.uploaded_at) > cutoff
            ):
                pass

            else:
                video_path = Path(video.file_path)

                if (
                    video_path.is_symlink()
                    or not is_inside_directory(
                        video_path, UPLOAD_DIR
                    )
                ):
                    summary["skipped_unsafe"] += 1
                else:
                    candidates.append(video.id)
                    continue

            if cache_id:
                protected_cache_ids.add(cache_id)

        summary["expired_candidates"] = len(candidates)

        for video_id in candidates:
            lock = try_acquire_video_lock(video_id)

            if lock is None:
                summary["skipped_locked"] += 1
                continue

            try:
                # Refresh the record while holding the lock.
                db.expire_all()
                video = db.get(Video, video_id)

                if video is None:
                    continue

                if video.id in SAMPLE_VIDEO_IDS:
                    summary["skipped_samples"] += 1
                    continue

                if video.visitor_id is None:
                    summary["skipped_legacy"] += 1
                    continue

                if (
                    video.id in active_video_ids
                    or video.status in ACTIVE_STATUSES
                ):
                    summary["skipped_active"] += 1
                    continue

                uploaded_at = normalize_datetime(
                    video.uploaded_at
                )

                if (
                    uploaded_at is None
                    or uploaded_at > cutoff
                ):
                    continue

                video_path = Path(video.file_path)

                if (
                    video_path.is_symlink()
                    or not is_inside_directory(
                        video_path, UPLOAD_DIR
                    )
                ):
                    summary["skipped_unsafe"] += 1
                    continue

                cache_id = get_cache_id(video)

                if dry_run:
                    logger.info(
                        "[DRY RUN] Would delete video %s",
                        video_id,
                    )
                    continue

                # Delete the primary file before removing
                # its database record. If deletion fails,
                # preserve the record for a later retry.
                if video_path.is_file():
                    if not safe_unlink(
                        video_path,
                        UPLOAD_DIR,
                        dry_run=False,
                    ):
                        summary["errors"] += 1
                        continue

                    summary["deleted_files"] += 1

                # Delete the database record and its
                # related indexing/search records.
                try:
                    db.delete(video)
                    db.commit()
                    summary["deleted_records"] += 1

                except Exception:
                    db.rollback()
                    summary["errors"] += 1
                    logger.exception(
                        "Database deletion failed: %s",
                        video_id,
                    )
                    continue

                metadata_path = (
                    METADATA_DIR / f"{video_id}.json"
                )

                if metadata_path.is_file():
                    if safe_unlink(
                        metadata_path,
                        METADATA_DIR,
                        dry_run=False,
                    ):
                        summary["deleted_files"] += 1
                    else:
                        summary["errors"] += 1

                if (
                    cache_id
                    and cache_id not in protected_cache_ids
                ):
                    for cache_file in get_cache_files(
                        cache_id
                    ):
                        if safe_unlink(
                            cache_file,
                            CACHE_DIR,
                            dry_run=False,
                        ):
                            summary["deleted_files"] += 1
                        else:
                            summary["errors"] += 1

            finally:
                release_video_lock(lock)

    logger.info("Cleanup result: %s", summary)
    return summary


if __name__ == "__main__":
    result = cleanup_expired_videos(
        dry_run=True
    )
    print(json.dumps(result, indent=2))
