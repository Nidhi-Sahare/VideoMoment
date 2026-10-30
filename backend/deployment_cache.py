
import hashlib
import hmac
import json
from pathlib import Path

from deployment_config import (
    IS_DEMO,
    IS_PUBLIC,
    SAMPLE_VIDEO_IDS,
)
from sample_video_identity import get_sample_video_id


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEPLOYMENT_CACHE_DIR = PROJECT_ROOT / "deployment_cache"

ALLOWED_CACHE_NAMES = frozenset({
    "siglip",
    "whisper",
    "audio_embeddings",
    "yolo",
    "ocr",
})


def use_deployment_cache(video_path) -> bool:
    """
    Use exported caches for registered sample videos in
    demo or public mode.

    Visitor uploads continue through the ordinary cache
    and indexing pipeline.
    """
    if not (IS_DEMO or IS_PUBLIC):
        return False

    try:
        video_id = get_sample_video_id(video_path)
    except (
            PermissionError,
            ValueError,
            TypeError,
            KeyError,
            FileNotFoundError,
        ):
        return False

    return video_id in SAMPLE_VIDEO_IDS


def get_deployment_cache_dir(video_id: str) -> Path:
    if video_id not in SAMPLE_VIDEO_IDS:
        raise PermissionError(
            "Deployment caches are restricted to registered samples."
        )

    return DEPLOYMENT_CACHE_DIR / video_id


def get_deployment_manifest(video_id: str) -> dict | None:
    if not (IS_DEMO or IS_PUBLIC):
        return None

    manifest_path = (
        get_deployment_cache_dir(video_id) / "manifest.json"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Deployment cache manifest missing for {video_id}."
        )

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if not isinstance(manifest, dict):
        raise ValueError("Invalid deployment cache manifest.")

    if manifest.get("video_id") != video_id:
        raise ValueError(
            "Deployment cache manifest video ID mismatch."
        )

    if not isinstance(manifest.get("files"), dict):
        raise ValueError("Invalid deployment cache file mapping.")

    if not isinstance(manifest.get("sha256"), dict):
        raise ValueError("Missing deployment cache checksums.")

    return manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_deployment_cache(
    video_id: str,
    cache_name: str,
) -> Path:
    if not (IS_DEMO or IS_PUBLIC):
        raise RuntimeError(
            "Deployment caches require demo or public mode."
        )

    if cache_name not in ALLOWED_CACHE_NAMES:
        raise ValueError(
            f"Unknown deployment cache type: {cache_name}"
        )

    manifest = get_deployment_manifest(video_id)

    filename = manifest["files"].get(cache_name)
    expected_hash = manifest["sha256"].get(cache_name)

    if not isinstance(filename, str) or not filename:
        raise FileNotFoundError(
            f"Required deployment cache missing: {cache_name}"
        )

    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(
            f"Invalid checksum for deployment cache: {cache_name}"
        )

    cache_dir = get_deployment_cache_dir(video_id).resolve()
    cache_path = (cache_dir / filename).resolve()

    if not cache_path.is_relative_to(cache_dir):
        raise ValueError("Invalid deployment cache path.")

    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Deployment cache file not found: {cache_path}"
        )

    actual_hash = sha256_file(cache_path)

    if not hmac.compare_digest(
        actual_hash.lower(),
        expected_hash.lower(),
    ):
        raise ValueError(
            f"Deployment cache checksum mismatch: {cache_name}"
        )

    return cache_path
