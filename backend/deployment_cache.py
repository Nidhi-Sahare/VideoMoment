import json
from pathlib import Path
from deployment_config import IS_DEMO,SAMPLE_VIDEO_IDS

BASE_DIR=Path(__file__).resolve().parent
PROJECT_ROOT=BASE_DIR.parent
DEPLOYMENT_CACHE_DIR=PROJECT_ROOT/"deployment_cache"

def get_deployment_cache_dir(video_id:str)->Path:
    return DEPLOYMENT_CACHE_DIR/video_id

def get_deployment_manifest(video_id:str)->dict|None:
    if not IS_DEMO:
        return None
    if video_id not in SAMPLE_VIDEO_IDS:
        raise PermissionError("This video is unavailable in the public demo.")

    manifest_path=get_deployment_cache_dir(video_id)/"manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Deployment cache manifest missing for {video_id}.")

    with manifest_path.open("r",encoding="utf-8") as file:
        manifest=json.load(file)

    if manifest.get("video_id")!=video_id:
        raise ValueError("Deployment cache manifest video ID mismatch.")

    return manifest

def resolve_deployment_cache(video_id:str,cache_name:str)->Path:
    if not IS_DEMO:
        raise RuntimeError("Deployment caches are only used in demo mode.")

    manifest=get_deployment_manifest(video_id)
    files=manifest.get("files",{})

    if cache_name not in files:
        raise FileNotFoundError(
            f"Required deployment cache missing: {cache_name}")

    cache_dir=get_deployment_cache_dir(video_id).resolve()
    cache_path=(cache_dir/files[cache_name]).resolve()

    if not cache_path.is_relative_to(cache_dir):
        raise ValueError("Invalid deployment cache path.")

    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Deployment cache file not found: {cache_path}")

    return cache_path