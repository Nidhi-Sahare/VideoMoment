import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from sqlalchemy import select
from database import SessionLocal
from models import Video
from sample_videos import SAMPLE_VIDEOS
from audio_frame import get_audio_cache_paths
from video_index import get_video_id

PROJECT_ROOT=Path(__file__).resolve().parent.parent
SOURCE_CACHE=PROJECT_ROOT/"video_cache"
OUTPUT_ROOT=PROJECT_ROOT/"deployment_cache"

def sha256_file(path):
    digest=hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda:file.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()

def find_unique_cache(cache_id,modality,extension):
    pattern=f"{cache_id}_{modality}_*.{extension}"
    matches=sorted(SOURCE_CACHE.glob(pattern))

    if len(matches)!=1:
        raise RuntimeError(
            f"Expected exactly one cache matching {pattern}; "
            f"found {len(matches)}.")

    return matches[0]

def find_video(video_id):
    with SessionLocal() as db:
        video=db.execute(
            select(Video).where(Video.id==video_id)
        ).scalar_one_or_none()

        if video is None:
            raise RuntimeError(
                f"Video missing from database: {video_id}")

        if not video.indexed:
            raise RuntimeError(
                f"Video is not indexed: {video_id}")

        path=Path(video.file_path).resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Video file missing: {path}")

        return path

def validate_audio_pair(transcript_path,embeddings_path):
    with transcript_path.open("r",encoding="utf-8") as file:
        transcript=json.load(file)

    if not isinstance(transcript,list):
        raise ValueError("Invalid Whisper transcript.")

    transcript_hash=hashlib.sha256(
        json.dumps(
            transcript,
            sort_keys=True,
            ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()

    with np.load(embeddings_path,allow_pickle=False) as data:
        embeddings=data["embeddings"]
        cached_hash=str(data["transcript_hash"].item())

    if cached_hash!=transcript_hash:
        raise ValueError(
            "Whisper transcript and MiniLM embeddings "
            "do not match.")

    if embeddings.ndim!=2:
        raise ValueError(
            "Audio embeddings must be a 2D array.")

    if embeddings.shape[0]!=len(transcript):
        raise ValueError(
            "Transcript segment count does not match "
            "audio embedding count.")

def copy_verified(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)

    source_hash=sha256_file(source)
    destination_hash=sha256_file(destination)

    if source_hash!=destination_hash:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch: {source.name}")

    return destination_hash

def export_sample(sample):
    video_id=sample["video_id"]
    video_path=find_video(video_id)
    cache_id=get_video_id(video_path)

    print(f"\n{sample['title']}")
    print(f"Cache ID: {cache_id}")

    whisper_path,audio_path=get_audio_cache_paths(video_path)

    for path in (whisper_path,audio_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required audio cache missing: {path}")

    validate_audio_pair(whisper_path,audio_path)

    cache_sources={
        "siglip":find_unique_cache(
            cache_id,
            "siglip",
            "npz"),
        "whisper":whisper_path,
        "audio_embeddings":audio_path,
        "yolo":find_unique_cache(
            cache_id,
            "yolo11n",
            "json"),
        "ocr":find_unique_cache(
            cache_id,
            "easyocr",
            "json")}

    with np.load(cache_sources["siglip"],allow_pickle=False) as data:
        if not data.files:
            raise ValueError("SigLIP cache is empty.")

    for modality in ("yolo","ocr"):
        with cache_sources[modality].open("r",encoding="utf-8") as file:
            json.load(file)

    destination_dir=OUTPUT_ROOT/video_id
    destination_dir.mkdir(parents=True,exist_ok=True)

    files={}
    checksums={}

    for modality,source in cache_sources.items():
        destination=destination_dir/source.name
        checksums[modality]=copy_verified(source,destination)
        files[modality]=destination.name
        print(f"  {modality}: {source.name}")

    manifest={
        "video_id":video_id,
        "title":sample["title"],
        "source_cache_id":cache_id,
        "files":files,
        "sha256":checksums}

    manifest_path=destination_dir/"manifest.json"

    with manifest_path.open("w",encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False)

    print("  Manifest created.")

def main():
    if not SOURCE_CACHE.is_dir():
        raise FileNotFoundError(
            f"Cache directory missing: {SOURCE_CACHE}")

    print("="*60)
    print("VIDEOMOMENT — DEPLOYMENT CACHE EXPORT")
    print("="*60)

    for sample in SAMPLE_VIDEOS:
        export_sample(sample)

    print("\nAll deployment caches exported.")
    print(f"Location: {OUTPUT_ROOT}")

if __name__=="__main__":
    main()