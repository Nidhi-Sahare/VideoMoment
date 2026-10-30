import hashlib
import json
import os
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from deployment_config import IS_DEMO
from deployment_cache import (
    resolve_deployment_cache,
    use_deployment_cache,
)
from sample_video_identity import get_sample_video_id

BASE_DIR=Path(__file__).resolve().parent
PROJECT_ROOT=BASE_DIR.parent
CACHE_DIR=PROJECT_ROOT/"video_cache"
CACHE_DIR.mkdir(parents=True,exist_ok=True)
INDEX_VERSION=1

def get_video_id(video_path):
    path=Path(video_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    stat=path.stat()
    identity=f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]

def get_cache_path(video_path,module_name,settings):
    video_id=get_video_id(video_path)
    settings_json=json.dumps(settings,sort_keys=True)
    settings_id=hashlib.sha256(settings_json.encode("utf-8")).hexdigest()[:12]
    return CACHE_DIR/f"{video_id}_{module_name}_{settings_id}.json"


def make_json_serializable(value):
    if isinstance(value,dict):
        return {
            str(key):make_json_serializable(item)
            for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [make_json_serializable(item) for item in value]
    if isinstance(value,np.ndarray):
        return make_json_serializable(value.tolist())
    if isinstance(value,np.integer):
        return int(value)
    if isinstance(value,np.floating):
        return float(value)
    if isinstance(value,np.bool_):
        return bool(value)
    if isinstance(value,Path):
        return str(value)
    return value

def write_json_atomic(path,data):
    path=Path(path)
    temporary_path=path.with_suffix(".tmp")
    try:
        with temporary_path.open("w",encoding="utf-8") as file:
            json.dump(make_json_serializable(data),file,indent=2,ensure_ascii=False)
        os.replace(temporary_path,path)
    finally:
        temporary_path.unlink(missing_ok=True)

def load_or_compute(cache_path,compute_fn):
    cache_path=Path(cache_path)
    if cache_path.exists():
        print(f"[CACHE] Loading: {cache_path.name}",flush=True)
        with cache_path.open("r",encoding="utf-8") as file:
            return json.load(file)
    if IS_DEMO:
        raise FileNotFoundError(
            "Cache is missing. Computation is disabled "
            "in public demo mode.")
    print(f"[CACHE] Computing: {cache_path.name}",flush=True)
    started=time.perf_counter()
    results=make_json_serializable(compute_fn())
    write_json_atomic(cache_path,results)
    print(
        f"[CACHE] Saved: {cache_path.name} "
        f"in {time.perf_counter()-started:.2f}s",
        flush=True)
    return results

def get_cached_yolo(video_path,sample_interval=1.0,confidence_threshold=0.4):
    if use_deployment_cache(video_path):
        video_id=get_sample_video_id(video_path)
        cache_path=resolve_deployment_cache(video_id,"yolo")
        print(f"[DEMO CACHE] Loading YOLO: {cache_path.name}",flush=True)
        with cache_path.open("r",encoding="utf-8") as file:
            return json.load(file)

    from yolo_detection import detect_objects
    cache_path=get_cache_path(
        video_path,
        "yolo11n",
        {
            "sample_interval":sample_interval,
            "confidence_threshold":confidence_threshold})
    return load_or_compute(
        cache_path,
        lambda:detect_objects(
            video_path,
            sample_interval=sample_interval,
            confidence_threshold=confidence_threshold))

def get_cached_ocr(video_path,sample_interval=3.0,progress_callback=None):
    if use_deployment_cache(video_path):
        video_id=get_sample_video_id(video_path)
        cache_path=resolve_deployment_cache(video_id,"ocr")
        print(f"[DEMO CACHE] Loading OCR: {cache_path.name}",flush=True)
        with cache_path.open("r",encoding="utf-8") as file:
            return json.load(file)

    from ocr_detection import extract_text
    cache_path=get_cache_path(
        video_path,
        "easyocr",
        {"sample_interval":sample_interval})

    if cache_path.exists():
        print("[OCR CACHE] Existing OCR results found.",flush=True)
        return load_or_compute(
            cache_path,
            lambda:extract_text(
                video_path,
                sample_interval=sample_interval))

    print(
        "[OCR CACHE] No completed OCR cache. "
        "Starting extraction.",
        flush=True)
    return load_or_compute(
        cache_path,
        lambda:extract_text(
            video_path,
            sample_interval=sample_interval,
            progress_callback=progress_callback))

def report_progress(callback,stage,progress,message):
    progress=max(0,min(100,int(progress)))
    event={
        "stage":stage,
        "progress":progress,
        "message":message}
    print(f"[INDEX {progress}%] {message}",flush=True)
    if callback is not None:
        callback(event)

def print_timing_report(timings,total_seconds):
    print("\n"+"="*55)
    print("VIDEOMOMENT INDEXING PERFORMANCE")
    print("="*55)
    for name,seconds in timings.items():
        percentage=seconds/total_seconds*100 if total_seconds>0 else 0.0
        print(
            f"{name:<22} "
            f"{seconds:>9.2f}s "
            f"({percentage:>5.1f}%)")
    print("-"*55)
    print(f"{'TOTAL':<22} {total_seconds:>9.2f}s")
    print("="*55,flush=True)

def index_video(
    video_path,
    progress_callback=None,
    visual_interval=0.5,
    ocr_interval=3.0,
    object_interval=1.0,
    confidence_threshold=0.4):

    if IS_DEMO:
        raise PermissionError(
            "Video indexing is disabled "
            "in public demo mode.")

    video_path=Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if (
        not np.isfinite(visual_interval)
        or not np.isfinite(ocr_interval)
        or not np.isfinite(object_interval)
        or visual_interval<=0
        or ocr_interval<=0
        or object_interval<=0
    ):
        raise ValueError("Sampling intervals must be positive.")

    if (
        not np.isfinite(confidence_threshold)
        or not 0<=confidence_threshold<=1
    ):
        raise ValueError(
            "confidence_threshold must be between 0 and 1.")

    total_started=time.perf_counter()
    timings={}

    def record_timing(stage_name,started):
        seconds=time.perf_counter()-started
        timings[stage_name]=round(seconds,3)
        print(f"[TIMING] {stage_name}: {seconds:.2f}s",flush=True)

    video_cache_id=get_video_id(video_path)
    print(f"\n[INDEX] Video: {video_path.name}",flush=True)
    print(f"[INDEX] Cache ID: {video_cache_id}",flush=True)

    report_progress(
        progress_callback,
        "starting",
        0,
        "Starting video indexing")

    report_progress(
        progress_callback,
        "visual",
        0,
        "Indexing visual frames with SigLIP")

    stage_started=time.perf_counter()
    from video_frames import get_cached_frame_embeddings

    timestamps,visual_embeddings=get_cached_frame_embeddings(
        str(video_path),
        sample_interval=visual_interval)

    record_timing("SigLIP",stage_started)
    visual_count=len(timestamps)

    if visual_embeddings.shape[0]!=visual_count:
        raise RuntimeError(
            "Visual embedding count does not "
            "match timestamps.")

    report_progress(
        progress_callback,
        "visual",
        25,
        f"SigLIP complete: {visual_count} frames")

    report_progress(
        progress_callback,
        "transcription",
        25,
        "Indexing speech with Whisper")

    stage_started=time.perf_counter()
    from audio_frame import transcribe_video,get_transcript_embeddings

    transcript=transcribe_video(str(video_path))
    record_timing("Whisper",stage_started)

    report_progress(
        progress_callback,
        "transcription",
        45,
        f"Whisper complete: {len(transcript)} segments")

    report_progress(
        progress_callback,
        "audio_embeddings",
        45,
        "Generating transcript embeddings")

    stage_started=time.perf_counter()

    if transcript:
        audio_embeddings=get_transcript_embeddings(
            str(video_path),
            transcript)
        if len(audio_embeddings)!=len(transcript):
            raise RuntimeError(
                "Audio embedding count does not "
                "match transcript.")
        audio_embedding_count=len(audio_embeddings)
    else:
        audio_embedding_count=0

    record_timing("MiniLM",stage_started)

    report_progress(
        progress_callback,
        "audio_embeddings",
        60,
        "Transcript embeddings complete")

    report_progress(
        progress_callback,
        "objects",
        60,
        "Indexing objects with YOLO")

    stage_started=time.perf_counter()

    objects=get_cached_yolo(
        str(video_path),
        sample_interval=object_interval,
        confidence_threshold=confidence_threshold)

    record_timing("YOLO",stage_started)

    report_progress(
        progress_callback,
        "objects",
        80,
        f"YOLO complete: {len(objects)} records")

    report_progress(
        progress_callback,
        "ocr",
        80,
        "Starting on-screen text extraction with EasyOCR")

    stage_started=time.perf_counter()

    def on_ocr_progress(event):
        processed=int(event.get("processed",0))
        total=int(event.get("total",0))
        timestamp=float(event.get("timestamp",0.0))
        frame_seconds=float(event.get("frame_seconds",0.0))

        if total<=0:
            return

        fraction=min(1.0,max(0.0,processed/total))
        overall_progress=80+int(fraction*15)
        message=(
            f"EasyOCR: {processed}/{total} frames "
            f"processed | video t={timestamp:.1f}s "
            f"| last frame={frame_seconds:.1f}s")

        report_progress(
            progress_callback,
            "ocr",
            overall_progress,
            message)

    ocr_records=get_cached_ocr(
        str(video_path),
        sample_interval=ocr_interval,
        progress_callback=on_ocr_progress)

    record_timing("EasyOCR",stage_started)
    ocr_elapsed=timings["EasyOCR"]

    report_progress(
        progress_callback,
        "ocr",
        95,
        f"EasyOCR complete: "
        f"{len(ocr_records)} records "
        f"in {ocr_elapsed:.2f}s")

    finalization_started=time.perf_counter()
    elapsed=round(time.perf_counter()-total_started,3)

    summary={
        "video_cache_id":video_cache_id,
        "video_path":str(video_path),
        "index_version":INDEX_VERSION,
        "status":"indexed",
        "indexed_at":datetime.now(timezone.utc).isoformat(),
        "processing_time_seconds":elapsed,
        "features":{
            "visual_frames":visual_count,
            "transcript_segments":len(transcript),
            "audio_embeddings":audio_embedding_count,
            "object_records":len(objects),
            "ocr_records":len(ocr_records)},
        "settings":{
            "visual_interval":visual_interval,
            "ocr_interval":ocr_interval,
            "object_interval":object_interval,
            "confidence_threshold":confidence_threshold},
        "scene_indexing":"not_yet_integrated",
        "stage_timings_seconds":timings.copy()}

    manifest_path=CACHE_DIR/f"{video_cache_id}_index_manifest.json"
    write_json_atomic(manifest_path,summary)

    record_timing("Finalization",finalization_started)

    total_elapsed=round(time.perf_counter()-total_started,3)
    summary["processing_time_seconds"]=total_elapsed
    summary["stage_timings_seconds"]=timings.copy()

    report_progress(
        progress_callback,
        "completed",
        100,
        "Video indexing completed successfully")

    print_timing_report(timings,total_elapsed)
    print(f"\n[INDEX] Completed in {total_elapsed:.2f}s",flush=True)
    return summary

if __name__=="__main__":
    video_path="sample_videos/nasa.mp4"
    print("\n--- VIDEO INDEXING TEST ---")
    result=index_video(video_path)
    print("\n--- INDEXING SUMMARY ---")
    print(json.dumps(result,indent=2))