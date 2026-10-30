import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from deployment_config import IS_DEMO
from deployment_cache import (
    resolve_deployment_cache,
    use_deployment_cache,
)
from sample_video_identity import get_sample_video_id

WHISPER_MODEL_NAME="base"
TEXT_MODEL_NAME="all-MiniLM-L6-v2"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
WHISPER_FP16=DEVICE.type=="cuda"
BASE_DIR=Path(__file__).resolve().parent
PROJECT_ROOT=BASE_DIR.parent
CACHE_DIR=PROJECT_ROOT/"video_cache"
CACHE_DIR.mkdir(parents=True,exist_ok=True)
DIAGNOSTICS_DIR=PROJECT_ROOT/"experiments"/"diagnostics"
whisper_model=None
text_model=None

def get_device_info():
    info={
        "device":str(DEVICE),
        "cuda_available":torch.cuda.is_available(),
        "whisper_fp16":WHISPER_FP16,
        "whisper_model":WHISPER_MODEL_NAME,
        "text_model":TEXT_MODEL_NAME}
    if DEVICE.type=="cuda":
        info["gpu_name"]=torch.cuda.get_device_name(0)
    return info

def get_whisper_model():
    global whisper_model
    if IS_DEMO:
        raise PermissionError(
            "Whisper inference is disabled in public demo mode.")
    if whisper_model is None:
        import whisper
        print(f"Loading Whisper model on {DEVICE}...",flush=True)
        whisper_model=whisper.load_model(
            WHISPER_MODEL_NAME,
            device=str(DEVICE))
    return whisper_model

def get_text_model():
    global text_model
    if text_model is None:
        print(f"Loading text embedding model on {DEVICE}...",flush=True)
        text_model=SentenceTransformer(
            TEXT_MODEL_NAME,
            device=str(DEVICE))
    return text_model

def get_audio_cache_paths(video_path):
    path=Path(video_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    stat=path.stat()
    video_identity=f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
    video_id=hashlib.sha256(video_identity.encode("utf-8")).hexdigest()[:16]

    transcription_settings={
        "model":WHISPER_MODEL_NAME,
        "fp16":WHISPER_FP16,
        "task":"transcribe",
        "language":None}

    transcription_id=hashlib.sha256(
        json.dumps(transcription_settings,sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    transcript_path=CACHE_DIR/f"{video_id}_whisper_{transcription_id}.json"

    embedding_settings={
        "text_model":TEXT_MODEL_NAME,
        "normalize_embeddings":True,
        "transcription_id":transcription_id}

    embedding_id=hashlib.sha256(
        json.dumps(embedding_settings,sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    embeddings_path=CACHE_DIR/f"{video_id}_audio_embeddings_{embedding_id}.npz"
    return transcript_path,embeddings_path

def get_deployment_audio_paths(video_path):
    video_id = get_sample_video_id(video_path)
    transcript_path=resolve_deployment_cache(video_id,"whisper")
    embeddings_path=resolve_deployment_cache(video_id,"audio_embeddings")
    return transcript_path,embeddings_path

def transcribe_video(video_path):
    if use_deployment_cache(video_path):
        transcript_path,_=get_deployment_audio_paths(video_path)
        print(f"[DEMO CACHE] Loading Whisper: {transcript_path.name}",flush=True)
        with transcript_path.open("r",encoding="utf-8") as file:
            transcript=json.load(file)
        if not isinstance(transcript,list):
            raise ValueError("Invalid deployment Whisper transcript.")
        return transcript

    transcript_path,_=get_audio_cache_paths(video_path)

    if transcript_path.exists():
        print(f"Loading Whisper cache: {transcript_path.name}",flush=True)
        with transcript_path.open("r",encoding="utf-8") as file:
            return json.load(file)

    print("Transcribing video with Whisper...",flush=True)
    model=get_whisper_model()
    result=model.transcribe(str(video_path),fp16=WHISPER_FP16)

    transcript=[]
    for segment in result["segments"]:
        transcript.append({
            "start":float(segment["start"]),
            "end":float(segment["end"]),
            "text":segment["text"].strip()})

    temporary_path=transcript_path.with_suffix(".tmp")
    try:
        with temporary_path.open("w",encoding="utf-8") as file:
            json.dump(transcript,file,indent=2,ensure_ascii=False)
        os.replace(temporary_path,transcript_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Whisper cache saved: {transcript_path.name}",flush=True)
    return transcript

def get_transcript_embeddings(video_path,transcript):
    current_hash=hashlib.sha256(
        json.dumps(
            transcript,
            sort_keys=True,
            ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()

    if use_deployment_cache(video_path):
        _, embeddings_path = get_deployment_audio_paths(video_path)
        print(
            f"[DEMO CACHE] Loading audio embeddings: {embeddings_path.name}",
            flush=True)

        with np.load(embeddings_path,allow_pickle=False) as data:
            cached_embeddings=data["embeddings"]
            cached_hash=str(data["transcript_hash"].item())

        valid=(
            cached_hash==current_hash
            and cached_embeddings.ndim==2
            and cached_embeddings.shape[0]==len(transcript)
            and np.all(np.isfinite(cached_embeddings)))

        if not valid:
            raise ValueError(
                "Deployment audio embeddings are "
                "invalid or do not match the transcript.")
        return cached_embeddings

    _,embeddings_path=get_audio_cache_paths(video_path)

    if embeddings_path.exists():
        print(
            f"Loading audio embeddings cache: {embeddings_path.name}",
            flush=True)

        with np.load(embeddings_path,allow_pickle=False) as data:
            cached_embeddings=data["embeddings"]
            cached_hash=str(data["transcript_hash"].item())

        if (
            cached_hash==current_hash
            and cached_embeddings.shape[0]==len(transcript)
        ):
            return cached_embeddings

        print(
            "Audio embedding cache is outdated. "
            "Recomputing...",
            flush=True)

    print("Computing transcript embeddings...",flush=True)
    texts=[item["text"] for item in transcript]
    model=get_text_model()
    embeddings=model.encode(texts,normalize_embeddings=True)
    embeddings=np.asarray(embeddings,dtype=np.float32)

    temporary_path=embeddings_path.with_suffix(".tmp")
    try:
        with temporary_path.open("wb") as file:
            np.savez_compressed(
                file,
                embeddings=embeddings,
                transcript_hash=np.array(current_hash))
        os.replace(temporary_path,embeddings_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Audio embedding cache saved: {embeddings_path.name}",flush=True)
    return embeddings

def get_text_results(video_path,query):
    transcript=transcribe_video(video_path)
    if not transcript:
        return []

    text_embeddings=get_transcript_embeddings(video_path,transcript)
    model=get_text_model()
    query_embedding=model.encode(query,normalize_embeddings=True)
    scores=text_embeddings@query_embedding

    results=[]
    for i,item in enumerate(transcript):
        results.append({
            "start":item["start"],
            "end":item["end"],
            "text":item["text"],
            "score":float(scores[i])})

    results.sort(key=lambda item:item["score"],reverse=True)
    return results

def build_context_windows(transcript,neighbors=1,max_gap_seconds=2.0):
    if neighbors<0:
        raise ValueError("neighbors must be nonnegative.")
    if max_gap_seconds<0:
        raise ValueError("max_gap_seconds must be nonnegative.")
    if not transcript:
        return []

    windows=[]
    count=len(transcript)

    for i in range(count):
        left=i
        right=i

        for _ in range(neighbors):
            previous=left-1
            if previous<0:
                break
            gap=transcript[left]["start"]-transcript[previous]["end"]
            if gap>max_gap_seconds:
                break
            left=previous

        for _ in range(neighbors):
            following=right+1
            if following>=count:
                break
            gap=transcript[following]["start"]-transcript[right]["end"]
            if gap>max_gap_seconds:
                break
            right=following

        context_segments=transcript[left:right+1]
        context_text=" ".join(
            segment["text"].strip()
            for segment in context_segments
            if segment["text"].strip())

        windows.append({
            "segment_index":i,
            "context_start":float(context_segments[0]["start"]),
            "context_end":float(context_segments[-1]["end"]),
            "context_text":context_text,
            "context_indices":list(range(left,right+1))})

    return windows

def get_contextual_text_results(
    video_path,
    query,
    neighbors=1,
    max_gap_seconds=2.0):

    transcript=transcribe_video(video_path)
    if not transcript:
        return []

    direct_embeddings=get_transcript_embeddings(video_path,transcript)
    windows=build_context_windows(
        transcript,
        neighbors=neighbors,
        max_gap_seconds=max_gap_seconds)

    model=get_text_model()
    query_embedding=model.encode(query,normalize_embeddings=True)
    direct_scores=direct_embeddings@query_embedding

    context_texts=[window["context_text"] for window in windows]
    context_embeddings=model.encode(
        context_texts,
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=False)
    context_embeddings=np.asarray(context_embeddings,dtype=np.float32)
    context_scores=context_embeddings@query_embedding

    results=[]
    for i,segment in enumerate(transcript):
        direct_score=float(direct_scores[i])
        context_score=float(context_scores[i])
        window=windows[i]

        results.append({
            "start":float(segment["start"]),
            "end":float(segment["end"]),
            "text":segment["text"],
            "score":direct_score,
            "direct_score":direct_score,
            "context_score":context_score,
            "context_gain":context_score-direct_score,
            "context_start":window["context_start"],
            "context_end":window["context_end"],
            "context_text":window["context_text"],
            "context_indices":window["context_indices"]})

    results.sort(key=lambda item:item["direct_score"],reverse=True)
    return results