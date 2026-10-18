import hashlib
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,SiglipModel
from deployment_config import IS_DEMO
from deployment_cache import resolve_deployment_cache
from sample_video_identity import get_sample_video_id

MODEL_NAME="google/siglip-base-patch16-224"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_BATCH_SIZE=16 if device.type=="cuda" else 4
BASE_DIR=Path(__file__).resolve().parent
PROJECT_ROOT=BASE_DIR.parent
CACHE_DIR=PROJECT_ROOT/"video_cache"
CACHE_DIR.mkdir(parents=True,exist_ok=True)
DIAGNOSTICS_DIR=PROJECT_ROOT/"experiments"/"diagnostics"
CACHE_VERSION=1
processor=None
model=None

def load_siglip():
    global processor,model
    if processor is None:
        print("Loading SigLIP processor...",flush=True)
        processor=AutoProcessor.from_pretrained(MODEL_NAME)
    if model is None:
        print(f"Loading SigLIP model on {device}...",flush=True)
        model=SiglipModel.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
    return processor,model

def get_device_info():
    info={
        "device":str(device),
        "cuda_available":torch.cuda.is_available(),
        "batch_size":DEFAULT_BATCH_SIZE}
    if device.type=="cuda":
        info["gpu_name"]=torch.cuda.get_device_name(0)
    return info

def get_cache_path(video_path,sample_interval):
    path=Path(video_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    if sample_interval<=0:
        raise ValueError("sample_interval must be greater than zero.")

    stat=path.stat()
    video_identity=f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
    video_id=hashlib.sha256(video_identity.encode("utf-8")).hexdigest()[:16]
    settings={
        "model":MODEL_NAME,
        "sample_interval":sample_interval,
        "cache_version":CACHE_VERSION}
    settings_id=hashlib.sha256(
        json.dumps(settings,sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return CACHE_DIR/f"{video_id}_siglip_{settings_id}.npz"

def encode_frame_batch(images):
    if IS_DEMO:
        raise PermissionError("Frame indexing is disabled in demo mode.")
    if not images:
        return np.empty((0,0),dtype=np.float32)

    siglip_processor,siglip_model=load_siglip()
    inputs=siglip_processor(images=images,return_tensors="pt")
    inputs={
        key:value.to(device) if isinstance(value,torch.Tensor) else value
        for key,value in inputs.items()}

    with torch.inference_mode():
        output=siglip_model.get_image_features(**inputs)
        if hasattr(output,"pooler_output"):
            embeddings=output.pooler_output
        else:
            embeddings=output
        embeddings=torch.nn.functional.normalize(embeddings,dim=-1)

    return embeddings.detach().float().cpu().numpy()

def extract_frame_embeddings(video_path,sample_interval=0.5,batch_size=None):
    if IS_DEMO:
        raise PermissionError("Frame extraction is disabled in demo mode.")
    if sample_interval<=0:
        raise ValueError("sample_interval must be greater than zero.")
    if batch_size is None:
        batch_size=DEFAULT_BATCH_SIZE
    if not isinstance(batch_size,int) or batch_size<=0:
        raise ValueError("batch_size must be a positive integer.")

    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Unable to open video.")

    fps=cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=0:
        cap.release()
        raise ValueError("Invalid video FPS.")

    frame_interval=max(1,round(fps*sample_interval))
    frame_embeddings=[]
    timestamps=[]
    batch_images=[]
    batch_timestamps=[]
    frame_number=0

    load_siglip()

    def flush_batch():
        if not batch_images:
            return
        embeddings=encode_frame_batch(batch_images)
        if embeddings.shape[0]!=len(batch_timestamps):
            raise RuntimeError("SigLIP batch embedding count mismatch.")
        frame_embeddings.append(embeddings)
        timestamps.extend(batch_timestamps)
        batch_images.clear()
        batch_timestamps.clear()

    try:
        while True:
            grabbed=cap.grab()
            if not grabbed:
                break

            if frame_number%frame_interval==0:
                ret,frame=cap.retrieve()
                if not ret:
                    raise RuntimeError(
                        f"Failed to retrieve frame {frame_number}")
                timestamp=frame_number/fps
                rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                batch_images.append(Image.fromarray(rgb_frame))
                batch_timestamps.append(timestamp)
                if len(batch_images)>=batch_size:
                    flush_batch()

            frame_number+=1

        flush_batch()
    finally:
        cap.release()

    if not frame_embeddings:
        return (
            np.empty((0,),dtype=np.float64),
            np.empty((0,0),dtype=np.float32))

    return (
        np.asarray(timestamps,dtype=np.float64),
        np.concatenate(frame_embeddings,axis=0).astype(np.float32))

def load_siglip_cache(cache_path):
    cache_path=Path(cache_path)
    with np.load(cache_path,allow_pickle=False) as data:
        timestamps=data["timestamps"]
        embeddings=data["embeddings"]

    valid=(
        timestamps.ndim==1
        and embeddings.ndim==2
        and embeddings.shape[0]==len(timestamps)
        and (embeddings.shape[1]>0 or len(timestamps)==0)
        and np.all(np.isfinite(timestamps))
        and np.all(np.isfinite(embeddings)))

    if not valid:
        raise ValueError(f"Invalid SigLIP cache: {cache_path}")

    return timestamps,embeddings

def get_cached_frame_embeddings(video_path,sample_interval=0.5):
    if IS_DEMO:
        if sample_interval!=0.5:
            raise ValueError(
                "Public demo supports the pre-indexed "
                "0.5-second visual interval only.")

        video_id = get_sample_video_id(video_path)
        cache_path=resolve_deployment_cache(video_id,"siglip")
        print(
            f"[DEMO CACHE] Loading SigLIP: {cache_path.name}",flush=True)
        return load_siglip_cache(cache_path)

    cache_path=get_cache_path(video_path,sample_interval)

    if cache_path.exists():
        print(
            f"Loading SigLIP frame cache: {cache_path.name}",
            flush=True)
        try:
            return load_siglip_cache(cache_path)
        except (OSError,ValueError,KeyError) as error:
            print("Unable to read SigLIP cache:",error)

    print("Computing SigLIP frame embeddings...",flush=True)
    timestamps,embeddings=extract_frame_embeddings(
        video_path,
        sample_interval)

    temporary_path=cache_path.with_suffix(".tmp")
    try:
        with open(temporary_path,"wb") as file:
            np.savez_compressed(
                file,
                timestamps=timestamps,
                embeddings=embeddings)
        os.replace(temporary_path,cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"SigLIP frame cache saved: {cache_path.name}",flush=True)
    return timestamps,embeddings
def get_visual_results(video_path, query, sample_interval=0.5):
    """
    Retrieve video frames matching a natural-language query
    using cached SigLIP image embeddings.
    """
    if not isinstance(query, str) or not query.strip():
        return []

    timestamps, frame_embeddings = get_cached_frame_embeddings(
        video_path,
        sample_interval=sample_interval,
    )

    if len(timestamps) == 0:
        return []

    siglip_processor, siglip_model = load_siglip()

    inputs = siglip_processor(
        text=[query.strip()],
        padding="max_length",
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        text_features = siglip_model.get_text_features(**inputs)

        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output

        text_features = torch.nn.functional.normalize(
            text_features,
            dim=-1,
        )

    query_embedding = (
        text_features.detach()
        .float()
        .cpu()
        .numpy()[0]
    )

    similarities = frame_embeddings @ query_embedding

    results = []

    for timestamp, similarity in zip(timestamps, similarities):
        results.append({
            "timestamp": float(timestamp),
            "similarity_score": float(max(0.0, similarity)),
            "modality": "visual",
        })

    results.sort(
        key=lambda item: item["similarity_score"],
        reverse=True,
    )

    return results