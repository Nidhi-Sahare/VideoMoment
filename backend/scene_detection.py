from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from video_frames import get_cached_frame_embeddings

CACHE_DIR=Path(__file__).resolve().parent.parent/"video_cache"
CACHE_DIR.mkdir(parents=True,exist_ok=True)
CACHE_VERSION=1

def get_scene_cache_path(video_path,sample_interval):
    path=Path(video_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    stat=path.stat()
    identity={
        "path":str(path),
        "size":stat.st_size,
        "modified":stat.st_mtime_ns,
        "interval":sample_interval,
        "version":CACHE_VERSION}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return CACHE_DIR/f"{key}_scenes.json"

def extract_histograms(video_path,sample_interval=0.5):
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Unable to open video.")
    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps<=0:
        cap.release()
        raise ValueError("Invalid video FPS.")
    frame_interval=max(1,round(fps*sample_interval))
    timestamps=[]
    histograms=[]
    frame_number=0
    try:
        while True:
            success,frame=cap.read()
            if not success:
                break
            if frame_number%frame_interval==0:
                timestamp=frame_number/fps
                small=cv2.resize(frame,(160,90))
                hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
                hist=cv2.calcHist([hsv],[0,1],None,[24,16],[0,180,0,256])
                hist=hist.astype(np.float32)
                hist/=max(float(hist.sum()),1e-12)
                timestamps.append(timestamp)
                histograms.append(hist.flatten())
            frame_number+=1
    finally:
        cap.release()
    return np.asarray(timestamps,dtype=np.float64),np.asarray(histograms,dtype=np.float32)

def robust_normalize(values):
    values=np.asarray(values,dtype=np.float64)
    if len(values)==0:
        return values
    median=np.median(values)
    mad=np.median(np.abs(values-median))
    scale=max(1.4826*mad,1e-6)
    return np.maximum((values-median)/scale,0.0)

def detect_scene_boundaries(video_path,sample_interval=0.5,threshold=4.0,min_scene_duration=1.5):
    if sample_interval<=0:
        raise ValueError("sample_interval must be positive.")
    if min_scene_duration<0:
        raise ValueError("min_scene_duration cannot be negative.")
    cache_path=get_scene_cache_path(video_path,sample_interval)
    if cache_path.exists():
        print(f"Loading scene cache: {cache_path.name}")
        with open(cache_path,"r",encoding="utf-8") as file:
            return json.load(file)

    print("Extracting frame color histograms...")
    hist_times,histograms=extract_histograms(video_path,sample_interval)
    print("Loading cached SigLIP embeddings...")
    siglip_times,embeddings=get_cached_frame_embeddings(video_path,sample_interval)
    count=min(len(hist_times),len(siglip_times))
    if count<2:
        return {
            "boundaries":[],
            "transitions":[],
            "sample_interval":sample_interval}

    hist_times=hist_times[:count]
    histograms=histograms[:count]
    embeddings=embeddings[:count]
    if not np.allclose(hist_times,siglip_times[:count],atol=1e-3):
        raise ValueError("Histogram and SigLIP timestamps are not aligned.")

    histogram_changes=[]
    for index in range(1,count):
        distance=cv2.compareHist(histograms[index-1],histograms[index],cv2.HISTCMP_BHATTACHARYYA)
        histogram_changes.append(float(distance))
    histogram_changes=np.asarray(histogram_changes,dtype=np.float64)
    semantic_changes=1.0-np.sum(embeddings[:-1]*embeddings[1:],axis=1)
    semantic_changes=np.clip(semantic_changes,0.0,2.0)
    hist_scores=robust_normalize(histogram_changes)
    semantic_scores=robust_normalize(semantic_changes)
    combined_scores=0.5*hist_scores+0.5*semantic_scores
    candidate_indices=np.where(combined_scores>=threshold)[0]

    clusters=[]
    for index in candidate_indices:
        index=int(index)
        if not clusters:
            clusters.append([index])
            continue
        previous=clusters[-1][-1]
        if hist_times[index+1]-hist_times[previous+1]<min_scene_duration:
            clusters[-1].append(index)
        else:
            clusters.append([index])

    selected_indices=[]
    for cluster in clusters:
        best=max(cluster,key=lambda i:combined_scores[i])
        selected_indices.append(best)

    boundaries=[float(hist_times[index+1]) for index in selected_indices]
    transitions=[]
    for index in selected_indices:
        transitions.append({
            "timestamp":float(hist_times[index+1]),
            "combined_score":float(combined_scores[index]),
            "histogram_change":float(histogram_changes[index]),
            "semantic_change":float(semantic_changes[index])})

    result={
        "boundaries":boundaries,
        "transitions":transitions,
        "sample_interval":sample_interval,
        "threshold":threshold,
        "min_scene_duration":min_scene_duration}

    with open(cache_path,"w",encoding="utf-8") as file:
        json.dump(result,file,indent=2)
    print(f"Scene cache saved: {cache_path.name}")
    return result

if __name__=="__main__":
    video_path="sample_videos/nasa.mp4"
    result=detect_scene_boundaries(
        video_path,
        sample_interval=0.5,
        threshold=4.0,
        min_scene_duration=1.5)

    print("\nDETECTED SCENE BOUNDARIES\n")
    for transition in result["transitions"]:
        print(
            f"Timestamp: {transition['timestamp']:.2f}s | "
            f"Score: {transition['combined_score']:.2f} | "
            f"Histogram: {transition['histogram_change']:.3f} | "
            f"Semantic: {transition['semantic_change']:.3f}")
    print(f"\nTotal boundaries: {len(result['boundaries'])}")