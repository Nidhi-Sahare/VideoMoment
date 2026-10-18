from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT=Path(__file__).resolve().parent.parent
MODEL_PATH=str(PROJECT_ROOT/"yolo11n.pt")
DEVICE="cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_BATCH_SIZE=16 if DEVICE.startswith("cuda") else 4
model=None

def get_yolo_model():
    global model
    if model is None:
        print(f"Loading YOLO model on {DEVICE}...",flush=True)
        model=YOLO(MODEL_PATH)
    return model

def get_device_info():
    info={
        "device":DEVICE,
        "cuda_available":torch.cuda.is_available(),
        "batch_size":DEFAULT_BATCH_SIZE,
        "model":MODEL_PATH}
    if torch.cuda.is_available():
        info["gpu_name"]=torch.cuda.get_device_name(0)
    return info

def detect_objects(video_path,sample_interval=0.5,confidence_threshold=0.4,batch_size=None):
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if sample_interval<=0:
        raise ValueError("sample_interval must be greater than zero.")
    if not 0<=confidence_threshold<=1:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    if batch_size is None:
        batch_size=DEFAULT_BATCH_SIZE
    if not isinstance(batch_size,int) or batch_size<=0:
        raise ValueError("batch_size must be a positive integer.")

    detector=get_yolo_model()
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps=cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=0:
        cap.release()
        raise ValueError("Invalid video FPS.")

    frame_interval=max(1,round(fps*sample_interval))
    results=[]
    batch_frames=[]
    batch_timestamps=[]
    frame_number=0

    def process_batch():
        if not batch_frames:
            return
        predictions=detector.predict(
            source=batch_frames,
            device=DEVICE,
            conf=confidence_threshold,
            verbose=False,
            stream=False)

        if len(predictions)!=len(batch_timestamps):
            raise RuntimeError(
                "YOLO prediction count does not match "
                "the sampled frame count.")

        for timestamp,prediction in zip(batch_timestamps,predictions):
            objects=[]
            for box in prediction.boxes:
                confidence=float(box.conf[0])
                if confidence<confidence_threshold:
                    continue
                class_id=int(box.cls[0])
                class_name=detector.names[class_id]
                bbox=[float(value) for value in box.xyxy[0].tolist()]
                objects.append({
                    "name":class_name,
                    "confidence":confidence,
                    "bbox":bbox})
            results.append({
                "timestamp":float(timestamp),
                "objects":objects})

        batch_frames.clear()
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
                        f"Failed to retrieve YOLO frame {frame_number}")
                timestamp=frame_number/fps
                batch_frames.append(frame)
                batch_timestamps.append(timestamp)
                if len(batch_frames)>=batch_size:
                    process_batch()
            frame_number+=1
        process_batch()
    finally:
        cap.release()

    return results

if __name__=="__main__":
    video_path="sample_videos/nasa.mp4"
    print("\nYOLO DEVICE CONFIGURATION")
    print(get_device_info())

    results=detect_objects(
        video_path,
        sample_interval=1.0,
        confidence_threshold=0.4)

    print("\nYOLO OBJECT DETECTION RESULTS\n")
    print("Frames analyzed:",len(results))

    for result in results:
        if 18<=result["timestamp"]<=25:
            print(result)