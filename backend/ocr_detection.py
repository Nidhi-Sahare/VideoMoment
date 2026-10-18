import logging
import time
from pathlib import Path

import cv2
import easyocr
import numpy as np
import torch

OCR_LANGUAGES=["en"]
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
ocr_reader=None
logger=logging.getLogger(__name__)

def get_device_info():
    info={
        "device":DEVICE,
        "cuda_available":torch.cuda.is_available(),
        "languages":OCR_LANGUAGES,
        "reader_loaded":ocr_reader is not None,}
    if torch.cuda.is_available():
        info["gpu_name"]=torch.cuda.get_device_name(0)
    return info

def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        print(f"[OCR] Loading EasyOCR on {DEVICE}...",flush=True)
        started=time.perf_counter()
        ocr_reader=easyocr.Reader(OCR_LANGUAGES,gpu=(DEVICE=="cuda"))
        print(f"[OCR] Model loaded in {time.perf_counter()-started:.2f}s",flush=True)
    return ocr_reader

def extract_text(video_path,sample_interval=0.5,progress_callback=None):
    video_path=Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not np.isfinite(sample_interval) or sample_interval<=0:
        raise ValueError("sample_interval must be greater than zero.")
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Unable to open video: {video_path}")
    started=time.perf_counter()
    try:
        fps=cap.get(cv2.CAP_PROP_FPS)
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps<=0:
            raise ValueError("Could not determine video FPS.")
        if total_frames<=0:
            raise ValueError("Could not determine video frame count.")
        frame_interval=max(1,round(fps*sample_interval))
        sampled_frames=range(0,total_frames,frame_interval)
        total_samples=len(sampled_frames)
        print(f"[OCR] Video: {video_path.name}",flush=True)
        print(f"[OCR] FPS: {fps:.2f}, frames: {total_frames}, sampled frames: {total_samples}",flush=True)
        reader=get_ocr_reader()
        results=[]
        for sample_number,frame_number in enumerate(sampled_frames,start=1):
            frame_started=time.perf_counter()
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_number)
            ret,frame=cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read video frame {frame_number} at {frame_number/fps:.2f}s")
            timestamp=frame_number/fps
            detections=reader.readtext(frame)
            texts=[]
            for detection in detections:
                bbox,text,confidence=detection
                if not text.strip():
                    continue
                texts.append({
                    "text":str(text),
                    "confidence":float(confidence),
                    "bbox":bbox.tolist() if isinstance(bbox,np.ndarray) else bbox,})
            results.append({"timestamp":float(timestamp),"texts":texts})
            frame_seconds=time.perf_counter()-frame_started
            elapsed_seconds=time.perf_counter()-started
            print(f"[OCR] {sample_number}/{total_samples} | t={timestamp:.2f}s | detections={len(texts)} | frame_time={frame_seconds:.2f}s",flush=True)
            if progress_callback is not None:
                progress_callback({
                    "processed":sample_number,
                    "total":total_samples,
                    "timestamp":float(timestamp),
                    "elapsed_seconds":elapsed_seconds,
                    "frame_seconds":frame_seconds,})
        print(f"[OCR] Completed {len(results)} frames in {time.perf_counter()-started:.2f}s",flush=True)
        return results
    except Exception:
        logger.exception("OCR extraction failed for %s",video_path)
        raise
    finally:
        cap.release()

if __name__=="__main__":
    video_path="sample_videos/nasa.mp4"
    print("\nEASYOCR DEVICE CONFIGURATION")
    print(get_device_info())

    def show_progress(event):
        print(f"[PROGRESS] {event['processed']}/{event['total']}",flush=True)

    results=extract_text(video_path,sample_interval=3.0,progress_callback=show_progress)
    print("\nOCR RESULTS AROUND ARTEMIS II\n")
    print("Frames analyzed:",len(results))
    for result in results:
        if 18<=result["timestamp"]<=25:
            print(f"\nTimestamp: {result['timestamp']:.2f}s")
            if not result["texts"]:
                print("No text detected.")
                continue
            for item in result["texts"]:
                print(f"Text: {item['text']} | Confidence: {item['confidence']:.3f}")