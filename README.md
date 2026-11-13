# VideoMoment: Natural-Language Semantic Video Retrieval Engine

**Tech Stack:** Python 3.10+ | PyTorch | OpenCV | CLIP / SigLIP | FAISS | Streamlit

> **Search hours of unindexed video using plain English.** VideoMoment indexes visual scenes and spoken context into a multimodal vector space, pinpointing the exact timestamp of any action, object, or event in milliseconds.


## Demo

<p align="center">
  <img src="assets/demo.gif" alt="VideoMoment Live Demo" width="850"/>
</p>

*Natural-language query retrieving exact video timestamps in real time.*

**Live App:** [Insert Hugging Face / Streamlit Live Link]

---

## Why VideoMoment?
Scrubbing through long lecture recordings, CCTV footage, or raw video clips manually is slow and error-prone. Standard keyword search fails when events are not explicitly mentioned in transcripts. **VideoMoment** solves this by aligning frame-level visual semantics with natural-language queries, enabling instant zero-shot moment localization without custom fine-tuning.

---

## System Architecture

```text
[Raw Video Input]
       │
       ├──► Smart Keyframe & Scene-Change Extractor (OpenCV / PySceneDetect)
       │          │
       │          ▼
       │    Vision-Language Encoder (CLIP / SigLIP) ──► Frame Embeddings
       │
       └──► Audio Track Extractor ──► Whisper ASR ────► Transcript Embeddings
                                                               │
                                                               ▼
                                              [Vector Database: FAISS / ChromaDB]
                                                               │
[User Query: "Person walking a dog in red jacket"] ────────────┤
                                                               ▼
                                          Temporal Non-Max Suppression & Ranking
                                                               │
                                                               ▼
                                       [Top-K Exact Timestamps + Confidence Scores]
```

---

## Key Features
* **Adaptive Scene-Aware Sampling:** Skips redundant static frames using histogram and embedding deltas, cutting indexing time by **[X]%** compared to uniform FPS sampling.
* **Multimodal Fusion (Visual + Audio):** Combines visual frame embeddings with speech-to-text timestamps so queries work whether an event is *seen* or *spoken*.
* **Sub-Second Vector Search:** Powered by **FAISS / ChromaDB** cosine similarity search with temporal smoothing to return clean, non-overlapping video intervals.
* **Interactive Web UI:** Built with **Streamlit / Gradio** featuring clickable timestamp chips that jump the video player directly to the matched moment.

---

## Performance Benchmarks
*Tested on [e.g., 1-hour 1080p MP4 video / ActivityNet subset] on [e.g., NVIDIA T4 GPU / CPU]:*

| Metric | Uniform Sampling (1 FPS) | VideoMoment (Adaptive Scene Sampling) |
| :--- | :---: | :---: |
| **Indexing Speed** | [e.g., 142 sec] | **[e.g., 48 sec (2.9x faster)]** |
| **Query Retrieval Latency** | [e.g., 85 ms] | **[e.g., 32 ms]** |
| **Recall@1 (IoU > 0.5)** | [e.g., 64.2%] | **[e.g., 71.8%]** |

---

## Quickstart (Run Locally)

### 1. Clone the Repository
```bash
git clone https://github.com[your-username]/VideoMoment.git
cd VideoMoment
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the App
```bash
streamlit run app.py
# Or: python main.py --video path/to/video.mp4 --query "your search query"
```

---

## Project Structure
```text
VideoMoment/
├── app.py                 # Interactive Streamlit/Gradio frontend
├── src/
│   ├── sampler.py         # Adaptive keyframe & scene detection
│   ├── encoder.py         # Multimodal CLIP/SigLIP + Whisper feature extraction
│   ├── retriever.py       # Vector indexing & temporal interval merging
│   └── utils.py           # Video timestamp slicing & helper functions
├── assets/
│   └── demo.gif           # Walkthrough demonstration GIF
├── requirements.txt       # Pinned dependencies
└── README.md
```

---

## Future Improvements
* [ ] Export vision encoder to **ONNX / TensorRT FP16** for edge deployment.
* [ ] Add **Qwen2.5-VL** re-ranking on top-3 candidate clips for complex multi-step action queries.
* [ ] Add **Dockerfile** for one-command cloud deployment.

---

## Author
**Nidhi Sahare**  
BE in Artificial Intelligence & Machine Learning | B. M. S. College of Engineering (BMSCE)  
[LinkedIn](https://linkedin.com) | [GitHub](https://github.com[your-username])
