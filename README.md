
# VideoMoment

**Find the exact moment in a video, just by describing it.**

VideoMoment is a multimodal video retrieval system that lets you search through videos using natural language.

Instead of watching an entire video or manually searching through timestamps, you can describe what you're looking for and jump directly to the relevant moment.

It combines computer vision, speech understanding, multimodal retrieval, and a lightweight local LLM to make video content easier to explore.

---

## Demo

![VideoMoment Demo](assets/demo.gif)

*Upload a video or import one from YouTube, search using natural language, and jump directly to matching moments.*

---

## Features

- **Natural Language Search** — Describe a scene, action, object, or spoken moment to find it.
- **Multimodal Retrieval** — Search using visual content, speech, detected objects, and text visible in video frames.
- **Timestamped Results** — Retrieve ranked video moments with start and end timestamps.
- **Video Uploads and YouTube Imports** — Process locally uploaded MP4 files or import videos through YouTube URLs.
- **Contextual Question Answering** — Ask questions about video content using a lightweight local LLM through Ollama.
- **Interactive Playback** — Click a retrieved moment to jump directly to its timestamp.

---

## How It Works

VideoMoment processes each video through multiple models, with each model contributing a different type of information.

| Component | Technology | Purpose |
|---|---|---|
| Visual Understanding | SigLIP | Match natural-language queries with video frames |
| Speech Recognition | Whisper | Convert spoken audio into timestamped transcripts |
| Semantic Embeddings | MiniLM | Represent and search spoken content |
| Object Detection | YOLO11 | Detect objects appearing in video frames |
| Optical Character Recognition | EasyOCR | Extract visible text from video frames |
| Local LLM | Ollama | Generate contextual answers using retrieved video information |
| Backend | FastAPI | Handle video processing, indexing, retrieval, and API requests |
| Database | PostgreSQL | Store video metadata and application data |

### Video Indexing

When a video is uploaded or imported, VideoMoment extracts information from its visual and audio content.

- SigLIP generates visual embeddings from sampled video frames.
- Whisper transcribes spoken audio.
- MiniLM generates semantic embeddings from transcript segments.
- YOLO11 detects objects in video frames.
- EasyOCR extracts text visible in frames.

The extracted information is cached and reused during retrieval, avoiding repeated full-video processing for every search.

### Moment Retrieval

When a user enters a natural-language query, VideoMoment searches across the available modalities.

The retrieval pipeline combines evidence from visual, audio, object, and OCR information, ranks relevant candidates, and returns timestamped video moments.

### Contextual Question Answering

VideoMoment integrates a lightweight local LLM through Ollama.

The LLM uses retrieved video context to generate natural-language answers while the multimodal retrieval system remains responsible for identifying relevant video evidence.

---

## Architecture

```text
                 VIDEO INPUT
                      |
             +--------+--------+
             |                 |
         MP4 Upload       YouTube URL
             |                 |
             +--------+--------+
                      |
                      v
                VIDEO INDEXING
                      |
       +--------------+--------------+
       |              |              |
       v              v              v
     SigLIP         Whisper         YOLO11
       |              |              |
     Visual          MiniLM        Objects
   Embeddings          |              |
       |           Transcript       EasyOCR
       |           Embeddings         |
       |              |          Visible Text
       |              |              |
       +--------------+--------------+
                      |
                      v
               CACHED FEATURES
                      |
                      v
              MULTIMODAL SEARCH
                      ^
                      |
              Natural Language Query
                      |
                      v
             Retrieval + Ranking
                      |
             +--------+--------+
             |                 |
             v                 v
       Matching Moments     Ollama LLM
             |                 |
             v                 v
       Video Timestamps   Contextual Answers
```

---

## Project Structure

```text
VideoMomentRetrevial/
|
|-- backend/
|   |-- main.py
|   |-- multimodal_retrieval.py
|   |-- temporal_localization.py
|   |-- video_frames.py
|   |-- ...
|
|-- frontend/
|   |-- index.html
|   |-- style.css
|   |-- script.js
|
|-- assets/
|   |-- demo.gif
|
|-- README.md
```

---

## Running Locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd VideoMomentRetrevial
```

### 2. Create a Python virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies from the requirements file.

### 3. Configure PostgreSQL

Create a PostgreSQL database and configure the backend database connection.

### 4. Configure Ollama

Install [Ollama](https://ollama.com/) and pull the lightweight model configured for the project.

Ensure Ollama is running to use contextual question answering.

### 5. Start the backend

```powershell
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

API documentation:

http://127.0.0.1:8000/docs

### 6. Start the frontend

Open a separate terminal:

```powershell
cd frontend
python -m http.server 5500
```

Open http://127.0.0.1:5500 in your browser.

You can now upload a video, import a YouTube URL, and search for relevant moments.

---

## Current Status

VideoMoment is under active development.

The project includes multimodal video indexing and retrieval, a FastAPI backend, PostgreSQL integration, a web interface, and local LLM support through Ollama.

Current work focuses on improving retrieval quality, processing performance, and the overall video search experience.

---

*Finding a moment in a video should be as simple as describing it.*
