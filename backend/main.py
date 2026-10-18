import json
import logging
import threading
from datetime import datetime,timezone
from pathlib import Path
from typing import Annotated,Literal
from uuid import UUID,uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from sqlalchemy import select,update
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from database import SessionLocal
from models import Video,IndexingJob,SearchHistory
from sample_videos_api import router as sample_videos_router
from video_url_import import router as video_url_import_router

from deployment_config import (
    IS_DEMO,
    FRONTEND_ORIGINS,
    require_local_mode,
    require_demo_video_access,
)

BASE_DIR=Path(__file__).resolve().parent
PROJECT_ROOT=BASE_DIR.parent
UPLOAD_DIR=PROJECT_ROOT/"uploads"
METADATA_DIR=PROJECT_ROOT/"video_metadata"
CACHE_DIR=PROJECT_ROOT/"video_cache"
for directory in (UPLOAD_DIR,METADATA_DIR,CACHE_DIR):
    directory.mkdir(parents=True,exist_ok=True)

ALLOWED_EXTENSIONS={".mp4",".webm",".avi",".mkv"}
MAX_UPLOAD_BYTES=2*1024*1024*1024
CHUNK_SIZE=1024*1024
logger=logging.getLogger(__name__)
metadata_lock=threading.RLock()
indexing_lock=threading.Lock()
search_lock=threading.Lock()
active_index_jobs=set()

def recover_interrupted_indexing_jobs():
    logger.info("Checking for interrupted indexing jobs...")
    with SessionLocal() as db:
        interrupted_videos=db.scalars(select(Video).where(Video.status.in_(["queued","indexing"]))).all()
        interrupted_jobs=db.scalars(select(IndexingJob).where(IndexingJob.status.in_(["queued","indexing"]))).all()
        for video in interrupted_videos:
            video.status="interrupted"
            video.indexed=False
            video.stage="interrupted"
            video.message="Indexing was interrupted by a previous server shutdown. Start indexing again."
            video.error=None
        for job in interrupted_jobs:
            job.status="interrupted"
            job.stage="interrupted"
            job.message="Indexing interrupted by server shutdown."
            job.completed_at=utc_now()
        db.commit()
    logger.info("Recovered %d interrupted videos and %d interrupted indexing jobs.",len(interrupted_videos),len(interrupted_jobs))

app=FastAPI(title="Video Moment Retrieval API",description="Multimodal video retrieval with PostgreSQL persistence.",version="0.4.0")
FRONTEND_DIR=PROJECT_ROOT/"frontend"
app.mount("/static",StaticFiles(directory=str(FRONTEND_DIR)),name="frontend")

@app.get("/app",include_in_schema=False)
def serve_frontend():
    return FileResponse(FRONTEND_DIR/"index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from video_query_api import router as video_query_router
app.include_router(video_query_router,dependencies=[Depends(require_local_mode)] if IS_DEMO else [],)
app.include_router(sample_videos_router)
app.include_router(
    video_url_import_router,
    dependencies=[Depends(require_local_mode)] if IS_DEMO else [],
)
@app.on_event("startup")
def startup_recovery():
    recover_interrupted_indexing_jobs()

class HealthResponse(BaseModel):
    status:str
    service:str
    version:str

class VideoResponse(BaseModel):
    video_id:str
    filename:str
    status:str
    uploaded_at:str

class VideoStatusResponse(BaseModel):
    video_id:str
    filename:str
    status:str
    indexed:bool
    progress:int=0
    stage:str|None=None
    message:str|None=None
    error:str|None=None
    indexing_summary:dict|None=None

class IndexResponse(BaseModel):
    video_id:str
    status:str
    message:str

class SearchRequest(BaseModel):
    query:str=Field(min_length=1,max_length=1000)
    top_k:int=Field(default=5,ge=1,le=20)
    fusion_method:Literal["fixed","query_aware","reliability_aware"]="reliability_aware"

class SearchResponse(BaseModel):
    video_id:str
    query:str
    fusion_method:str
    total_results:int
    processing_time_seconds:float
    moments:list[dict]

def utc_now():
    return datetime.now(timezone.utc)

def validate_video_id(video_id:str)->str:
    try:
        parsed=UUID(hex=video_id)
    except (ValueError,AttributeError,TypeError):
        raise HTTPException(status_code=400,detail="Invalid video ID.")
    if parsed.hex!=video_id:
        raise HTTPException(status_code=400,detail="Invalid video ID format.")
    return video_id

def datetime_to_string(value):
    if value is None:
        return utc_now().isoformat()
    return value.isoformat()

def video_to_metadata(video:Video)->dict:
    path=Path(video.file_path)
    return {
        "video_id":video.id,
        "filename":video.filename,
        "extension":path.suffix.lower(),
        "size_bytes":video.file_size,
        "status":video.status,
        "indexed":video.indexed,
        "progress":video.progress,
        "stage":video.stage,
        "message":video.message,
        "error":video.error,
        "indexing_summary":video.indexing_summary,
        "uploaded_at":datetime_to_string(video.uploaded_at),}

def resolve_video_file(metadata:dict)->Path:
    video_id=validate_video_id(metadata["video_id"])
    extension=metadata.get("extension")
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=500,detail="Invalid stored video extension.")
    path=UPLOAD_DIR/f"{video_id}{extension}"
    if not path.is_file():
        raise HTTPException(status_code=404,detail="Uploaded video file is missing.")
    return path

def import_legacy_video(video_id:str)->bool:
    validate_video_id(video_id)
    legacy_path=METADATA_DIR/f"{video_id}.json"
    if not legacy_path.is_file():
        return False
    with metadata_lock:
        with SessionLocal() as db:
            existing=db.get(Video,video_id)
            if existing is not None:
                return True
            try:
                with legacy_path.open("r",encoding="utf-8") as file:
                    metadata=json.load(file)
            except (OSError,json.JSONDecodeError):
                logger.exception("Failed reading legacy metadata: %s",video_id)
                raise HTTPException(status_code=500,detail="Legacy metadata is invalid.")
            if metadata.get("video_id")!=video_id:
                raise HTTPException(status_code=500,detail="Legacy video ID mismatch.")
            extension=metadata.get("extension")
            if extension not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=500,detail="Invalid legacy extension.")
            video_path=UPLOAD_DIR/f"{video_id}{extension}"
            if not video_path.is_file():
                raise HTTPException(status_code=404,detail="Legacy video file is missing.")
            uploaded_at=utc_now()
            try:
                if metadata.get("uploaded_at"):
                    uploaded_at=datetime.fromisoformat(metadata["uploaded_at"])
            except (ValueError,TypeError):
                pass
            video=Video(
                id=video_id,
                filename=metadata.get("filename",video_path.name),
                file_path=str(video_path),
                file_size=metadata.get("size_bytes",video_path.stat().st_size),
                status=metadata.get("status","uploaded"),
                indexed=bool(metadata.get("indexed",False)),
                progress=int(metadata.get("progress",0)),
                stage=metadata.get("stage"),
                message=metadata.get("message"),
                error=metadata.get("error"),
                indexing_summary=metadata.get("indexing_summary"),
                uploaded_at=uploaded_at,)
            if video.status in ("queued","indexing"):
                video.status="interrupted"
                video.indexed=False
                video.stage="interrupted"
                video.message="Previous indexing was interrupted. Start indexing again."
            db.add(video)
            db.commit()
            logger.info("Imported legacy video %s",video_id)
            return True

def load_metadata(video_id:str)->dict:
    validate_video_id(video_id)
    with SessionLocal() as db:
        video=db.get(Video,video_id)
        if video is not None:
            return video_to_metadata(video)
    if import_legacy_video(video_id):
        with SessionLocal() as db:
            video=db.get(Video,video_id)
            if video is not None:
                return video_to_metadata(video)
    raise HTTPException(status_code=404,detail="Video not found.")

def update_metadata(video_id:str,**changes)->dict:
    validate_video_id(video_id)
    allowed_fields={"status","indexed","progress","stage","message","error","indexing_summary"}
    with metadata_lock:
        with SessionLocal() as db:
            video=db.get(Video,video_id)
            if video is None:
                raise HTTPException(status_code=404,detail="Video not found.")
            for key,value in changes.items():
                if key not in allowed_fields:
                    raise ValueError(f"Unsupported metadata field: {key}")
                setattr(video,key,value)
            db.commit()
            db.refresh(video)
            return video_to_metadata(video)

def get_latest_indexing_job(db,video_id:str):
    return db.scalars(select(IndexingJob).where(IndexingJob.video_id==video_id).order_by(IndexingJob.id.desc()).limit(1)).first()

def update_indexing_job(video_id:str,**changes):
    with SessionLocal() as db:
        job=get_latest_indexing_job(db,video_id)
        if job is None:
            return
        allowed_fields={"status","progress","stage","message","error","started_at","completed_at"}
        for key,value in changes.items():
            if key not in allowed_fields:
                raise ValueError(f"Unsupported job field: {key}")
            setattr(job,key,value)
        db.commit()

def get_existing_index_manifest(video_path:Path):
    from video_index import get_video_id
    cache_id=get_video_id(str(video_path))
    manifest_path=CACHE_DIR/f"{cache_id}_index_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open("r",encoding="utf-8") as file:
            manifest=json.load(file)
        if manifest.get("status")=="indexed" and manifest.get("index_version")==1 and manifest.get("video_cache_id")==cache_id:
            return manifest
    except (OSError,json.JSONDecodeError):
        logger.exception("Could not read index manifest.")
    return None

def sync_existing_index(video_id:str)->dict:
    metadata=load_metadata(video_id)
    if metadata.get("indexed"):
        return metadata
    if metadata.get("status") in ("queued","indexing"):
        return metadata
    video_path=resolve_video_file(metadata)
    manifest=get_existing_index_manifest(video_path)
    if manifest is None:
        return metadata
    return update_metadata(video_id,status="indexed",indexed=True,progress=100,stage="completed",message="Existing feature index found.",error=None,indexing_summary=manifest)

def run_indexing_job(video_id:str):
    with indexing_lock:
        try:
            metadata=load_metadata(video_id)
            video_path=resolve_video_file(metadata)
            update_metadata(video_id,status="indexing",indexed=False,progress=0,stage="starting",message="Preparing video indexing.",error=None)
            update_indexing_job(video_id,status="indexing",progress=0,stage="starting",message="Preparing video indexing.",started_at=utc_now())
            from video_index import index_video

            def progress_callback(event):
                progress=int(event.get("progress",0))
                stage=event.get("stage","indexing")
                message=event.get("message","Indexing video.")
                update_metadata(video_id,status="indexing",indexed=False,progress=progress,stage=stage,message=message,error=None)
                update_indexing_job(video_id,status="indexing",progress=progress,stage=stage,message=message)

            summary=index_video(str(video_path),progress_callback=progress_callback)
            update_metadata(video_id,status="indexed",indexed=True,progress=100,stage="completed",message="Video indexing completed.",error=None,indexing_summary=summary)
            update_indexing_job(video_id,status="indexed",progress=100,stage="completed",message="Video indexing completed.",error=None,completed_at=utc_now())
            logger.info("Indexing completed: %s",video_id)
        except Exception as error:
            logger.exception("Indexing failed: %s",video_id)
            try:
                update_metadata(video_id,status="failed",indexed=False,stage="failed",message="Video indexing failed.",error=str(error))
                update_indexing_job(video_id,status="failed",stage="failed",message="Video indexing failed.",error=str(error),completed_at=utc_now())
            except Exception:
                logger.exception("Could not save indexing failure.")
        finally:
            with metadata_lock:
                active_index_jobs.discard(video_id)

def start_indexing_job(video_id:str):
    with metadata_lock:
        if video_id in active_index_jobs:
            return False
        with SessionLocal() as db:
            video=db.get(Video,video_id)
            if video is None:
                raise HTTPException(status_code=404,detail="Video not found.")
            if video.status in ("queued","indexing"):
                return False
            job=IndexingJob(video_id=video_id,status="queued",progress=0,stage="queued",message="Waiting for indexing worker.")
            video.status="queued"
            video.indexed=False
            video.progress=0
            video.stage="queued"
            video.message="Waiting for indexing worker."
            video.error=None
            db.add(job)
            db.commit()
        active_index_jobs.add(video_id)
        try:
            worker=threading.Thread(target=run_indexing_job,args=(video_id,),daemon=True,name=f"index-{video_id[:8]}")
            worker.start()
            return True
        except Exception as error:
            active_index_jobs.discard(video_id)
            update_metadata(video_id,status="failed",indexed=False,stage="failed",message="Could not start indexing.",error=str(error))
            update_indexing_job(video_id,status="failed",stage="failed",error=str(error),completed_at=utc_now())
            raise

@app.get("/health",response_model=HealthResponse,tags=["System"])
def health():
    return HealthResponse(status="ok",service="video-moment-retrieval",version=app.version)

@app.post("/videos/upload",response_model=VideoResponse,status_code=201,tags=["Videos"])
async def upload_video(file:Annotated[UploadFile,File()]):
    require_local_mode()
    original_filename=Path(file.filename or "").name
    extension=Path(original_filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400,detail="Unsupported video format.")
    video_id=uuid4().hex
    destination=UPLOAD_DIR/f"{video_id}{extension}"
    total_bytes=0
    try:
        with destination.open("wb") as output:
            while True:
                chunk=await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes+=len(chunk)
                if total_bytes>MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413,detail="Video exceeds 2 GB limit.")
                output.write(chunk)
        if total_bytes==0:
            raise HTTPException(status_code=400,detail="Uploaded file is empty.")

        def save_video_record():
            with SessionLocal() as db:
                video=Video(
                    id=video_id,
                    filename=original_filename,
                    file_path=str(destination),
                    file_size=total_bytes,
                    status="uploaded",
                    indexed=False,
                    progress=0,
                    stage="uploaded",
                    message="Video uploaded. Start indexing.",)
                db.add(video)
                db.commit()
                db.refresh(video)
                return datetime_to_string(video.uploaded_at)

        uploaded_at=await run_in_threadpool(save_video_record)
        return VideoResponse(video_id=video_id,filename=original_filename,status="uploaded",uploaded_at=uploaded_at)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

@app.post("/videos/{video_id}/index",response_model=IndexResponse,status_code=202,tags=["Indexing"])
def index_video_endpoint(video_id:str):
    require_local_mode()
    metadata=load_metadata(video_id)
    resolve_video_file(metadata)
    metadata=sync_existing_index(video_id)
    if metadata.get("indexed"):
        return IndexResponse(video_id=video_id,status="indexed",message="Video is already indexed.")
    with metadata_lock:
        metadata=load_metadata(video_id)
        if video_id in active_index_jobs or metadata.get("status") in ("queued","indexing"):
            return IndexResponse(video_id=video_id,status=metadata["status"],message="Indexing is already in progress.")
        started=start_indexing_job(video_id)
    return IndexResponse(video_id=video_id,status="queued" if started else "indexing",message="Background indexing started.")

@app.get("/videos/{video_id}/status",response_model=VideoStatusResponse,tags=["Videos"])
def get_video_status(video_id:str):
    require_demo_video_access(video_id)
    metadata=sync_existing_index(video_id)
    return VideoStatusResponse(
        video_id=metadata["video_id"],
        filename=metadata["filename"],
        status=metadata.get("status","uploaded"),
        indexed=metadata.get("indexed",False),
        progress=metadata.get("progress",0),
        stage=metadata.get("stage"),
        message=metadata.get("message"),
        error=metadata.get("error"),
        indexing_summary=metadata.get("indexing_summary"),)

@app.post("/videos/{video_id}/search",response_model=SearchResponse,tags=["Search"])
async def search_video_endpoint(video_id:str,request:SearchRequest):
    require_demo_video_access(video_id)
    metadata=await run_in_threadpool(sync_existing_index,video_id)
    if not metadata.get("indexed",False):
        raise HTTPException(status_code=409,detail="Video is not indexed yet. Start indexing and wait until it completes.")
    video_path=resolve_video_file(metadata)
    if not request.query.strip():
        raise HTTPException(status_code=422,detail="Query cannot be blank.")
    try:
        from retrieval_service import search_video

        def perform_search():
            with search_lock:
                return search_video(video_path=str(video_path),query=request.query,top_k=request.top_k,fusion_method=request.fusion_method)

        result=await run_in_threadpool(perform_search)

        def save_search_history():
            with SessionLocal() as db:
                record=SearchHistory(
                    video_id=video_id,
                    query=result["query"],
                    fusion_method=result["fusion_method"],
                    top_k=request.top_k,
                    processing_time_seconds=result["processing_time_seconds"],
                    total_results=result["total_results"],
                    results=result["moments"],)
                db.add(record)
                db.commit()

        await run_in_threadpool(save_search_history)
        return SearchResponse(
            video_id=video_id,
            query=result["query"],
            fusion_method=result["fusion_method"],
            total_results=result["total_results"],
            processing_time_seconds=result["processing_time_seconds"],
            moments=result["moments"],)
    except (ValueError,FileNotFoundError) as error:
        raise HTTPException(status_code=400,detail=str(error)) from error
    except Exception as error:
        logger.exception("Video search failed: %s",video_id)
        raise HTTPException(status_code=500,detail="Video search failed. Check server logs.") from error

@app.get("/videos/{video_id}/stream",tags=["Videos"])
def stream_video(video_id:str):
    require_demo_video_access(video_id)
    metadata=load_metadata(video_id)
    video_path=resolve_video_file(metadata)
    media_types={
        ".mp4":"video/mp4",
        ".webm":"video/webm",
        ".avi":"video/x-msvideo",
        ".mkv":"video/x-matroska",}
    return FileResponse(path=video_path,media_type=media_types[metadata["extension"]],filename=metadata["filename"],content_disposition_type="inline")

@app.get("/",tags=["System"])
def root():
    return {
        "message":"Video Moment Retrieval API",
        "version":app.version,
        "database":"PostgreSQL",
        "documentation":"/docs",
        "endpoints":[
            "/health",
            "/videos/upload",
            "/videos/{video_id}/index",
            "/videos/{video_id}/status",
            "/videos/{video_id}/search",
            "/videos/{video_id}/stream",],}