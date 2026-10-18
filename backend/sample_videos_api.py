from fastapi import APIRouter,HTTPException
from pydantic import BaseModel,Field
from sample_videos import SAMPLE_VIDEOS

router=APIRouter(prefix="/videos",tags=["Sample Videos"])

class SampleVideoResponse(BaseModel):
    video_id:str
    title:str
    description:str
    category:str
    example_queries:list[str]=Field(default_factory=list)
    stream_url:str
    status_url:str
    search_url:str

class SampleVideoListResponse(BaseModel):
    total:int
    videos:list[SampleVideoResponse]

def build_sample_response(sample:dict)->SampleVideoResponse:
    video_id=sample["video_id"]
    return SampleVideoResponse(
        video_id=video_id,
        title=sample["title"],
        description=sample["description"],
        category=sample["category"],
        example_queries=sample.get("example_queries",[]),
        stream_url=f"/videos/{video_id}/stream",
        status_url=f"/videos/{video_id}/status",
        search_url=f"/videos/{video_id}/search",)

@router.get("/samples",response_model=SampleVideoListResponse)
def list_sample_videos():
    videos=[build_sample_response(sample) for sample in SAMPLE_VIDEOS]
    return SampleVideoListResponse(total=len(videos),videos=videos)

@router.get("/samples/{video_id}",response_model=SampleVideoResponse)
def get_sample_video(video_id:str):
    for sample in SAMPLE_VIDEOS:
        if sample["video_id"]==video_id:
            return build_sample_response(sample)
    raise HTTPException(status_code=404,detail="Sample video not found.")