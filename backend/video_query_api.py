import logging
import math
from typing import Literal
from fastapi import APIRouter,HTTPException
from pydantic import BaseModel,Field
from query_router import QueryIntentRouter

logger=logging.getLogger(__name__)
router=APIRouter(tags=["VideoMind AI"])
intent_router=QueryIntentRouter()

class VideoQueryRequest(BaseModel):
    query:str=Field(min_length=1,max_length=1000)
    top_k:int=Field(default=5,ge=1,le=20)
    fusion_method:Literal[
        "fixed",
        "query_aware",
        "reliability_aware"]="reliability_aware"

class VideoQueryResponse(BaseModel):
    video_id:str
    query:str
    intent:Literal["moment_search","video_qa","hybrid"]
    routing_confidence:float
    routing_reason:str
    answer:str|None=None
    citations:list[dict]=Field(default_factory=list)
    moments:list[dict]=Field(default_factory=list)
    total_results:int=0
    seek_to:float|None=None
    stream_url:str
    autoplay:bool=False
    fusion_method:str
    retrieval_time_seconds:float
    llm_used:bool=False
    llm_model:str|None=None
    llm_grounded:bool|None=None

def valid_timestamp(value)->float|None:
    if isinstance(value,bool):
        return None
    try:
        seconds=float(value)
    except (TypeError,ValueError,OverflowError):
        return None
    if not math.isfinite(seconds) or seconds<0:
        return None
    return seconds

def choose_seek_time(moments:list[dict],citations:list[dict]|None=None)->float|None:
    if citations:
        for citation in citations:
            if not isinstance(citation,dict):
                continue
            start=valid_timestamp(citation.get("start"))
            if start is not None:
                return start
    for moment in moments:
        if not isinstance(moment,dict):
            continue
        start=valid_timestamp(moment.get("start"))
        if start is not None:
            return start
    return None

@router.post(
    "/videos/{video_id}/query",
    response_model=VideoQueryResponse)
async def query_video(video_id:str,request:VideoQueryRequest):
    query=request.query.strip()
    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query cannot be blank.")

    routing=intent_router.classify(query)
    logger.info("VideoMind query intent: %s",routing.intent)

    from main import SearchRequest,search_video_endpoint

    search_request=SearchRequest(
        query=query,
        top_k=request.top_k,
        fusion_method=request.fusion_method)

    search_result=await search_video_endpoint(
        video_id=video_id,
        request=search_request)

    retrieval_result=search_result.model_dump()
    moments=retrieval_result.get("moments",[])
    if not isinstance(moments,list):
        moments=[]

    stream_url=f"/videos/{video_id}/stream"
    seek_to=choose_seek_time(moments)

    if routing.intent=="moment_search":
        return VideoQueryResponse(
            video_id=video_id,
            query=query,
            intent=routing.intent,
            routing_confidence=routing.confidence,
            routing_reason=routing.reason,
            answer=None,
            citations=[],
            moments=moments,
            total_results=len(moments),
            seek_to=seek_to,
            stream_url=stream_url,
            autoplay=False,
            fusion_method=request.fusion_method,
            retrieval_time_seconds=search_result.processing_time_seconds,
            llm_used=False,
            llm_model=None,
            llm_grounded=None)

    from llm_service import LLMService

    llm=LLMService()
    try:
        llm_result=await llm.answer_question(
            question=query,
            retrieval_result=retrieval_result)
    except RuntimeError as error:
        logger.warning(
            "Ollama unavailable for video %s: %s",
            video_id,error)
        raise HTTPException(
            status_code=503,
            detail=str(error)) from error
    except Exception as error:
        logger.exception("VideoMind LLM processing failed.")
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not generate a video answer. "
                "Check server logs.")) from error

    citations=[
        citation.__dict__.copy()
        for citation in llm_result.citations]

    valid_citations=[]
    for citation in citations:
        start=valid_timestamp(citation.get("start"))
        end=valid_timestamp(citation.get("end"))
        if start is None or end is None or end<start:
            continue

        matched=any(
            isinstance(moment,dict)
            and any(
                isinstance(item,dict)
                and item.get("modality") in ("audio","ocr")
                and valid_timestamp(item.get("timestamp"))==start
                and valid_timestamp(item.get("end"))==end
                for item in moment.get("evidence",[]))
            for moment in moments)

        if matched:
            valid_citations.append(citation)

    seek_to=choose_seek_time(moments,valid_citations)

    return VideoQueryResponse(
        video_id=video_id,
        query=query,
        intent=routing.intent,
        routing_confidence=routing.confidence,
        routing_reason=routing.reason,
        answer=llm_result.answer,
        citations=valid_citations,
        moments=moments,
        total_results=len(moments),
        seek_to=seek_to,
        stream_url=stream_url,
        autoplay=False,
        fusion_method=request.fusion_method,
        retrieval_time_seconds=search_result.processing_time_seconds,
        llm_used=True,
        llm_model=llm_result.model,
        llm_grounded=llm_result.grounded)