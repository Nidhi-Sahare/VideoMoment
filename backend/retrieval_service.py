import math
import time
from pathlib import Path
from typing import Any
from multimodal_retrieval import get_multimodal_results
from temporal_localization import localize_moments
import re
from query_analysis import analyze_query

SUPPORTED_FUSION_METHODS={"fixed","query_aware","reliability_aware"}
DEFAULT_FUSION_METHOD="reliability_aware"
DEFAULT_TOP_K=5
MAX_TOP_K=20

def validate_search_request(video_path:str,query:str,top_k:int,fusion_method:str)->str:
    path=Path(video_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    if not isinstance(query,str) or not query.strip():
        raise ValueError("Search query must be a non-empty string.")
    if not isinstance(top_k,int) or isinstance(top_k,bool):
        raise ValueError("top_k must be an integer.")
    if not 1<=top_k<=MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")
    if fusion_method not in SUPPORTED_FUSION_METHODS:
        raise ValueError("fusion_method must be one of: "+", ".join(sorted(SUPPORTED_FUSION_METHODS)))
    return str(path)

def json_safe(value:Any)->Any:
    if value is None:
        return None
    if isinstance(value,bool):
        return value
    if isinstance(value,str):
        return value
    if isinstance(value,int):
        return int(value)
    if isinstance(value,float):
        return value if math.isfinite(value) else None
    if hasattr(value,"item"):
        try:
            return json_safe(value.item())
        except (TypeError,ValueError,AttributeError):
            pass
    if isinstance(value,dict):
        return {str(key):json_safe(item) for key,item in value.items()}
    if isinstance(value,(list,tuple,set)):
        return [json_safe(item) for item in value]
    if hasattr(value,"tolist"):
        try:
            return json_safe(value.tolist())
        except (TypeError,ValueError,AttributeError):
            pass
    return str(value)

def safe_number(value:Any,default:float=0.0)->float:
    try:
        number=float(value)
    except (TypeError,ValueError,OverflowError):
        return default
    if not math.isfinite(number):
        return default
    return number

def extract_interval(moment:dict)->tuple[float,float]:
    for field in ("improved_interval","shot_visual_interval","visual_interval","audio_interval","baseline_interval"):
        interval=moment.get(field)
        if not isinstance(interval,(list,tuple)) or len(interval)!=2 or interval[0] is None or interval[1] is None:
            continue
        start=safe_number(interval[0],-1.0)
        end=safe_number(interval[1],-1.0)
        if start>=0 and end>start:
            return start,end
    peak=max(0.0,safe_number(moment.get("peak_timestamp",0.0)))
    return peak,peak+1.0

def format_evidence(evidence:Any)->list[dict]:
    if not isinstance(evidence,list):
        return []
    formatted=[]
    for item in evidence:
        if not isinstance(item,dict):
            continue
        formatted.append({
            "modality":str(item.get("modality","unknown")),
            "timestamp":safe_number(item.get("timestamp",0.0)),
            "end":safe_number(item["end"]) if item.get("end") is not None else None,
            "score":safe_number(item.get("score",0.0)),
            "description":str(item.get("description","")),})
    return formatted

def format_moment(moment:dict,rank:int)->dict:
    start,end=extract_interval(moment)
    evidence=format_evidence(moment.get("evidence",[]))
    modalities=moment.get("modalities",[])
    if not isinstance(modalities,(list,tuple,set)):
        modalities=[]
    return {
        "rank":rank,
        "start":round(start,3),
        "end":round(end,3),
        "duration":round(end-start,3),
        "peak_timestamp":round(safe_number(moment.get("peak_timestamp",start)),3),
        "relevance_score":round(safe_number(moment.get("relevance_score",0.0)),6),
        "modalities":sorted(str(modality) for modality in modalities),
        "localization_mode":moment.get("localization_mode"),
        "localization_reason":moment.get("localization_reason"),
        "shot_constrained":moment.get("shot_constrained"),
        "evidence":evidence,}

def remove_duplicate_moments(localized_results:list[dict],overlap_threshold:float=0.90)->list[dict]:
    def interval_iou(first,second):
        start_a,end_a=extract_interval(first)
        start_b,end_b=extract_interval(second)
        intersection=max(0.0,min(end_a,end_b)-max(start_a,start_b))
        union=max(end_a,end_b)-min(start_a,start_b)
        if union<=0:
            return 0.0
        return intersection/union

    ranked=sorted(localized_results,key=lambda moment:safe_number(moment.get("relevance_score",0.0)),reverse=True)
    unique=[]
    for candidate in ranked:
        duplicate=any(interval_iou(candidate,existing)>=overlap_threshold for existing in unique)
        if not duplicate:
            unique.append(candidate)
    return unique

def print_profile(timings:dict):
    total=timings.get("total",0.0)
    print("\n"+"="*60,flush=True)
    print("SEARCH PERFORMANCE PROFILE",flush=True)
    print("="*60,flush=True)
    for stage,duration in timings.items():
        if stage=="total":
            continue
        percentage=100.0*duration/total if total>0 else 0.0
        print(f"{stage:<25} {duration:>9.3f}s ({percentage:>5.1f}%)",flush=True)
    print("-"*60,flush=True)
    print(f"{'TOTAL':<25} {total:>9.3f}s",flush=True)
    print("="*60,flush=True)

def filter_explicit_mentions(moments:list[dict],query:str)->list[dict]:
    analysis=analyze_query(query)
    target=analysis.get("mention_target")
    if not target:
        return moments
    pattern=re.compile(r"(?<!\w)"+re.escape(target)+r"(?!\w)",flags=re.IGNORECASE)
    verified=[]
    for moment in moments:
        evidence=moment.get("evidence",[])
        if not isinstance(evidence,list):
            continue
        for item in evidence:
            if not isinstance(item,dict):
                continue
            if item.get("modality")!="audio":
                continue
            description=str(item.get("description",""))
            if pattern.search(description):
                verified.append(moment)
                break
    return verified

def search_video(video_path:str,query:str,top_k:int=DEFAULT_TOP_K,fusion_method:str=DEFAULT_FUSION_METHOD)->dict:
    resolved_path=validate_search_request(
        video_path=video_path,
        query=query,
        top_k=top_k,
        fusion_method=fusion_method,)
    query=query.strip()
    start_time=time.perf_counter()
    timings={}
    print("\n"+"="*60,flush=True)
    print("VIDEO MOMENT RETRIEVAL SERVICE",flush=True)
    print("="*60,flush=True)
    print("Video:",resolved_path,flush=True)
    print("Query:",query,flush=True)
    print("Fusion:",fusion_method,flush=True)
    print("Top K:",top_k,flush=True)

    stage_start=time.perf_counter()
    retrieval_results=get_multimodal_results(
        video_path=resolved_path,
        query=query,
        top_k=top_k,
        fusion_method=fusion_method,)
    timings["multimodal_retrieval"]=time.perf_counter()-stage_start

    if not retrieval_results:
        elapsed=time.perf_counter()-start_time
        timings["total"]=elapsed
        print_profile(timings)
        return json_safe({
            "query":query,
            "fusion_method":fusion_method,
            "total_results":0,
            "processing_time_seconds":round(elapsed,3),
            "moments":[],})

    stage_start=time.perf_counter()
    localized_results=localize_moments(resolved_path,query,retrieval_results)
    timings["temporal_localization"]=time.perf_counter()-stage_start

    stage_start=time.perf_counter()
    unique_results=remove_duplicate_moments(localized_results,overlap_threshold=0.90)
    unique_results=filter_explicit_mentions(unique_results,query)
    timings["duplicate_removal"]=time.perf_counter()-stage_start

    stage_start=time.perf_counter()
    moments=[format_moment(moment,rank=index) for index,moment in enumerate(unique_results[:top_k],start=1)]
    timings["moment_formatting"]=time.perf_counter()-stage_start

    stage_start=time.perf_counter()
    elapsed=time.perf_counter()-start_time
    response={
        "query":query,
        "fusion_method":fusion_method,
        "total_results":len(moments),
        "processing_time_seconds":round(elapsed,3),
        "moments":moments,}
    response=json_safe(response)
    timings["response_serialization"]=time.perf_counter()-stage_start
    timings["total"]=time.perf_counter()-start_time

    print_profile(timings)
    print("\nSEARCH COMPLETED",flush=True)
    print("Results:",len(moments),flush=True)
    print("Time:",round(timings["total"],3),"seconds",flush=True)
    for moment in moments:
        print(
            f"  #{moment['rank']} "
            f"{moment['start']:.2f}s - "
            f"{moment['end']:.2f}s "
            f"(score={moment['relevance_score']:.4f})",
            flush=True,)
    return response

if __name__=="__main__":
    import json
    project_dir=Path(__file__).resolve().parent
    test_video=project_dir/"nasa.mp4"
    test_query="The narrator announces the first launch window for the Artemis II mission"
    result=search_video(
        video_path=str(test_video),
        query=test_query,
        top_k=3,
        fusion_method="reliability_aware",)
    print("\nFINAL STRUCTURED RESPONSE\n")
    print(json.dumps(result,indent=2,ensure_ascii=False))