import inspect
import math
import numpy as np
from sentence_transformers import SentenceTransformer
from video_frames import get_visual_results
from audio_frame import get_text_results
from video_index import get_cached_yolo,get_cached_ocr
from query_analysis import analyze_query

MODALITIES=("visual","audio","ocr","objects")
MODALITY_WEIGHTS={"visual":0.45,"audio":0.30,"ocr":0.15,"objects":0.10}
VALID_FUSION_METHODS=("fixed","query_aware","reliability_aware","condition_aware")
OCR_MIN_CONFIDENCE=0.40
ROBUST_Z_SCALE=3.0
RELIABILITY_FLOOR=0.15
RELIABILITY_EXPONENT=1.0
COVERAGE_PENALTY=0.20
AGREEMENT_STRENGTH=0.08
NORMALIZED_SCORE_CEILING=0.95
text_model=SentenceTransformer("all-MiniLM-L6-v2")

def safe_float(value,default=0.0):
    try:
        result=float(value)
        if math.isfinite(result):
            return result
    except (TypeError,ValueError):
        pass
    return default

def clamp(value,low=0.0,high=1.0):
    return max(low,min(high,float(value)))

def normalize_weights(weights):
    values={modality:max(0.0,safe_float(weights.get(modality,0.0))) for modality in MODALITIES}
    total=sum(values.values())
    if total<=0:
        return {modality:1.0/len(MODALITIES) for modality in MODALITIES}
    return {modality:values[modality]/total for modality in MODALITIES}

def get_fusion_weights(query,fusion_method="fixed"):
    if fusion_method not in VALID_FUSION_METHODS:
        raise ValueError(f"Unknown fusion method: {fusion_method}")
    if fusion_method=="fixed":
        return MODALITY_WEIGHTS.copy()
    analysis=analyze_query(query)
    weights=analysis["modality_weights"].copy()
    print("\n--- QUERY ANALYSIS ---")
    print("Detected query type:",analysis["query_type"])
    for modality in MODALITIES:
        print(f"{modality:10s}: {weights[modality]:.4f}")
    return weights

def to_records(results):
    if results is None:
        return []
    if isinstance(results,np.ndarray):
        return to_records(results.tolist())
    if isinstance(results,(list,tuple)):
        return [item if isinstance(item,dict) else {"text":str(item)} for item in results]
    if isinstance(results,dict):
        for key in ("results","segments","frames","detections"):
            if isinstance(results.get(key),list):
                return to_records(results[key])
        return [results]
    return [{"text":str(results)}]

def get_timestamp(record):
    for key in ("timestamp","time","start","start_time","frame_time"):
        value=record.get(key)
        if value is None:
            continue
        timestamp=safe_float(value,float("nan"))
        if math.isfinite(timestamp):
            return timestamp
    return None

def get_description(record):
    parts=[]
    for key in ("text","transcript","sentence","caption","description","label","class_name","name"):
        value=record.get(key)
        if isinstance(value,str) and value.strip():
            parts.append(value.strip())
    for key in ("objects","detections"):
        items=record.get(key,[])
        if not isinstance(items,list):
            continue
        for item in items:
            if not isinstance(item,dict):
                continue
            name=item.get("name") or item.get("class_name") or item.get("label")
            if name:
                parts.append(str(name))
    texts=record.get("texts",[])
    if isinstance(texts,list):
        for item in texts:
            if isinstance(item,dict):
                confidence=safe_float(item.get("confidence",0.0))
                value=item.get("text","")
                if confidence>=OCR_MIN_CONFIDENCE and isinstance(value,str) and value.strip():
                    parts.append(value.strip())
            elif isinstance(item,str) and item.strip():
                parts.append(item.strip())
    return " ".join(dict.fromkeys(parts))

def semantic_score(query,description):
    if not description.strip():
        return 0.0
    embeddings=text_model.encode([query,description],normalize_embeddings=True)
    similarity=float(np.dot(embeddings[0],embeddings[1]))
    return max(0.0,similarity)

def call_visual_module(video_path,query):
    parameters=inspect.signature(get_visual_results).parameters
    if "query" in parameters:
        return get_visual_results(video_path,query=query)
    if "text_query" in parameters:
        return get_visual_results(video_path,text_query=query)
    return get_visual_results(video_path)

def score_records(records,query,modality):
    scored=[]
    records=to_records(records)
    batched_scores={}
    if modality in ("ocr","objects"):
        descriptions=list(dict.fromkeys(
            get_description(record) for record in records
            if get_timestamp(record) is not None and get_description(record).strip()))
        if descriptions:
            query_embedding=text_model.encode([query],normalize_embeddings=True,convert_to_numpy=True)[0]
            description_embeddings=text_model.encode(descriptions,batch_size=32,normalize_embeddings=True,convert_to_numpy=True)
            similarities=description_embeddings@query_embedding
            batched_scores={description:max(0.0,float(score)) for description,score in zip(descriptions,similarities)}
    for record in records:
        timestamp=get_timestamp(record)
        if timestamp is None:
            continue
        description=get_description(record)
        if modality=="visual" and not description:
            description="SigLIP visual similarity to query"
        if modality=="visual":
            score=max(0.0,safe_float(record.get("similarity_score",record.get("score",0.0))))
        elif modality=="audio":
            raw_score=record.get("score")
            if raw_score is not None:
                score=max(0.0,safe_float(raw_score))
            else:
                score=semantic_score(query,description)
        else:
            score=batched_scores.get(description,0.0)
        if not description and modality!="visual":
            continue
        if modality=="objects" and score<0.10:
            continue
        if modality=="ocr" and score<0.10:
            continue
        end=safe_float(record.get("end",timestamp),timestamp)
        end=max(timestamp,end)
        scored.append({
            "timestamp":timestamp,
            "end":end,
            "modality":modality,
            "description":description,
            "score":score,
            "raw_result":record,})
    return scored

def calculate_modality_statistics(evidence):
    statistics={}
    for modality in MODALITIES:
        scores=np.asarray([item["score"] for item in evidence if item["modality"]==modality],dtype=np.float64)
        count=len(scores)
        if count==0:
            statistics[modality]={
                "count":0,
                "median":0.0,
                "scale":1.0,
                "maximum":0.0,
                "upper_background":0.0,
                "reliability":0.0,}
            continue
        median=float(np.median(scores))
        mad=float(np.median(np.abs(scores-median)))
        std=float(np.std(scores))
        scale=max(1.4826*mad,0.25*std,1e-3)
        maximum=float(np.max(scores))
        upper_background=float(np.percentile(scores,90))
        separation=max(0.0,(maximum-upper_background)/scale)
        separation_quality=separation/(separation+4.0)
        dispersion_quality=std/(std+0.10)
        availability_quality=count/(count+10.0)
        reliability=0.50*separation_quality+0.25*dispersion_quality+0.25*availability_quality
        statistics[modality]={
            "count":count,
            "median":median,
            "scale":scale,
            "maximum":maximum,
            "upper_background":upper_background,
            "reliability":clamp(reliability),}
    return statistics

def calibrate_evidence_scores(evidence,statistics):
    for item in evidence:
        stats=statistics[item["modality"]]
        z=(item["score"]-stats["median"])/stats["scale"]
        z=float(np.clip(z/ROBUST_Z_SCALE,-12.0,12.0))
        sigmoid=1.0/(1.0+math.exp(-z))
        item["normalized_score"]=NORMALIZED_SCORE_CEILING*sigmoid
    return evidence

def calculate_reliability_weights(query_weights,statistics):
    effective={}
    for modality in MODALITIES:
        stats=statistics[modality]
        if stats["count"]==0:
            effective[modality]=0.0
            continue
        reliability=max(RELIABILITY_FLOOR,stats["reliability"])
        effective[modality]=query_weights[modality]*reliability**RELIABILITY_EXPONENT
    return normalize_weights(effective)

def evidence_distance(item,anchor):
    start=item["timestamp"]
    end=item["end"]
    if start<=anchor<=end:
        return 0.0
    return min(abs(anchor-start),abs(anchor-end))

def calculate_temporal_agreement(best_by_modality,modality_weights):
    contributions=[]
    for modality,item in best_by_modality.items():
        contributions.append(modality_weights[modality]*item["normalized_score"])
    if len(contributions)<2:
        return 0.0
    contributions.sort(reverse=True)
    strongest=contributions[0]
    second=contributions[1]
    if strongest<=0:
        return 0.0
    return clamp(second/strongest)

def build_moments(evidence,merge_window,modality_weights=None,fusion_method="fixed",statistics=None):
    if modality_weights is None:
        modality_weights=MODALITY_WEIGHTS
    if statistics is None:
        statistics={}
    anchors=sorted({item["timestamp"] for item in evidence if item["modality"]!="audio"})
    anchors.extend((item["timestamp"]+item["end"])/2.0 for item in evidence if item["modality"]=="audio")
    anchors=sorted(set(anchors))
    moments=[]
    for anchor in anchors:
        nearby=[item for item in evidence if evidence_distance(item,anchor)<=merge_window]
        if not nearby:
            continue
        best_by_modality={}
        for item in nearby:
            modality=item["modality"]
            previous=best_by_modality.get(modality)
            if previous is None or item["score"]>previous["score"]:
                best_by_modality[modality]=item
        if fusion_method=="reliability_aware":
            numerator=sum(modality_weights[modality]*item["normalized_score"] for modality,item in best_by_modality.items())
            available_weight=sum(modality_weights[modality] for modality in best_by_modality)
            conditional_score=numerator/available_weight if available_weight>0 else 0.0
            coverage=clamp(available_weight)
            coverage_factor=1.0-COVERAGE_PENALTY*(1.0-coverage)
            agreement=calculate_temporal_agreement(best_by_modality,modality_weights)
            agreement_factor=1.0+AGREEMENT_STRENGTH*agreement
            fused_score=conditional_score*coverage_factor*agreement_factor
            fused_score=clamp(fused_score,0.0,NORMALIZED_SCORE_CEILING)
        else:
            numerator=sum(modality_weights[modality]*item["score"] for modality,item in best_by_modality.items())
            denominator=sum(modality_weights[modality] for modality in best_by_modality)
            fused_score=numerator/denominator if denominator>0 else 0.0
            agreement=0.0
        matched_evidence=[]
        for modality,item in best_by_modality.items():
            evidence_item={
                "modality":modality,
                "description":item["description"],
                "score":round(item["score"],4),
                "timestamp":round(item["timestamp"],2),
                "end":round(item["end"],2),}
            if fusion_method=="reliability_aware":
                evidence_item["normalized_score"]=round(item["normalized_score"],4)
                evidence_item["reliability"]=round(statistics[modality]["reliability"],4)
                evidence_item["weighted_contribution"]=round(modality_weights[modality]*item["normalized_score"],4)
            matched_evidence.append(evidence_item)
        moment={
            "timestamp":round(anchor,2),
            "relevance_score":round(fused_score,4),
            "evidence":matched_evidence,
            "modalities":list(best_by_modality),}
        if fusion_method=="reliability_aware":
            moment["temporal_agreement"]=round(agreement,4)
            moment["fusion_method"]=fusion_method
        moments.append(moment)
    return moments

def suppress_duplicates(moments,merge_window,top_k,fusion_method="fixed"):
    moments=sorted(moments,key=lambda item:item["relevance_score"],reverse=True)
    selected=[]

    def audio_segments(moment):
        return {(item["timestamp"],item["end"]) for item in moment["evidence"] if item["modality"]=="audio"}

    def get_playback_timestamp(moment):
        visual_evidence=[item for item in moment["evidence"] if item["modality"]=="visual"]
        if visual_evidence:
            return visual_evidence[0]["timestamp"]
        audio_evidence=[item for item in moment["evidence"] if item["modality"]=="audio"]
        if audio_evidence:
            item=audio_evidence[0]
            return (item["timestamp"]+item["end"])/2.0
        return moment["timestamp"]

    for moment in moments:
        anchor_timestamp=moment["timestamp"]
        playback_timestamp=get_playback_timestamp(moment)
        current_audio=audio_segments(moment)
        duplicate=False
        for existing in selected:
            existing_anchor=existing["anchor_timestamp"]
            existing_playback=existing["timestamp"]
            anchor_distance=abs(anchor_timestamp-existing_anchor)
            playback_distance=abs(playback_timestamp-existing_playback)
            if fusion_method=="reliability_aware":
                suppression_window=max(2.0,1.5*merge_window)
                close_anchor=anchor_distance<=suppression_window
                close_playback=playback_distance<=suppression_window
                duplicate=close_anchor or close_playback
            else:
                close_in_time=anchor_distance<=merge_window
                shared_audio=bool(current_audio.intersection(audio_segments(existing)))
                duplicate=close_in_time or shared_audio
            if duplicate:
                break
        if duplicate:
            continue
        moment["anchor_timestamp"]=anchor_timestamp
        moment["timestamp"]=playback_timestamp
        selected.append(moment)
        if len(selected)>=top_k:
            break
    return selected

def interval_gap(first_start,first_end,second_start,second_end):
    return max(0.0,second_start-first_end,first_start-second_end)

def _condition_aware_results(video_path,query,sample_interval,ocr_interval,confidence_threshold,merge_window,top_k):
    analysis=analyze_query(query)
    conditions=analysis.get("conditions",[])
    supported=(
        analysis.get("decomposition_status")=="decomposed"
        and analysis.get("temporal_relation")=="overlap"
        and len(conditions)==2
        and {condition["modality"] for condition in conditions}=={"visual","audio"})
    if not supported:
        print("\nCondition-aware decomposition not available for this query.")
        print("Using reliability-aware retrieval instead.")
        return get_multimodal_results(
            video_path=video_path,
            query=query,
            sample_interval=sample_interval,
            ocr_interval=ocr_interval,
            confidence_threshold=confidence_threshold,
            merge_window=merge_window,
            top_k=top_k,
            fusion_method="reliability_aware",)
    visual_condition=next(condition for condition in conditions if condition["modality"]=="visual")
    audio_condition=next(condition for condition in conditions if condition["modality"]=="audio")
    visual_query=visual_condition["text"]
    audio_query=audio_condition["text"]
    print("\n"+"="*55)
    print("CONDITION-AWARE RETRIEVAL")
    print("="*55)
    print("Visual condition:",visual_query)
    print("Audio condition:",audio_query)
    print("\nRunning SigLIP for visual condition...")
    visual_raw=call_visual_module(video_path,visual_query)
    visual_evidence=score_records(visual_raw,visual_query,"visual")
    print("Running Whisper retrieval for audio condition...")
    audio_raw=get_text_results(video_path,audio_query)
    audio_evidence=score_records(audio_raw,audio_query,"audio")
    print("Visual evidence records:",len(visual_evidence))
    print("Audio evidence records:",len(audio_evidence))
    if not visual_evidence or not audio_evidence:
        print("No candidates: missing required modality evidence.")
        return []
    evidence=visual_evidence+audio_evidence
    statistics=calculate_modality_statistics(evidence)
    calibrate_evidence_scores(evidence,statistics)
    candidates=[]
    for visual in visual_evidence:
        visual_start=visual["timestamp"]
        visual_end=visual["end"]
        for audio in audio_evidence:
            audio_start=audio["timestamp"]
            audio_end=audio["end"]
            gap=interval_gap(visual_start,visual_end,audio_start,audio_end)
            if gap>merge_window:
                continue
            visual_score=visual["normalized_score"]
            audio_score=audio["normalized_score"]
            condition_score=min(visual_score,audio_score)
            if merge_window>0:
                temporal_factor=1.0-0.25*min(gap/merge_window,1.0)
            else:
                temporal_factor=1.0
            fused_score=clamp(condition_score*temporal_factor,0.0,NORMALIZED_SCORE_CEILING)
            start=min(visual_start,audio_start)
            end=max(visual_end,audio_end)
            candidates.append({
                "timestamp":round(visual_start,2),
                "anchor_timestamp":round(visual_start,2),
                "start":round(start,2),
                "end":round(end,2),
                "relevance_score":round(fused_score,4),
                "modalities":["visual","audio"],
                "fusion_method":"condition_aware",
                "temporal_relation":"overlap",
                "temporal_gap":round(gap,4),
                "conditions":[
                    {**visual_condition,"supported_by":"visual"},
                    {**audio_condition,"supported_by":"audio"},],
                "evidence":[
                    {
                        "modality":"visual",
                        "description":visual["description"],
                        "timestamp":round(visual_start,2),
                        "end":round(visual_end,2),
                        "score":round(visual["score"],4),
                        "normalized_score":round(visual_score,4),
                        "condition_id":visual_condition["condition_id"],},
                    {
                        "modality":"audio",
                        "description":audio["description"],
                        "timestamp":round(audio_start,2),
                        "end":round(audio_end,2),
                        "score":round(audio["score"],4),
                        "normalized_score":round(audio_score,4),
                        "condition_id":audio_condition["condition_id"],},],})
    candidates.sort(key=lambda item:item["relevance_score"],reverse=True)
    selected=[]
    suppression_window=max(1.0,merge_window)
    for candidate in candidates:
        duplicate=any(abs(candidate["timestamp"]-existing["timestamp"])<=suppression_window for existing in selected)
        if duplicate:
            continue
        selected.append(candidate)
        if len(selected)>=top_k:
            break
    print("\nCondition-aware candidates:",len(candidates))
    print("Selected moments:",len(selected))
    return selected

def get_multimodal_results(video_path,query,sample_interval=0.5,ocr_interval=3.0,confidence_threshold=0.4,merge_window=2.0,top_k=10,fusion_method="fixed"):
    if not isinstance(query,str) or not query.strip():
        raise ValueError("Query must be a nonempty string.")
    if sample_interval<=0 or ocr_interval<=0:
        raise ValueError("Sampling intervals must be positive.")
    if merge_window<0:
        raise ValueError("merge_window cannot be negative.")
    if fusion_method not in VALID_FUSION_METHODS:
        raise ValueError(f"Invalid fusion method: {fusion_method}")
    if top_k<=0:
        return []
    if fusion_method=="condition_aware":
        return _condition_aware_results(
            video_path=video_path,
            query=query,
            sample_interval=sample_interval,
            ocr_interval=ocr_interval,
            confidence_threshold=confidence_threshold,
            merge_window=merge_window,
            top_k=top_k,)
    weights=get_fusion_weights(query,fusion_method=fusion_method)
    print("\n"+"="*55)
    print("MULTIMODAL RETRIEVAL")
    print("="*55)
    print("Fusion method:",fusion_method)
    print("Initial modality weights:",weights)
    print("\nRunning SigLIP...")
    visual_raw=call_visual_module(video_path,query)
    print("Running Whisper retrieval...")
    audio_raw=get_text_results(video_path,query)
    print("Running YOLO...")
    object_raw=get_cached_yolo(video_path,sample_interval=1.0,confidence_threshold=confidence_threshold)
    print("Running EasyOCR...")
    ocr_raw=get_cached_ocr(video_path,sample_interval=ocr_interval)
    raw_outputs={"visual":visual_raw,"audio":audio_raw,"objects":object_raw,"ocr":ocr_raw}
    evidence=[]
    print("\n--- MODALITY DIAGNOSTICS ---")
    for modality,raw in raw_outputs.items():
        records=to_records(raw)
        scored=score_records(records,query,modality)
        evidence.extend(scored)
        print(f"{modality}: {len(records)} raw records, {len(scored)} usable evidence records")
        for item in scored[:2]:
            print(f"  {item['timestamp']:.2f}s | {item['score']:.4f} | {item['description'][:100]}")
    if not evidence:
        return []
    statistics=None
    if fusion_method=="reliability_aware":
        statistics=calculate_modality_statistics(evidence)
        calibrate_evidence_scores(evidence,statistics)
        weights=calculate_reliability_weights(weights,statistics)
        print("\n--- RELIABILITY DIAGNOSTICS ---")
        for modality in MODALITIES:
            stats=statistics[modality]
            print(f"{modality:10s} | count={stats['count']:4d} | reliability={stats['reliability']:.4f} | effective_weight={weights[modality]:.4f}")
    moments=build_moments(evidence,merge_window,modality_weights=weights,fusion_method=fusion_method,statistics=statistics)
    return suppress_duplicates(moments,merge_window,top_k,fusion_method=fusion_method)

if __name__=="__main__":
    query="A rocket is shown while the narrator discusses the Artemis mission"
    results=get_multimodal_results(
        video_path="sample_videos/nasa.mp4",
        query=query,
        sample_interval=0.5,
        ocr_interval=3.0,
        merge_window=2.0,
        top_k=10,
        fusion_method="condition_aware",)
    print("\nRANKED VIDEO MOMENTS\n")
    if not results:
        print("No usable candidate moments found.")
    for rank,moment in enumerate(results,1):
        print(f"\n{rank}. Timestamp: {moment['timestamp']:.2f}s")
        print("   Fusion score:",f"{moment['relevance_score']:.4f}")
        print("   Modalities:",", ".join(moment["modalities"]))
        if "start" in moment:
            print("   Evidence window:",f"{moment['start']:.2f}s - {moment['end']:.2f}s")
        if "temporal_gap" in moment:
            print("   Temporal gap:",f"{moment['temporal_gap']:.2f}s")
        for item in moment["evidence"]:
            contribution=item.get("weighted_contribution")
            extra=f" | contribution={contribution:.4f}" if contribution is not None else ""
            print(f"   [{item['modality']}] {item['timestamp']:.2f}s | raw={item['score']:.4f}{extra} | {item['description'][:150]}")