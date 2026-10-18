import bisect
import math
import re
import sys
import numpy as np
from multimodal_retrieval import get_multimodal_results
from video_frames import get_visual_results
from scene_detection import detect_scene_boundaries
from query_analysis import analyze_query

VIDEO_PATH="sample_videos/nasa.mp4"
QUERY="The narrator announces the first launch window for the Artemis II mission"
OCR_SAMPLE_INTERVAL=3.0
OCR_MIN_SCORE_FOR_LOCALIZATION=0.35
OCR_MAX_DISTANCE_FROM_VISUAL=1.5
OCR_MAX_DISTANCE_FROM_PEAK=3.0
DIALOGUE_PAD_BEFORE_S=0.5
DIALOGUE_PAD_AFTER_S=0.5
DIRECT_SUPPORT_MIN_CONTENT_TOKENS=2

def safe_float(value,default=None):
    try:
        result=float(value)
    except (TypeError,ValueError,OverflowError):
        return default
    if not math.isfinite(result):
        return default
    return result

def valid_interval(interval,allow_point=False):
    if not isinstance(interval,(list,tuple)):
        return False
    if len(interval)!=2:
        return False
    start=safe_float(interval[0])
    end=safe_float(interval[1])
    if start is None or end is None:
        return False
    if start<0:
        return False
    if allow_point:
        return end>=start
    return end>start

def intersect_intervals(first,second):
    start=max(first[0],second[0])
    end=min(first[1],second[1])
    if end<=start:
        return None
    return float(start),float(end)

def union_intervals(first,second):
    return min(first[0],second[0]),max(first[1],second[1])

def clamp_interval(interval,video_duration):
    start=max(0.0,min(float(interval[0]),video_duration))
    end=max(0.0,min(float(interval[1]),video_duration))
    return min(start,end),max(start,end)

def distance_to_interval(timestamp,interval):
    start,end=interval
    if start<=timestamp<=end:
        return 0.0
    return min(abs(timestamp-start),abs(timestamp-end))

def get_visual_series(video_path,query):
    results=get_visual_results(video_path,query,sample_interval=0.5)
    points=[]
    for result in results:
        timestamp=safe_float(result.get("timestamp"))
        score=safe_float(result.get("score"))
        if timestamp is None or score is None:
            continue
        if timestamp<0:
            continue
        points.append((timestamp,score))
    points.sort(key=lambda item:item[0])
    return points

def find_visual_interval(peak_timestamp,visual_points,threshold_ratio=0.75,max_gap=1.0):
    if not visual_points:
        peak=float(peak_timestamp)
        return peak,peak
    times=np.asarray([item[0] for item in visual_points],dtype=float)
    scores=np.asarray([item[1] for item in visual_points],dtype=float)
    index=int(np.argmin(np.abs(times-peak_timestamp)))
    peak_score=float(scores[index])
    if peak_score<=0:
        timestamp=float(times[index])
        return timestamp,timestamp
    threshold=peak_score*threshold_ratio
    left=index
    right=index
    while left>0:
        if times[left]-times[left-1]>max_gap:
            break
        if scores[left-1]<threshold:
            break
        left-=1
    while right<len(times)-1:
        if times[right+1]-times[right]>max_gap:
            break
        if scores[right+1]<threshold:
            break
        right+=1
    return float(times[left]),float(times[right])

def get_audio_evidence(result):
    candidates=[]
    for item in result.get("evidence",[]):
        if item.get("modality")!="audio":
            continue
        start=safe_float(item.get("timestamp"))
        end=safe_float(item.get("end"))
        score=safe_float(item.get("score",0.0))
        if None in (start,end,score):
            continue
        if end<=start:
            continue
        candidates.append((score,start,end,item))
    if not candidates:
        return None
    score,start,end,item=max(candidates,key=lambda candidate:candidate[0])
    description=item.get("description","")
    return {
        "start":start,
        "end":end,
        "score":score,
        "description":description if isinstance(description,str) else ""}

def get_audio_interval(result):
    evidence=get_audio_evidence(result)
    if evidence is None:
        return None
    return evidence["start"],evidence["end"]

def get_shot_interval(timestamp,boundaries,video_duration):
    valid_boundaries=[]
    for boundary in boundaries:
        value=safe_float(boundary)
        if value is not None and 0<value<video_duration:
            valid_boundaries.append(value)
    valid_boundaries=sorted(set(valid_boundaries))
    index=bisect.bisect_right(valid_boundaries,timestamp)
    shot_start=valid_boundaries[index-1] if index>0 else 0.0
    shot_end=valid_boundaries[index] if index<len(valid_boundaries) else video_duration
    return shot_start,shot_end

def mode_from_query_type(query_type):
    query_type=str(query_type if query_type is not None else "mixed").strip().lower()
    if query_type in {"visual","object","objects","action","scene"}:
        return "visual"
    if query_type in {"audio","speech","spoken","narration","transcript"}:
        return "audio"
    if query_type in {"ocr","text","on_screen_text","screen_text"}:
        return "ocr"
    return "mixed"

def get_localization_mode(query):
    analysis=analyze_query(query)
    return mode_from_query_type(analysis.get("query_type","mixed"))

def get_ocr_interval(result,peak_timestamp,visual_interval,shot_interval,video_duration):
    candidates=[]
    diagnostics={
        "ocr_refinement_used":False,
        "ocr_candidate_count":0,
        "ocr_selected_timestamp":None,
        "ocr_selected_score":None,
        "ocr_reason":None}
    for item in result.get("evidence",[]):
        if item.get("modality")!="ocr":
            continue
        timestamp=safe_float(item.get("timestamp"))
        score=safe_float(item.get("score",0.0))
        if timestamp is None or score is None:
            continue
        if score<OCR_MIN_SCORE_FOR_LOCALIZATION:
            continue
        if not (shot_interval[0]<=timestamp<=shot_interval[1]):
            continue
        visual_distance=distance_to_interval(timestamp,visual_interval)
        peak_distance=abs(timestamp-peak_timestamp)
        if visual_distance>OCR_MAX_DISTANCE_FROM_VISUAL:
            continue
        if peak_distance>OCR_MAX_DISTANCE_FROM_PEAK:
            continue
        candidates.append({
            "timestamp":timestamp,
            "score":score,
            "visual_distance":visual_distance,
            "peak_distance":peak_distance})
    diagnostics["ocr_candidate_count"]=len(candidates)
    if not candidates:
        diagnostics["ocr_reason"]="No sufficiently close OCR evidence"
        return None,diagnostics
    selected=max(candidates,key=lambda item:(
        item["score"],-item["visual_distance"],-item["peak_distance"]))
    timestamp=selected["timestamp"]
    half_window=OCR_SAMPLE_INTERVAL/2.0
    estimated=(timestamp-half_window,timestamp+half_window)
    estimated=clamp_interval(estimated,video_duration)
    estimated=intersect_intervals(estimated,shot_interval)
    if estimated is None:
        diagnostics["ocr_reason"]="OCR uncertainty interval does not overlap the candidate shot"
        return None,diagnostics
    diagnostics.update({
        "ocr_refinement_used":True,
        "ocr_selected_timestamp":timestamp,
        "ocr_selected_score":selected["score"],
        "ocr_reason":"Nearby OCR observation; provisional sample-centered boundaries"})
    return estimated,diagnostics

def diagnose_evidence_alignment(evidence,localized_interval,tolerance=0.0):
    aligned=[]
    outside=[]
    invalid=[]
    for item in evidence:
        if not isinstance(item,dict):
            invalid.append(item)
            continue
        start=safe_float(item.get("timestamp"))
        end=safe_float(item.get("end",start))
        if start is None or end is None:
            invalid.append(item)
            continue
        if start<0 or end<start:
            invalid.append(item)
            continue
        interval_start=localized_interval[0]-tolerance
        interval_end=localized_interval[1]+tolerance
        if start==end:
            matches=interval_start<=start<=interval_end
        else:
            matches=start<interval_end and end>interval_start
        entry={
            "modality":item.get("modality","unknown"),
            "timestamp":start,
            "end":end,
            "score":item.get("score"),
            "description":item.get("description","")}
        if matches:
            aligned.append(entry)
        else:
            outside.append(entry)
    total_valid=len(aligned)+len(outside)
    alignment_ratio=len(aligned)/total_valid if total_valid else None
    return {
        "aligned_count":len(aligned),
        "outside_count":len(outside),
        "invalid_count":len(invalid),
        "alignment_ratio":alignment_ratio,
        "aligned_evidence":aligned,
        "outside_evidence":outside}

_STOPWORDS=frozenset({
    "a","an","the","and","or","but","nor","so","yet",
    "of","to","in","on","at","for","with","by","from",
    "into","onto","over","under","about","as","than",
    "then","too","very","also","just","if","not","no",
    "is","are","was","were","be","been","being","am",
    "do","does","did","done","have","has","had","having",
    "can","could","will","would","shall","should","may",
    "might","must","it","its","this","that","these",
    "those","there","here","he","she","they","him","her",
    "them","his","hers","their","theirs","i","you","we",
    "me","my","your","our","us","yourself","myself",
    "ourselves","who","whom","whose","which","what","when",
    "where","why","how","oh","yes"})

_UTTERANCE_PERSON_WORDS=frozenset({
    "i","you","we","me","my","your","our","us",
    "yourself","myself","ourselves"})

_TOKEN_PATTERN=re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

def _base_tokens(text):
    if not isinstance(text,str):
        return []
    text=text.lower().replace("\u2019","'")
    tokens=[]
    for token in _TOKEN_PATTERN.findall(text):
        if token.endswith("'s"):
            token=token[:-2]
        tokens.append(token)
    return [token for token in tokens if token]

def _fold_token(token):
    if len(token)>3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token

def _contains_contiguous(sequence,pattern):
    if not pattern or len(pattern)>len(sequence):
        return False
    width=len(pattern)
    for index in range(len(sequence)-width+1):
        if sequence[index:index+width]==pattern:
            return True
    return False

def assess_lexical_support(query,transcript_text):
    query_tokens=_base_tokens(query)
    transcript_tokens=_base_tokens(transcript_text)
    query_folded=[_fold_token(t) for t in query_tokens]
    transcript_folded=[_fold_token(t) for t in transcript_tokens]
    content=[]
    for token in query_tokens:
        if token in _STOPWORDS:
            continue
        folded=_fold_token(token)
        if folded not in content:
            content.append(folded)
    transcript_set=set(transcript_folded)
    matched=[t for t in content if t in transcript_set]
    verbatim=_contains_contiguous(transcript_folded,query_folded)
    direct=verbatim and len(content)>=DIRECT_SUPPORT_MIN_CONTENT_TOKENS
    return {
        "query_content_tokens":content,
        "matched_content_tokens":matched,
        "content_coverage":len(matched)/len(content) if content else 0.0,
        "verbatim":bool(verbatim),
        "direct":bool(direct)}

def get_utterance_cues(query):
    cues=[]
    if not isinstance(query,str):
        return cues
    if re.search(r"[\"\u201c\u201d]",query):
        cues.append("quotation_marks")
    bases={token.split("'")[0] for token in _base_tokens(query)}
    if bases&_UTTERANCE_PERSON_WORDS:
        cues.append("first_or_second_person")
    if "!" in query:
        cues.append("exclamation")
    return cues

def classify_query_intent(analysis):
    if "matched_signals" not in analysis:
        return {
            "explicit":True,
            "signalled_modalities":[],
            "query_type":analysis.get("query_type")}
    matched=analysis.get("matched_signals") or {}
    signalled=[
        modality for modality in ("visual","audio","ocr","objects")
        if matched.get(modality)]
    return {
        "explicit":bool(signalled),
        "signalled_modalities":signalled,
        "query_type":analysis.get("query_type")}

def resolve_localization(query,intent,legacy_mode,audio_evidence):
    cues=get_utterance_cues(query)
    support=None
    if audio_evidence is not None and audio_evidence.get("description"):
        support=assess_lexical_support(query,audio_evidence["description"])
    decision={
        "mode":legacy_mode,
        "confidence":"explicit",
        "hedge":False,
        "basis":"",
        "utterance_cues":cues,
        "lexical_support":support}
    if intent["explicit"]:
        signalled=intent.get("signalled_modalities") or []
        decision["basis"]=(
            "Query language indicates "+legacy_mode+" intent"
            +(" ("+", ".join(signalled)+")" if signalled else ""))
        return decision
    if audio_evidence is None:
        decision.update({
            "mode":"visual",
            "confidence":"uncertain",
            "basis":"Ambiguous query; no audio evidence in this moment; visual candidate (legacy default)"})
        return decision
    if support is not None and support["direct"] and cues:
        decision.update({
            "mode":"audio",
            "confidence":"evidence",
            "basis":"Ambiguous query; matched transcript contains the query verbatim and the query reads like speech ("+", ".join(cues)+")"})
        return decision
    missing=[]
    if support is None or not support["direct"]:
        missing.append("transcript does not directly contain the query")
    if not cues:
        missing.append("query has no utterance cues")
    decision.update({
        "mode":"mixed",
        "confidence":"uncertain",
        "hedge":True,
        "basis":"Ambiguous query; audio vs visual not established ("+"; ".join(missing)+")"})
    return decision

def valid_shot_boundaries(boundaries,video_duration):
    values=[]
    for boundary in boundaries or []:
        value=safe_float(boundary)
        if value is not None and 0<value<video_duration:
            values.append(value)
    return sorted(set(values))

def pad_speech_interval(speech_interval,shot_boundaries,video_duration,pad_before=None,pad_after=None):
    if pad_before is None:
        pad_before=DIALOGUE_PAD_BEFORE_S
    if pad_after is None:
        pad_after=DIALOGUE_PAD_AFTER_S
    pad_before=safe_float(pad_before)
    pad_after=safe_float(pad_after)
    if pad_before is None or pad_after is None or pad_before<0 or pad_after<0:
        raise ValueError("Dialogue padding must be finite and non-negative.")
    if speech_interval is None:
        return None
    speech_start=safe_float(speech_interval[0])
    speech_end=safe_float(speech_interval[1])
    if speech_start is None or speech_end is None:
        return None
    speech_start=max(0.0,speech_start)
    speech_end=min(float(video_duration),speech_end)
    if speech_end<=speech_start:
        return None
    cuts=valid_shot_boundaries(shot_boundaries,video_duration)
    lead_start=max(0.0,speech_start-pad_before)
    trail_end=min(float(video_duration),speech_end+pad_after)
    cuts_in_lead=[cut for cut in cuts if lead_start<cut<=speech_start]
    cuts_in_trail=[cut for cut in cuts if speech_end<=cut<trail_end]
    start=max(cuts_in_lead) if cuts_in_lead else lead_start
    end=min(cuts_in_trail) if cuts_in_trail else trail_end
    return float(start),float(end)

def choose_hedged_interval(speech_interval,visual_candidate):
    touching=(
        speech_interval[0]<=visual_candidate[1]
        and visual_candidate[0]<=speech_interval[1])
    if touching:
        return (
            union_intervals(visual_candidate,speech_interval),
            "mixed",
            "Uncertain modality: matched speech and visual candidate overlap; interval spans both so neither hypothesis is truncated",
            [])
    return (
        visual_candidate,
        "visual",
        "Uncertain modality: matched speech and visual candidate are disjoint; legacy visual candidate kept, audio interval reported as an alternative",
        [{
            "modality":"audio",
            "interval":speech_interval,
            "note":"matched transcript interval with padding"}])
def visual_local_maxima(visual_points):
    scores=[score for _,score in visual_points]
    if len(scores)<3:
        return scores
    peaks=[]
    for index,value in enumerate(scores):
        left=scores[index-1] if index>0 else float("-inf")
        right=scores[index+1] if index<len(scores)-1 else float("-inf")
        if value>left and value>=right:
            peaks.append(value)
    return peaks or scores

def empirical_tail(score,pool):
    if score is None or not pool:
        return None
    count=sum(1 for value in pool if value>=score)
    return (count+1.0)/(len(pool)+1.0)

def compute_tail_diagnostics(result,audio_evidence,visual_pool,audio_pool):
    visual_scores=[
        safe_float(item.get("score"))
        for item in result.get("evidence",[])
        if item.get("modality")=="visual"]
    visual_scores=[score for score in visual_scores if score is not None]
    visual_score=max(visual_scores) if visual_scores else None
    audio_score=audio_evidence["score"] if audio_evidence is not None else None
    return {
        "audio_tail":empirical_tail(audio_score,audio_pool),
        "audio_pool_size":len(audio_pool or []),
        "visual_tail":empirical_tail(visual_score,visual_pool),
        "visual_pool_size":len(visual_pool or []),
        "note":"Diagnostic only. Tails are relative to different candidate pools and do not establish semantic relevance. They do not affect localization."}

def choose_localized_interval(mode,visual_interval,shot_visual_interval,audio_interval,shot_constrained,ocr_interval=None):
    visual_candidate=shot_visual_interval
    if mode=="visual":
        return visual_candidate,"Visual query: audio does not extend playback boundaries"
    if mode=="audio":
        if audio_interval is not None:
            return audio_interval,"Audio query: matched transcript interval"
        return visual_candidate,"Audio query: no valid audio interval; visual fallback"
    if mode=="ocr":
        if ocr_interval is not None:
            return ocr_interval,"OCR query: nearby OCR observation with provisional sampling boundaries"
        return visual_candidate,"OCR query: no suitable OCR boundary evidence; visual/shot fallback"
    if audio_interval is not None:
        overlap=intersect_intervals(visual_candidate,audio_interval)
        if overlap is not None:
            return overlap,"Mixed query: overlapping visual and audio evidence"
        return visual_candidate,"Mixed query: audio and visual intervals do not overlap"
    return visual_candidate,"Mixed query: no valid audio interval"

def localize_moments(
    video_path,query,retrieval_results,threshold_ratio=0.75,
    video_duration=None,dialogue_pad_before=None,dialogue_pad_after=None,
    visual_points=None,shot_boundaries=None,compute_tails=False,
    audio_score_pool=None):

    analysis=analyze_query(query)
    intent=classify_query_intent(analysis)
    legacy_mode=mode_from_query_type(analysis.get("query_type","mixed"))

    if visual_points is None:
        visual_points=get_visual_series(video_path,query)

    if shot_boundaries is None:
        scene_result=detect_scene_boundaries(
            video_path,
            sample_interval=0.5,
            threshold=4.0,
            min_scene_duration=1.5)
        boundaries=scene_result["boundaries"]
    else:
        boundaries=shot_boundaries

    if video_duration is None:
        import cv2
        cap=cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        try:
            fps=cap.get(cv2.CAP_PROP_FPS)
            frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        finally:
            cap.release()
        if fps<=0:
            raise ValueError("Invalid video FPS")
        video_duration=frames/fps

    video_duration=float(video_duration)
    if not math.isfinite(video_duration) or video_duration<=0:
        raise ValueError("Invalid video duration")

    pad_speech_interval(
        (0.0,1.0),[],max(video_duration,1.0),
        dialogue_pad_before,dialogue_pad_after)

    visual_pool=None
    if compute_tails:
        visual_pool=visual_local_maxima(visual_points)
        if audio_score_pool is None:
            from audio_frame import get_text_results
            audio_score_pool=[
                safe_float(item.get("score"),0.0)
                for item in get_text_results(video_path,query)]

    localized=[]
    for result in retrieval_results:
        peak=safe_float(result.get("timestamp"))
        if peak is None:
            continue
        peak=max(0.0,min(peak,video_duration))

        visual_interval=find_visual_interval(
            peak,visual_points,threshold_ratio=threshold_ratio)

        audio_evidence=get_audio_evidence(result)
        audio_interval=(
            (audio_evidence["start"],audio_evidence["end"])
            if audio_evidence is not None else None)

        shot_interval=get_shot_interval(peak,boundaries,video_duration)
        intersection=intersect_intervals(visual_interval,shot_interval)

        if intersection is None:
            shot_visual_interval=shot_interval
            shot_constrained=True
        else:
            shot_visual_interval=intersection
            shot_constrained=True

        if not (shot_visual_interval[0]<=peak<=shot_visual_interval[1]):
            shot_visual_interval=(
                max(shot_interval[0],min(shot_visual_interval[0],peak)),
                min(shot_interval[1],max(shot_visual_interval[1],peak)))

        baseline_interval=visual_interval
        if audio_interval is not None:
            baseline_interval=union_intervals(baseline_interval,audio_interval)

        ocr_interval=None
        ocr_diagnostics={
            "ocr_refinement_used":False,
            "ocr_candidate_count":0,
            "ocr_selected_timestamp":None,
            "ocr_selected_score":None,
            "ocr_reason":"Not an OCR query"}

        if legacy_mode=="ocr":
            ocr_interval,ocr_diagnostics=get_ocr_interval(
                result=result,
                peak_timestamp=peak,
                visual_interval=shot_visual_interval,
                shot_interval=shot_interval,
                video_duration=video_duration)

        decision=resolve_localization(
            query=query,
            intent=intent,
            legacy_mode=legacy_mode,
            audio_evidence=audio_evidence)

        speech_interval=pad_speech_interval(
            audio_interval,boundaries,video_duration,
            dialogue_pad_before,dialogue_pad_after)

        localization_mode=decision["mode"]
        alternatives=[]
        output_peak=peak

        if decision["mode"]=="audio" and speech_interval is not None:
            improved_interval=speech_interval
            if decision["confidence"]=="explicit":
                reason="Audio query: matched transcript interval with playback padding"
            else:
                reason=(
                    "Audio (evidence-supported): "+decision["basis"]
                    +"; matched transcript interval with playback padding")
            output_peak=(audio_evidence["start"]+audio_evidence["end"])/2.0
            output_peak=max(0.0,min(output_peak,video_duration))

        elif decision["hedge"] and speech_interval is not None:
            improved_interval,localization_mode,reason,alternatives=choose_hedged_interval(
                speech_interval,shot_visual_interval)
            reason=reason+" ("+decision["basis"]+")"

        else:
            improved_interval,reason=choose_localized_interval(
                mode=decision["mode"],
                visual_interval=visual_interval,
                shot_visual_interval=shot_visual_interval,
                audio_interval=audio_interval,
                shot_constrained=shot_constrained,
                ocr_interval=ocr_interval)
            if decision["confidence"]!="explicit":
                reason=decision["basis"]

        baseline_interval=clamp_interval(baseline_interval,video_duration)
        improved_interval=clamp_interval(improved_interval,video_duration)

        evidence=result.get("evidence",[])
        if not isinstance(evidence,list):
            evidence=[]

        alignment=diagnose_evidence_alignment(evidence,improved_interval)
        tails=None
        if compute_tails:
            tails=compute_tail_diagnostics(
                result,audio_evidence,visual_pool,audio_score_pool)

        diagnostics={
            "intent":intent,
            "utterance_cues":decision["utterance_cues"],
            "lexical_support":decision["lexical_support"],
            "decision_basis":decision["basis"],
            "hedged":bool(decision["hedge"]),
            "speech_interval_padded":speech_interval,
            "alternatives":alternatives,
            "tails":tails}

        localized.append({
            "peak_timestamp":output_peak,
            "retrieval_peak_timestamp":peak,
            "relevance_score":float(result.get("relevance_score",0.0)),
            "modalities":result.get("modalities",[]),
            "visual_interval":visual_interval,
            "audio_interval":audio_interval,
            "ocr_interval":ocr_interval,
            "shot_interval":shot_interval,
            "shot_visual_interval":shot_visual_interval,
            "shot_constrained":shot_constrained,
            "baseline_interval":baseline_interval,
            "improved_interval":improved_interval,
            "localization_mode":localization_mode,
            "localization_reason":reason,
            "localization_confidence":decision["confidence"],
            "localization_diagnostics":diagnostics,
            "ocr_diagnostics":ocr_diagnostics,
            "evidence_alignment":alignment,
            "evidence":evidence})

    return localized

def _series(duration,step=0.5,baseline=0.05,regions=()):
    points=[]
    steps=int(round(duration/step))+1
    for index in range(steps):
        t=round(index*step,6)
        value=baseline
        for region in regions:
            r_start,r_end,r_value=region
            if r_end is None:
                if abs(t-r_start)<1e-9:
                    value=r_value
            elif r_start<=t<=r_end:
                value=r_value
        points.append((t,value))
    return points

def _moment(timestamp,visual=None,audio=None):
    evidence=[]
    modalities=[]
    if visual is not None:
        evidence.append({
            "modality":"visual",
            "timestamp":visual[0],
            "end":visual[0],
            "score":visual[1],
            "description":"SigLIP visual similarity to query"})
        modalities.append("visual")
    if audio is not None:
        evidence.append({
            "modality":"audio",
            "timestamp":audio[0],
            "end":audio[1],
            "score":audio[2],
            "description":audio[3]})
        modalities.append("audio")
    return {
        "timestamp":timestamp,
        "relevance_score":0.7,
        "modalities":modalities,
        "evidence":evidence}

def run_self_tests():
    failures=[]
    passed=[0]

    def check(name,condition,detail=""):
        if condition:
            passed[0]+=1
            print(f"  PASS  {name}")
        else:
            failures.append(f"{name} {detail}")
            print(f"  FAIL  {name} {detail}")

    def close(a,b,tol=1e-9):
        return abs(a[0]-b[0])<=tol and abs(a[1]-b[1])<=tol

    def run(query,moments,duration,series,cuts,**kwargs):
        return localize_moments(
            "unused.mp4",query,moments,
            video_duration=duration,
            visual_points=series,
            shot_boundaries=cuts,
            **kwargs)

    print("\nREGRESSION TESTS: evidence-driven localization\n")
    print("Implicit quoted dialogue (verified failure)")

    series=_series(
        200,regions=[(95.0,122.5,0.11),(106.0,None,0.1303)])
    cuts=[95.0,122.5]
    moment=_moment(
        106.0,
        visual=(106.0,0.1303),
        audio=(110.0,113.0,0.9062,"Say, oh shinchan, you brave Swallow!"))
    out=run(
        "Shinchan, you brave Swallow",
        [moment],200.0,series,cuts)[0]

    check(
        "legacy visual evidence is unchanged (wide shot interval)",
        close(out["shot_visual_interval"],(95.0,122.5)),
        str(out["shot_visual_interval"]))
    check(
        "interval is matched speech +/- 0.5 s, not the shot",
        close(out["improved_interval"],(109.5,113.5)),
        str(out["improved_interval"]))
    check("mode audio",out["localization_mode"]=="audio")
    check(
        "confidence is evidence-based",
        out["localization_confidence"]=="evidence")
    check(
        "peak re-centered on speech; retrieval peak preserved",
        out["peak_timestamp"]==111.5
        and out["retrieval_peak_timestamp"]==106.0)
    check(
        "audio_interval key preserved",
        close(out["audio_interval"],(110.0,113.0)))

    print("\nExplicit dialogue")
    series=_series(
        100,regions=[(10.0,40.0,0.11),(21.0,None,0.14)])
    moment=_moment(
        21.0,
        visual=(21.0,0.14),
        audio=(20.0,24.0,0.5,"Shinchan, wake up and get ready"))
    out=run(
        "Mother tells Shinchan to wake up",
        [moment],100.0,series,[10.0,40.0])[0]

    check(
        "speech is not clipped to the visual shot",
        close(out["improved_interval"],(19.5,24.5)),
        str(out["improved_interval"]))
    check(
        "explicit confidence",
        out["localization_confidence"]=="explicit"
        and out["localization_mode"]=="audio")

    print("\nVisual action with incidental verbatim transcript")
    series=_series(
        120,regions=[(48.0,54.0,0.25),(50.0,None,0.27)])
    moment=_moment(
        50.0,
        visual=(50.0,0.27),
        audio=(49.0,52.0,0.8,"a child riding a bicycle"))
    out=run(
        "A child riding a bicycle",
        [moment],120.0,series,[40.0,70.0])[0]

    check(
        "explicit visual query keeps the visual interval",
        close(out["improved_interval"],(48.0,54.0)),
        str(out["improved_interval"]))
    check(
        "audio match cannot override a clearly visual query",
        out["localization_mode"]=="visual"
        and out["localization_confidence"]=="explicit")

    print("\nAmbiguous queries")
    series=_series(
        100,regions=[(28.0,36.0,0.20),(31.0,None,0.22)])
    moment=_moment(
        31.0,
        visual=(31.0,0.22),
        audio=(30.0,33.0,0.7,"look at that red car"))
    out=run(
        "red car",[moment],100.0,series,[25.0,45.0])[0]

    check(
        "no utterance cue: uncertain hedge spans both hypotheses",
        out["localization_confidence"]=="uncertain"
        and out["localization_diagnostics"]["hedged"]
        and close(out["improved_interval"],(28.0,36.0)),
        f"{out['localization_confidence']} {out['improved_interval']}")

    series=_series(
        100,regions=[(24.5,25.5,0.22),(25.0,None,0.24)])
    moment=_moment(
        25.0,
        visual=(25.0,0.24),
        audio=(30.0,33.0,0.7,"look at that red car"))
    out=run(
        "red car",[moment],100.0,series,[20.0,29.0])[0]
    alternatives=out["localization_diagnostics"]["alternatives"]

    check(
        "disjoint hypotheses: legacy visual candidate kept",
        out["localization_mode"]=="visual"
        and out["localization_confidence"]=="uncertain"
        and close(out["improved_interval"],out["shot_visual_interval"]),
        str(out["improved_interval"]))
    check(
        "disjoint hypotheses: audio alternative reported",
        len(alternatives)==1
        and close(alternatives[0]["interval"],(29.5,33.5)),
        str(alternatives))

    moment=_moment(25.0,visual=(25.0,0.24))
    out=run(
        "red car",[moment],100.0,series,[20.0,29.0])[0]
    check(
        "no audio evidence: visual fallback, flagged uncertain",
        out["localization_mode"]=="visual"
        and out["localization_confidence"]=="uncertain")

    series=_series(100,regions=[(28.0,36.0,0.20)])
    moment=_moment(
        31.0,
        visual=(31.0,0.20),
        audio=(30.0,33.0,0.6,"Shinchan is waiting outside"))
    out=run(
        "Shinchan, you are late",
        [moment],100.0,series,[25.0,45.0])[0]
    check(
        "partial overlap with cue still uncertain (not audio)",
        out["localization_mode"]!="audio"
        and out["localization_confidence"]=="uncertain",
        f"{out['localization_mode']} {out['localization_confidence']}")

    series=_series(100,regions=[(28.0,36.0,0.20)])
    moment=_moment(
        31.0,
        visual=(31.0,0.20),
        audio=(30.0,33.0,0.6,"brave swallow, you shinchan"))
    out=run(
        "Shinchan, you brave Swallow",
        [moment],100.0,series,[25.0,45.0])[0]
    check(
        "all words present but not verbatim: uncertain, not audio",
        out["localization_mode"]!="audio"
        and out["localization_confidence"]=="uncertain",
        f"{out['localization_mode']} {out['localization_confidence']}")

    series=_series(
        100,regions=[(28.0,36.0,0.20),(31.0,None,0.22)])
    moment=_moment(
        31.0,
        visual=(31.0,0.22),
        audio=(30.0,33.0,0.7,"look at that red car"))
    low=run(
        "red car",[moment],100.0,series,[25.0,45.0],
        compute_tails=True,
        audio_score_pool=[0.05]*50+[0.95]*50)[0]
    high=run(
        "red car",[moment],100.0,series,[25.0,45.0],
        compute_tails=True,
        audio_score_pool=[0.01]*100)[0]

    check(
        "tails reported when requested",
        low["localization_diagnostics"]["tails"] is not None
        and low["localization_diagnostics"]["tails"]["audio_pool_size"]==100)
    check(
        "different tails change nothing about the decision",
        low["localization_diagnostics"]["tails"]["audio_tail"]
        !=high["localization_diagnostics"]["tails"]["audio_tail"]
        and close(low["improved_interval"],high["improved_interval"])
        and low["localization_mode"]==high["localization_mode"])

    print("\nDialogue crossing shot cuts")
    series=_series(200,regions=[(100.0,120.0,0.11)])
    moment=_moment(
        112.0,
        visual=(112.0,0.11),
        audio=(108.0,116.0,0.6,"the narrator talks for a while"))
    cuts=[107.8,112.0,116.2]
    out=run(
        "The narrator says it is time",
        [moment],200.0,series,cuts)[0]
    start,end=out["improved_interval"]

    check(
        "matched speech fully preserved across an interior cut",
        start<=108.0 and end>=116.0 and start<112.0<end,
        str(out["improved_interval"]))
    check(
        "padding stops at cuts inside the padding zones",
        close(out["improved_interval"],(107.8,116.2)),
        str(out["improved_interval"]))
    check(
        "pure function: cut exactly at speech start/end adds no padding",
        close(
            pad_speech_interval(
                (110.0,113.0),[110.0,113.0],200.0),
            (110.0,113.0)))

    print("\nVideo boundaries and padding configuration")
    check(
        "clamped at video start",
        close(pad_speech_interval((0.1,2.0),[],60.0),(0.0,2.5)))
    check(
        "clamped at video end",
        close(pad_speech_interval((58.5,60.0),[],60.0),(58.0,60.0)))
    check(
        "speech past the end is clamped, not dropped",
        close(pad_speech_interval((59.0,61.0),[],60.0),(58.5,60.0)))
    check(
        "zero padding returns the exact speech interval",
        close(
            pad_speech_interval(
                (110.0,113.0),[],200.0,0.0,0.0),
            (110.0,113.0)))
    check(
        "speech outside the video returns None",
        pad_speech_interval((70.0,75.0),[],60.0) is None)

    raised=False
    try:
        pad_speech_interval((1.0,2.0),[],60.0,-1.0,0.5)
    except ValueError:
        raised=True
    check("negative padding is rejected",raised)

    out=run(
        "Mother tells Shinchan to wake up",
        [_moment(
            3.0,
            visual=(3.0,0.1),
            audio=(0.0,4.0,0.5,"wake up"))],
        60.0,_series(60),[],
        dialogue_pad_before=2.0,
        dialogue_pad_after=1.0)[0]
    check(
        "custom padding honored and clamped at 0",
        close(out["improved_interval"],(0.0,5.0)),
        str(out["improved_interval"]))

    print("\nPreserved behavior")
    check(
        "get_localization_mode: audio / visual / ocr / mixed",
        get_localization_mode(
            "The narrator announces the launch window")=="audio"
        and get_localization_mode(
            "A rocket is visible on the screen")=="visual"
        and get_localization_mode(
            "The words Artemis II appear on the screen")=="ocr"
        and get_localization_mode(
            "A rocket is shown while the narrator discusses the mission")=="mixed")

    wide=(95.0,122.5)
    check(
        "choose_localized_interval unchanged for all four modes",
        choose_localized_interval(
            "visual",wide,wide,(110.0,113.0),True)[0]==wide
        and choose_localized_interval(
            "audio",wide,wide,(110.0,113.0),True)[0]==(110.0,113.0)
        and choose_localized_interval(
            "ocr",wide,wide,None,True)[0]==wide
        and choose_localized_interval(
            "mixed",wide,wide,(110.0,113.0),True)[0]==(110.0,113.0))

    series=_series(
        120,regions=[(58.0,70.0,0.2),(62.0,None,0.25)])
    moment=_moment(
        62.0,
        visual=(62.0,0.25),
        audio=(60.0,64.0,0.6,"the mission to the moon"))
    out=run(
        "A rocket is shown while the narrator discusses the Artemis mission",
        [moment],120.0,series,[50.0,80.0])[0]
    check(
        "explicit multimodal path unchanged (legacy mixed policy)",
        out["localization_mode"]=="mixed"
        and out["localization_confidence"]=="explicit"
        and close(out["improved_interval"],(60.0,64.0)),
        str(out["improved_interval"]))

    check(
        "lexical support: verbatim phrase detected",
        assess_lexical_support(
            "Shinchan, you brave Swallow",
            "Say, oh shinchan, you brave Swallow!")["direct"])
    check(
        "lexical support: scattered words are not verbatim",
        not assess_lexical_support(
            "red car","the car was red")["verbatim"])
    check(
        "lexical support: one content word is not a phrase",
        not assess_lexical_support(
            "the shinchan","oh the shinchan")["direct"])

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        for failure in failures:
            print("  -",failure)
        raise SystemExit(1)
    print(f"PASS: all {passed[0]} regression checks passed.")

if __name__=="__main__":
    if "--self-test" in sys.argv:
        run_self_tests()
        raise SystemExit(0)

    retrieval_results=get_multimodal_results(
        VIDEO_PATH,QUERY,fusion_method="reliability_aware")
    moments=localize_moments(VIDEO_PATH,QUERY,retrieval_results)

    print("\nQUERY-DEPENDENT TEMPORAL LOCALIZATION\n")
    for index,moment in enumerate(moments[:10],start=1):
        print(f"\n{index}. Moment")
        print("   Peak:",moment["peak_timestamp"])
        print("   Mode:",moment["localization_mode"])
        print("   Visual:",moment["visual_interval"])
        print("   Audio:",moment["audio_interval"])
        print("   OCR:",moment["ocr_interval"])
        print("   Improved:",moment["improved_interval"])
        print("   Reason:",moment["localization_reason"])
        print("   Confidence:",moment["localization_confidence"])
        print("   OCR diagnostics:",moment["ocr_diagnostics"])
        print(
            "   Evidence alignment:",
            {
                key:value
                for key,value in moment["evidence_alignment"].items()
                if key not in {"aligned_evidence","outside_evidence"}})