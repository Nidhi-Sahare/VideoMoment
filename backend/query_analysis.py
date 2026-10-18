import re
from collections import defaultdict

MODALITIES=("visual","audio","ocr","objects")
PRIOR=1.0

QUERY_SIGNALS={
    "audio":{
        "say":3.0,"saying":3.0,"tell":3.0,"tells":3.0,"telling":3.0,"told":3.0,
        "ask":3.0,"asks":3.0,"asking":3.0,"asked":3.0,
        "instruct":3.0,"instructs":3.0,"instructing":3.0,"instructed":3.0,
        "warn":3.0,"warns":3.0,"warning":3.0,"warned":3.0,
        "speaking":2.5,"talking":2.5,"explaining":2.5,"describing":2.5,
        "narrator":3.0,"narration":3.0,"speaker":2.5,"says":3.0,"said":3.0,
        "mentions":3.0,"mention":3.0,"announces":3.0,"announce":3.0,
        "explains":2.5,"explain":2.5,"discusses":2.5,"discuss":2.5,
        "talks":2.5,"speaks":2.5,"speech":2.5,"heard":2.5,"audio":2.5,
        "pronounces":2.5,"spoken":3.0,"voiceover":3.0,"dialogue":3.0,
        "conversation":2.5,},
    "visual":{
        "visible":2.5,"shown":2.5,"shows":2.5,"appears":2.5,"appear":2.5,
        "seen":2.5,"watch":1.5,"scene":2.0,"footage":2.0,"visually":2.5,
        "looks":2.0,"wearing":2.0,"color":2.0,"colour":2.0,"colors":2.0,
        "colours":2.0,"moving":1.5,"movement":1.5,"walking":1.5,
        "running":1.5,"riding":1.5,"flying":1.5,"standing":1.5,
        "sitting":1.5,"enters":1.5,"leaves":1.5,"walk":1.5,"walks":1.5,
        "run":1.5,"runs":1.5,"ride":1.5,"rides":1.5,"fly":1.5,"flies":1.5,
        "land":1.5,"lands":1.5,"jump":1.5,"jumps":1.5,},
    "ocr":{
        "text":3.5,"written":3.5,"displayed":2.0,"subtitle":4.0,
        "subtitles":4.0,"caption":3.5,"captions":3.5,"title":2.5,
        "titles":2.5,"label":2.5,"labels":2.5,"sign":2.5,"signboard":3.0,
        "reads":3.5,"lettering":3.5,"words":3.5,"word":2.5,
        "phrase":2.0,"logo":1.5,},
    "objects":{
        "object":2.5,"objects":2.5,"person":1.5,"people":1.5,
        "vehicle":1.5,"animal":1.5,"detect":2.0,"detected":2.0,
        "count":2.0,"counting":2.0,},}

PHRASE_SIGNALS={
    "ocr":[
        (r"\b(?:word|words|text|phrase|letters)\b"
         r".{0,65}\b(?:appear|appears|shown|show|displayed|visible|written)\b",7.0),
        (r"\b(?:appear|appears|shown|displayed|written)"
         r"\b.{0,40}\b(?:on\s+(?:the\s+)?screen|in\s+(?:the\s+)?subtitles)\b",3.0),
        (r"\b(?:on[- ]screen)\s+(?:text|words|writing|caption|title)\b",7.0),
        (r"\b(?:text|words|writing|caption|title)"
         r"\s+(?:on|in)\s+(?:the\s+)?(?:screen|video|frame)\b",7.0),
        (r"\b(?:sign|signboard|label|poster)\s+(?:reads|says|displays)\b",7.0),
        (r"\b(?:what|which)\s+(?:does|do)\s+(?:the\s+)?"
         r"(?:sign|screen|label|text)\s+(?:say|read)\b",7.0),
        (r"\b(?:subtitles?|captions?)\s+(?:say|read|show|display)\b",6.0),],
    "audio":[
        (r"\b(?:narrator|speaker|presenter|voiceover)"
         r"\b.{0,65}\b(?:say|says|said|mention|mentions|announce|announces|"
         r"explain|explains|discuss|discusses)\b",5.0),
        (r"\b(?:spoken|said|heard|mentioned|announced|narrated)\b",3.0),
        (r"\b(?:someone|person|people)\s+(?:says|speaks|talks|mentions)\b",4.0),
        (r"\b(?:in\s+the\s+audio|in\s+the\s+narration|during\s+the\s+speech)\b",5.0),],
    "visual":[
        (r"\b(?:visible|shown|seen)\s+(?:on|in)"
         r"\s+(?:the\s+)?(?:screen|video|frame|footage)\b",3.0),
        (r"\b(?:a|an|the)\s+\w+(?:\s+\w+){0,3}"
         r"\s+(?:is|are)\s+(?:visible|shown|seen)\b",3.0),
        (r"\b(?:someone|person|people)"
         r"\s+(?:walking|running|riding|standing|sitting|moving)\b",3.0),],
    "objects":[
        (r"\b(?:how many|number of|count the|count all)\b",4.0),
        (r"\b(?:detect|identify|locate)\s+(?:the\s+)?"
         r"(?:objects?|people|persons?|vehicles?|animals?)\b",3.0),],}

OCR_EXPLICIT={
    "text","written","subtitle","subtitles","caption","captions",
    "lettering","words","signboard","reads",}

VISUAL_AMBIGUOUS={
    "shown","shows","appear","appears","visible","displayed","seen",}

def tokenize(query):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?",query.lower())

def _add_signal(scores,matches,modality,name,strength):
    scores[modality]+=strength
    if name not in matches[modality]:
        matches[modality].append(name)

def _detect_phrase_signals(query,scores,matches):
    for modality,patterns in PHRASE_SIGNALS.items():
        for pattern,strength in patterns:
            if re.search(pattern,query):
                _add_signal(scores,matches,modality,"phrase:"+pattern,strength)

def _normalize_weights(scores):
    total=sum(scores[m] for m in MODALITIES)
    if total<=0:
        return {m:0.25 for m in MODALITIES}
    weights={m:round(scores[m]/total,4) for m in MODALITIES}
    largest=max(MODALITIES,key=lambda m:scores[m])
    difference=round(1.0-sum(weights.values()),4)
    weights[largest]=round(weights[largest]+difference,4)
    return weights

def _analyze_legacy(query):
    normalized_query=" ".join(query.lower().split())
    tokens=tokenize(query)
    token_set=set(tokens)
    scores=defaultdict(float)
    matches=defaultdict(list)
    for modality in MODALITIES:
        scores[modality]=PRIOR
    for modality,signals in QUERY_SIGNALS.items():
        for word,strength in signals.items():
            if word in token_set:
                _add_signal(scores,matches,modality,word,strength)
    _detect_phrase_signals(normalized_query,scores,matches)
    has_audio=bool(matches["audio"])
    has_visual=bool(matches["visual"])
    has_ocr=bool(matches["ocr"])
    has_objects=bool(matches["objects"])
    explicit_ocr=bool(token_set&OCR_EXPLICIT)
    strong_ocr_phrase=any(signal.startswith("phrase:") for signal in matches["ocr"])
    text_focused=explicit_ocr or strong_ocr_phrase
    if text_focused:
        for word in VISUAL_AMBIGUOUS:
            if word in matches["visual"]:
                scores["visual"]-=QUERY_SIGNALS["visual"].get(word,0.0)
                matches["visual"].remove(word)
        if not (token_set&{"scene","footage","wearing","moving","walking","running","riding","flying"}):
            for signal in list(matches["visual"]):
                if signal.startswith("phrase:"):
                    pattern=signal[len("phrase:"):]
                    for candidate,strength in PHRASE_SIGNALS["visual"]:
                        if candidate==pattern:
                            scores["visual"]-=strength
                            matches["visual"].remove(signal)
                            break
        scores["visual"]=max(PRIOR,scores["visual"])
        has_visual=bool(matches["visual"])
    if has_audio and (has_visual or has_ocr):
        query_type="multimodal"
    elif text_focused:
        query_type="ocr"
    elif has_audio:
        query_type="audio"
    elif has_visual:
        query_type="visual"
    elif has_objects:
        query_type="objects"
    else:
        query_type="visual"
    if query_type=="objects":
        scores["visual"]+=2.0
    if query_type=="ocr":
        scores["ocr"]+=3.0
        scores["visual"]+=0.5
    if query_type=="visual":
        scores["visual"]+=1.0
    if query_type=="audio":
        scores["audio"]+=1.0
    weights=_normalize_weights(scores)
    return {
        "query":query,
        "tokens":tokens,
        "query_type":query_type,
        "signal_scores":{m:round(scores[m],4) for m in MODALITIES},
        "modality_weights":weights,
        "matched_signals":{m:matches[m] for m in MODALITIES},}

TEMPORAL_CONNECTORS=("while","whilst","as","when","during")

AUDIO_SUBJECT_PATTERN=re.compile(
    r"\b(?:narrator|speaker|presenter|voiceover|"
    r"narration|audio|commentator)\b",re.IGNORECASE)

AUDIO_ACTION_PATTERN=re.compile(
    r"\b(?:"
    r"say|says|saying|said|"
    r"tell|tells|telling|told|"
    r"ask|asks|asking|asked|"
    r"instruct|instructs|instructing|instructed|"
    r"warn|warns|warning|warned|"
    r"mention|mentions|mentioned|mentioning|"
    r"announce|announces|announced|announcing|"
    r"explain|explains|explained|explaining|"
    r"discuss|discusses|discussed|discussing|"
    r"talk|talks|talked|talking|"
    r"speak|speaks|speaking|spoke|"
    r"describe|describes|described|describing|"
    r"narrate|narrates|narrated|narrating"
    r")\b",re.IGNORECASE)

OCR_INTENT_PATTERN=re.compile(
    r"\b(?:words?|text|subtitles?|captions?|"
    r"letters?|signboard|lettering|written|"
    r"on-screen text)\b",re.IGNORECASE)

VISUAL_INTENT_PATTERN=re.compile(
    r"\b(?:visible|shown|shows|seen|appears?|"
    r"footage|scene|walk(?:s|ing)?|"
    r"run(?:s|ning)?|rid(?:e|es|ing)|"
    r"fl(?:y|ies|ying)|land(?:s|ing)?|"
    r"jump(?:s|ing)?|standing|sitting|"
    r"moving|wearing)\b",re.IGNORECASE)

def _clean_clause(text):
    text=re.sub(r"\s+"," ",text).strip()
    text=text.strip(" ,.;:-")
    return text

def _classify_condition(text):
    if not text:
        return "unknown",0.0
    has_audio_subject=bool(AUDIO_SUBJECT_PATTERN.search(text))
    has_audio_action=bool(AUDIO_ACTION_PATTERN.search(text))
    has_ocr=bool(OCR_INTENT_PATTERN.search(text))
    has_visual=bool(VISUAL_INTENT_PATTERN.search(text))
    if has_audio_subject and has_audio_action:
        return "audio",0.95
    if has_ocr and not has_audio_subject:
        return "ocr",0.90
    if has_audio_subject:
        return "audio",0.80
    if has_visual:
        return "visual",0.85
    if has_audio_action:
        return "audio",0.65
    return "visual",0.55

def _split_temporal_clauses(query):
    pattern=r"\b("+"|".join(TEMPORAL_CONNECTORS)+r")\b"
    matches=list(re.finditer(pattern,query,flags=re.IGNORECASE))
    if len(matches)!=1:
        return None
    match=matches[0]
    left=_clean_clause(query[:match.start()])
    right=_clean_clause(query[match.end():])
    if not left or not right:
        return None
    return left,right,match.group(1).lower()

def _make_condition(condition_id,modality,text,confidence,required=True):
    return {
        "condition_id":condition_id,
        "modality":modality,
        "text":text,
        "required":required,
        "extraction_confidence":round(confidence,4),}

def extract_conditions(query,query_type=None):
    if not isinstance(query,str) or not query.strip():
        raise ValueError("Query must be a nonempty string.")
    query=_clean_clause(query)
    if query_type is None:
        query_type=_analyze_legacy(query)["query_type"]
    split=_split_temporal_clauses(query)
    if split is not None:
        left,right,connector=split
        left_modality,left_confidence=_classify_condition(left)
        right_modality,right_confidence=_classify_condition(right)
        distinct_modalities=left_modality!=right_modality
        supported_modalities=left_modality in MODALITIES and right_modality in MODALITIES
        sufficient_confidence=left_confidence>=0.80 and right_confidence>=0.80
        if distinct_modalities and supported_modalities and sufficient_confidence:
            conditions=[
                _make_condition("C001",left_modality,left,left_confidence),
                _make_condition("C002",right_modality,right,right_confidence),]
            required_modalities=list(dict.fromkeys(condition["modality"] for condition in conditions))
            return {
                "conditions":conditions,
                "required_modalities":required_modalities,
                "temporal_relation":"overlap",
                "temporal_connector":connector,
                "condition_extraction_confidence":round(min(left_confidence,right_confidence),4),
                "decomposition_status":"decomposed",}
    if query_type=="ocr":
        modality="ocr"
    elif query_type=="audio":
        modality="audio"
    elif query_type in ("visual","objects"):
        modality="visual"
    elif query_type=="multimodal":
        modality="multimodal"
    else:
        modality="general"
    confidence=0.85 if modality in MODALITIES else 0.40
    condition=_make_condition("C001",modality,query,confidence)
    return {
        "conditions":[condition],
        "required_modalities":[modality] if modality in MODALITIES else [],
        "temporal_relation":"unspecified",
        "temporal_connector":None,
        "condition_extraction_confidence":confidence,
        "decomposition_status":"single_condition",}

def extract_mention_target(query):
    if not isinstance(query,str):
        return None
    text=query.strip()
    if not text:
        return None
    patterns=[
        r"^(?:where|when)\s+(?:was|were|is|are)\s+(.+?)\s+mentioned\s*\??$",
        r"^(?:where|when)\s+does\s+(?:the\s+)?(?:narrator|speaker)\s+(?:say|mention)\s+(.+?)\s*\??$",
        r"^(?:find|locate)\s+(?:where\s+)?(?:the\s+)?(?:word|phrase)\s+(.+?)\s+(?:is\s+)?(?:said|spoken|mentioned)\s*\??$",]
    for pattern in patterns:
        match=re.fullmatch(pattern,text,flags=re.IGNORECASE)
        if not match:
            continue
        target=match.group(1).strip(" \t\n\r\"'?.!")
        if target:
            return target
    return None

def analyze_query(query):
    if not isinstance(query,str) or not query.strip():
        raise ValueError("Query must be a nonempty string.")
    result=_analyze_legacy(query)
    structured=extract_conditions(query,query_type=result["query_type"])
    result.update(structured)
    result["mention_target"]=extract_mention_target(query)
    return result

def print_analysis(result):
    print("\n"+"="*65)
    print("QUERY:",result["query"])
    print("TYPE:",result["query_type"])
    print("-"*65)
    print("\nModality weights:")
    for modality,weight in result["modality_weights"].items():
        print(f"  {modality:10s}: {weight:.4f}")
    print("\nConditions:")
    for condition in result["conditions"]:
        print(f"  [{condition['condition_id']}] {condition['modality'].upper()}")
        print(f"       {condition['text']}")
        print(f"       confidence={condition['extraction_confidence']:.2f}")
    print("\nRequired modalities:",result["required_modalities"])
    print("Temporal relation:",result["temporal_relation"])
    print("Decomposition:",result["decomposition_status"])

def run_self_tests():
    cases=[
        {
            "query":"A rocket is shown while the narrator discusses the Artemis mission",
            "modalities":{"visual","audio"},
            "relation":"overlap",},
        {
            "query":"A spacecraft is shown while the narrator describes the mission",
            "modalities":{"visual","audio"},
            "relation":"overlap",},
        {
            "query":"The narrator discusses the mission's launch preparations",
            "modalities":{"audio"},
            "relation":"unspecified",},
        {
            "query":"An astronaut is visible",
            "modalities":{"visual"},
            "relation":"unspecified",},
        {
            "query":"The words Artemis II appear on the screen",
            "modalities":{"ocr"},
            "relation":"unspecified",},
        {
            "query":"A person rides a bicycle through a city",
            "modalities":{"visual"},
            "relation":"unspecified",},
        {
            "query":"A helicopter lands on the Moon",
            "modalities":{"visual"},
            "relation":"unspecified",},]
    for case in cases:
        result=analyze_query(case["query"])
        assert set(result["required_modalities"])==case["modalities"],"Incorrect modalities for: "+case["query"]
        assert result["temporal_relation"]==case["relation"],"Incorrect temporal relation for: "+case["query"]
        assert abs(sum(result["modality_weights"].values())-1.0)<1e-3
        for key in ("query","tokens","query_type","signal_scores","modality_weights","matched_signals"):
            assert key in result
    print("\nPASS: all structured query-analysis self-tests passed.")

if __name__=="__main__":
    run_self_tests()
    test_queries=[
        "The narrator announces the first launch window for the Artemis II mission",
        "A rocket is visible on the screen",
        "The words Artemis II appear on the screen",
        "A rocket is shown while the narrator discusses the Artemis mission",
        "An astronaut is visible",
        "The narrator discusses the mission's launch preparations",
        "A helicopter lands on the Moon",
        "A person rides a bicycle through a city",
        "The narrator explains how astronauts will travel around the Moon",
        "A spacecraft is shown while the narrator describes the mission",
        "The subtitles display the words welcome home",
        "A person walks while the speaker explains the process",]
    for query in test_queries:
        print_analysis(analyze_query(query))