import re
from dataclasses import asdict,dataclass
from typing import Literal

QueryIntent=Literal["moment_search","video_qa","hybrid"]

@dataclass(frozen=True)
class RoutingResult:
    intent:QueryIntent
    confidence:float
    reason:str

    def to_dict(self)->dict:
        return asdict(self)

class QueryIntentRouter:
    LOCATION_PATTERNS=[
        r"\bwhen\s+(?:in|during)\s+(?:the|this)\s+video\b",
        r"\bwhen\s+(?:does|did|do|is|was|were)\b",
        r"\bwhere\s+(?:does|did|do|is|was|were)\b",
        r"\b(?:find|locate)\b",
        r"\b(?:jump|skip|seek)\s+to\b",
        r"\btake\s+me\s+to\b",
        r"\bshow\s+me\s+(?:the\s+)?(?:part|moment|scene|clip|section)\b",
        r"\bat\s+what\s+(?:time|point|timestamp)\b",
        r"\b(?:which|what)\s+(?:part|section|moment|scene)\b",
        r"\b(?:timestamp|timecode)\b",
        r"\bwhere\s+in\s+(?:the|this)\s+video\b",
        r"\b(?:part|moment|scene)\s+where\b",
        r"\b(?:point|time)\s+(?:in|of)\s+(?:the|this)\s+video\b",]

    ANSWER_PATTERNS=[
        r"\bwhat\s+(?:does|did|do|is|are|was|were)\b",
        r"\bwhy\b",
        r"\bhow\b",
        r"\bexplain\b",
        r"\bsummariz(?:e|es|ed|ing|ation)\b",
        r"\bdescribe\b",
        r"\bcompare\b",
        r"\blist\b",
        r"\bwho\b",
        r"\bwhat\s+are\s+the\s+(?:main|key)\b",
        r"\bwhat\s+happened\b",
        r"\bwhat\s+can\s+we\s+(?:learn|understand)\b",]

    EXPLICIT_HYBRID_PATTERNS=[
        r"\b(?:explain|summarize|describe|compare)\b.*"
        r"\b(?:show|find|locate|timestamp|where|when)\b",
        r"\b(?:what|why|how)\b.*"
        r"\b(?:and|also)\b.*"
        r"\b(?:show|find|locate|timestamp)\b",
        r"\b(?:answer|explain|summarize|describe|compare)\b.*"
        r"\b(?:with|along\s+with)\b.*"
        r"\b(?:timestamps?|video\s+moments?)\b",
        r"\b(?:find|locate|show)\b.*"
        r"\b(?:and|then)\b.*"
        r"\b(?:explain|summarize|describe)\b",]

    def classify(self,query:str)->RoutingResult:
        if not isinstance(query,str) or not query.strip():
            raise ValueError("Query must be a nonempty string.")
        normalized=" ".join(query.lower().split())
        if self._matches(normalized,self.EXPLICIT_HYBRID_PATTERNS):
            return RoutingResult(
                intent="hybrid",
                confidence=0.95,
                reason="User explicitly requests both an answer and video moment localization.",)
        has_location=self._matches(normalized,self.LOCATION_PATTERNS)
        has_answer=self._matches(normalized,self.ANSWER_PATTERNS)
        if has_location:
            return RoutingResult(
                intent="moment_search",
                confidence=0.90,
                reason="User is asking to locate a moment or event within the video.",)
        if has_answer:
            return RoutingResult(
                intent="video_qa",
                confidence=0.90,
                reason="User is requesting an explanation, summary, or factual answer.",)
        return RoutingResult(
            intent="moment_search",
            confidence=0.55,
            reason="Descriptive query without explicit question-answering instructions.",)

    @staticmethod
    def _matches(query:str,patterns:list[str])->bool:
        return any(re.search(pattern,query) for pattern in patterns)

def route_query(query:str)->dict:
    return QueryIntentRouter().classify(query).to_dict()

def run_tests():
    router=QueryIntentRouter()
    test_cases=[
        ("When does NASA discuss the Artemis II launch window?","moment_search"),
        ("Find the moment where the rocket appears.","moment_search"),
        ("Where in the video does the presenter appear?","moment_search"),
        ("Show me the part where the rocket launches.","moment_search"),
        ("At what time is the spacecraft visible?","moment_search"),
        ("What does NASA say about the launch window?","video_qa"),
        ("Why is the Artemis II mission important?","video_qa"),
        ("Explain the launch schedule.","video_qa"),
        ("Summarize the key announcements.","video_qa"),
        ("What happened during the mission?","video_qa"),
        ("Explain the launch schedule and show me where it is discussed.","hybrid"),
        ("Summarize the announcement with timestamps.","hybrid"),
        ("Find the rocket launch and explain what happens.","hybrid"),
        ("What does NASA announce, and also show the moment?","hybrid"),
        ("Rocket launch footage","moment_search"),]

    print("="*72)
    print("VIDEOMIND — QUERY INTENT ROUTER")
    print("="*72)
    passed=0
    for query,expected in test_cases:
        result=router.classify(query)
        success=result.intent==expected
        if success:
            passed+=1
        status="PASS" if success else "FAIL"
        print(f"\n[{status}] {query}")
        print(f"  Expected: {expected}")
        print(f"  Actual:   {result.intent}")
        print(f"  Reason:   {result.reason}")
    print("\n"+"="*72)
    print(f"RESULT: {passed}/{len(test_cases)} tests passed")
    print("="*72)
    if passed!=len(test_cases):
        print(f"\nWARNING: {len(test_cases)-passed} routing test(s) failed.")

if __name__=="__main__":
    run_tests()