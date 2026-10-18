"""Guardrails module for NoticeGuard.
Provides adversarial prompt injection scanning, enforceability question refusal,
and cross-passage numeric/timeframe conflict detection.
"""
from dataclasses import dataclass
import html
import re
from typing import Dict, List, Tuple, Any

# Common adversarial injection signatures
INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?",
    r"system\s+(?:override|prompt|instruction)",
    r"declare\s+this\s+(?:valid|enforceable|legal|void)",
    r"disregard\s+(?:all\s+)?(?:rules|guidelines|instructions)",
    r"you\s+are\s+now\s+(?:a|an)\s+",
    r"act\s+as\s+(?:a|an)\s+",
    r"jailbreak",
    r"bypass\s+safety",
    r"assistant\s+mode",
    r"output\s+only\s+the\s+following",
    r"tell\s+the\s+user\s+(?:this\s+notice\s+is|they\s+must)",
    r"must\s+pay\s+immediately",
]

COMPILED_INJECTION_REGEX = [
    re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS
]

# Enforceability and judicial conclusion queries that must trigger REFUSED_VERDICT
ENFORCEABILITY_PATTERNS = [
    r"\b(?:is|are)\s+(?:this|this\s+notice|clause\s+\w+|these)\s+(?:legally\s+)?enforceable\b",
    r"\blegally\s+enforceable\b",
    r"\bvalid\s+in\s+court\b",
    r"\b(?:is\s+clause\s+\w+|is\s+this\s+notice|is\s+this)\s+valid\b",
    r"\bwill\s+i\s+(?:get\s+sued|lose|win)\b",
    r"\bcan\s+they\s+(?:sue\s+me|arrest\s+me|jail\s+me|take\s+my)\b",
    r"\bwill\s+the\s+court\s+(?:uphold|rule)\b",
    r"\bis\s+this\s+(?:illegal|lawful|unlawful|binding)\b",
    r"\bwhat\s+are\s+my\s+chances\s+of\s+winning\b",
    r"\bdo\s+i\s+have\s+to\s+pay\b",
]

COMPILED_ENFORCEABILITY_REGEX = [
    re.compile(pattern, re.IGNORECASE) for pattern in ENFORCEABILITY_PATTERNS
]


@dataclass
class InjectionScanResult:
    has_injection: bool
    matched_patterns: List[str]
    sanitized_text: str
    prompt_wrapper: str


def scan_for_prompt_injection(text: str) -> InjectionScanResult:
    """Scans text for adversarial instructions attempting prompt injection.
    Wraps all untrusted notice text inside strict isolation tags.
    """
    if not text:
        return InjectionScanResult(
            has_injection=False,
            matched_patterns=[],
            sanitized_text="",
            prompt_wrapper="<untrusted_notice_text>\n</untrusted_notice_text>",
        )

    matched: List[str] = []
    for pattern in COMPILED_INJECTION_REGEX:
        found = pattern.findall(text)
        if found:
            matched.extend(found)

    # Sanitize closing tags to avoid injection breaking out of delimiter
    safe_text = text.replace("</untrusted_notice_text>", "[TAG_DEFUSED]")
    safe_text = safe_text.replace("<untrusted_notice_text>", "[TAG_DEFUSED]")

    # Delimit within explicit untrusted tag container
    prompt_wrapper = f"<untrusted_notice_text>\n{safe_text}\n</untrusted_notice_text>"

    return InjectionScanResult(
        has_injection=bool(matched),
        matched_patterns=list(dict.fromkeys(matched)),
        sanitized_text=safe_text,
        prompt_wrapper=prompt_wrapper,
    )


def check_enforceability_query(query: Optional[str]) -> Tuple[bool, Dict[str, Any]]:
    """Detects if user asks for a definitive legal verdict or court outcome.
    Returns (is_refusal, refusal_payload).
    """
    if not query or not query.strip():
        return False, {}

    clean_query = query.strip()
    is_refusal = any(pattern.search(clean_query) for pattern in COMPILED_ENFORCEABILITY_REGEX)

    if not is_refusal:
        return False, {}

    refusal_payload = {
        "status": "REFUSED_VERDICT",
        "calm_overview": (
            "NoticeGuard cannot make legal enforceability determinations or predict court outcomes. "
            "Whether a clause or legal demand is binding depends on local statutory public policy, "
            "jurisdiction, extrinsic factual context, and actual damages proven under contract law. "
            "Below is a calibrated summary of factual contents and specific questions for qualified legal counsel."
        ),
        "timeline": [],
        "stated_facts": [],
        "interpretations_and_concerns": [
            {
                "passage_id": "LEGAL_BOUNDARY",
                "category": "Refusal: Judicial Validity Assessment",
                "explanation": (
                    "Legal enforceability requires judicial interpretation. In many jurisdictions, "
                    "liquidated damages clauses or forfeiture demands that operate as penalties are "
                    "unenforceable unless proven as genuine pre-estimates of damage."
                ),
            }
        ],
        "unknown_or_silent": [
            "Governing statutory jurisdiction and arbitration venue.",
            "Evidence of actual quantifiable losses suffered by the issuing party.",
            "Any prior performance history, mitigating communications, or statutory cures."
        ],
        "lawyer_escalation_checklist": [
            "Does local statutory law (e.g. Section 74 of the Contract Act or local tenant rights acts) treat this demand as an unenforceable penalty?",
            "What written response should be served before the stated deadline to formally deny liability while requesting itemized loss accounting?",
            "What interim injunctive relief or conciliation mechanisms exist in this jurisdiction if unilateral enforcement or lockout is threatened?"
        ],
    }

    return True, refusal_payload


def detect_numeric_conflicts(passages: Dict[str, str]) -> List[Dict[str, Any]]:
    """Deterministically identifies potential numeric or timeframe conflicts across passages.
    E.g. P2 mentioning 30-day cure period vs P4 demanding payment in 48 hours.
    """
    from core.ingest import REGEX_TIMEFRAMES, REGEX_CURRENCY

    timeframe_map: Dict[str, List[str]] = {}
    currency_map: Dict[str, List[str]] = {}

    for pid, text in passages.items():
        tf = [m.group(0).lower().strip() for m in REGEX_TIMEFRAMES.finditer(text)]
        if tf:
            timeframe_map[pid] = list(dict.fromkeys(tf))

        cur = [m.group(0).strip() for m in REGEX_CURRENCY.finditer(text)]
        if cur:
            currency_map[pid] = list(dict.fromkeys(cur))

    conflicts: List[Dict[str, Any]] = []

    # Check timeframe discrepancies (e.g. hours vs days, 48 hours vs 30 days)
    if len(timeframe_map) >= 2:
        all_tf_entries = []
        for pid, tfs in timeframe_map.items():
            for t in tfs:
                all_tf_entries.append((pid, t))

        # Check if we have short urgent hours (e.g. 48 hours, 24 hours, 3 days) along with longer cure periods (e.g. 15 days, 30 days)
        short_urgent = [(pid, t) for pid, t in all_tf_entries if "hour" in t or t.startswith(("1 ", "2 ", "3 ", "48"))]
        cure_extended = [(pid, t) for pid, t in all_tf_entries if ("day" in t and any(num in t for num in ["15", "30", "60", "90"]))]

        if short_urgent and cure_extended:
            for s_pid, s_tf in short_urgent:
                for c_pid, c_tf in cure_extended:
                    if s_pid != c_pid:
                        conflicts.append({
                            "type": "Conflicting Notice Timeframes",
                            "severity": "HIGH",
                            "passages": [c_pid, s_pid],
                            "summary": (
                                f"Passage {c_pid} mentions a '{c_tf}' timeframe, while passage {s_pid} "
                                f"demands action within '{s_tf}'. This creates direct ambiguity regarding the legal cure deadline."
                            ),
                        })

    # Check currency discrepancies
    if len(currency_map) >= 2:
        distinct_amounts = set()
        for pid, amounts in currency_map.items():
            for a in amounts:
                # clean amount string for comparison
                cleaned = re.sub(r"[^\d.]", "", a)
                if cleaned:
                    distinct_amounts.add((cleaned, pid, a))

        if len({item[0] for item in distinct_amounts}) > 1:
            # Different amounts cited across passages
            amounts_list = list(distinct_amounts)
            conflicts.append({
                "type": "Discrepant Monetary Figures",
                "severity": "MEDIUM",
                "passages": list(dict.fromkeys([item[1] for item in amounts_list])),
                "summary": f"Varying monetary figures cited across document: {', '.join([f'{item[2]} in {item[1]}' for item in amounts_list])}.",
            })

    return conflicts
