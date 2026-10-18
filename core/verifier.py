"""Deterministic verification module for NoticeGuard.
Provides substring grounding verification for stated facts and numeric preservation checks.
Calibrated label: 'Deterministic Substring Check (Verifies quote presence in source passage)'.
"""
from dataclasses import dataclass, field
import re
from typing import Dict, List, Tuple, Any

EXACT_SUBSTRING_MATCH = "EXACT_SUBSTRING_MATCH"
UNVERIFIED_PARAPHRASE = "UNVERIFIED_PARAPHRASE_CHECK_SOURCE"


def _normalize_text(text: str) -> str:
    """Normalizes whitespace and removes extraneous outer quote marks."""
    if not text:
        return ""
    # Strip leading/trailing quotation marks if the LLM wrapped it
    cleaned = text.strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()
    if (cleaned.startswith("“") and cleaned.endswith("”")) or (
        cleaned.startswith("‘") and cleaned.endswith("’")
    ):
        cleaned = cleaned[1:-1].strip()
    # Normalize multiple whitespace characters to single spaces
    return re.sub(r"\s+", " ", cleaned).strip()


@dataclass
class VerificationReport:
    """Summary of deterministic quote verification."""
    total_quotes: int = 0
    exact_matches: int = 0
    unverified_paraphrases: int = 0
    grounding_score_pct: float = 0.0
    verified_items: List[Dict[str, Any]] = field(default_factory=list)


def verify_quotes(
    stated_facts: List[Dict[str, Any]],
    passages: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], VerificationReport]:
    """Deterministically checks if each verbatim_quote is an exact substring of the cited passage.
    Stamps each item with:
      verification_status: 'EXACT_SUBSTRING_MATCH' or 'UNVERIFIED_PARAPHRASE_CHECK_SOURCE'
    """
    if not stated_facts:
        return [], VerificationReport()

    verified_items: List[Dict[str, Any]] = []
    exact_count = 0
    unverified_count = 0

    for item in stated_facts:
        # Create a copy so we don't mutate unpredictably
        fact = dict(item)
        pid = fact.get("passage_id", "").strip()
        quote = fact.get("verbatim_quote", "")

        norm_quote = _normalize_text(quote)
        passage_text = passages.get(pid, "")
        norm_passage = _normalize_text(passage_text)

        # Check if passage exists and normalized quote is a substring
        if norm_quote and passage_text and norm_quote in norm_passage:
            fact["verification_status"] = EXACT_SUBSTRING_MATCH
            fact["verification_label"] = "✅ Deterministic Substring Match"
            fact["is_grounded"] = True
            exact_count += 1
        else:
            # Check if quote exists in any OTHER passage
            found_in_alt = None
            if norm_quote:
                for other_pid, other_text in passages.items():
                    if other_pid != pid and norm_quote in _normalize_text(other_text):
                        found_in_alt = other_pid
                        break

            fact["verification_status"] = UNVERIFIED_PARAPHRASE
            if found_in_alt:
                fact["verification_label"] = (
                    f"⚠️ Quote Misattributed (Found in [{found_in_alt}], not [{pid}])"
                )
            else:
                fact["verification_label"] = (
                    f"⚠️ Unverified Paraphrase — Quote not found verbatim in [{pid}]"
                )
            fact["is_grounded"] = False
            unverified_count += 1

        verified_items.append(fact)

    total = len(verified_items)
    pct = round((exact_count / total * 100.0), 1) if total > 0 else 0.0

    report = VerificationReport(
        total_quotes=total,
        exact_matches=exact_count,
        unverified_paraphrases=unverified_count,
        grounding_score_pct=pct,
        verified_items=verified_items,
    )

    return verified_items, report


def verify_number_preservation(source_text: str, target_text: str) -> Tuple[bool, List[str]]:
    """Regex helper (ready for translation or summary verification) to check
    whether all numeric tokens (amounts, days, percentages) in source_text
    are preserved in target_text.
    """
    if not source_text:
        return True, []

    # Extract all standalone numbers and currency figures
    number_pattern = re.compile(r"\b\d+(?:,\d+)*(?:\.\d+)?\b")
    source_numbers = set(number_pattern.findall(source_text))

    if not target_text:
        return False, sorted(list(source_numbers))

    target_numbers = set(number_pattern.findall(target_text))
    missing = [num for num in source_numbers if num not in target_numbers]

    return len(missing) == 0, sorted(missing)
