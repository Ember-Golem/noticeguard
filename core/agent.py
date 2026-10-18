"""Inference agent module for NoticeGuard.
Interacts with local Gemma via Ollama (http://127.0.0.1:11434) with zero remote telemetry.
Includes a rich Developer Mock Mode fallback for local development without Ollama.
"""
from dataclasses import dataclass, field
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any
import psutil
import requests

from config import OLLAMA_BASE_URL, DEFAULT_MODEL, OLLAMA_TIMEOUT_SECONDS
from core.guardrails import scan_for_prompt_injection, check_enforceability_query


@dataclass
class TriageExecutionResult:
    """Execution telemetry and structured output from notice analysis."""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    ram_mb: float = 0.0
    model_name: str = ""
    is_mock: bool = False
    error_message: Optional[str] = None
    injection_detected: bool = False


SYSTEM_PROMPT = """You are NoticeGuard, an offline legal notice and termination letter triage assistant.
Your goal is to calmly extract stated facts, deadlines, and identify ambiguities in short-form notices.
You are NOT a lawyer and CANNOT provide legal counsel.

CRITICAL INSTRUCTIONS:
1. Output valid JSON ONLY matching the requested schema. Do NOT wrap in markdown formatting or codeblocks.
2. In 'stated_facts', 'verbatim_quote' MUST be an EXACT copy-paste substring from the indicated passage_id. Do NOT rephrase or extrapolate.
3. In 'interpretations_and_concerns', identify harsh penalties, conflicting deadlines, or intimidation tactics. Keep them strictly separated from stated facts.
4. If the notice is missing key terms (e.g. governing law, arbitration seat, cure mechanism) or if the user asks a question the document does not answer, list it in 'unknown_or_silent'.
5. If the user query asks whether something is legally enforceable, valid in court, or predicts a lawsuit outcome, set status to 'REFUSED_VERDICT'.

JSON SCHEMA:
{
  "status": "ANALYZED" or "DOCUMENT_IS_SILENT" or "REFUSED_VERDICT",
  "calm_overview": "Plain-language summary of what this notice demands and why",
  "timeline": [
    {"deadline_or_timeframe": "...", "action_or_consequence": "...", "passage_id": "P1"}
  ],
  "stated_facts": [
    {"passage_id": "P1", "verbatim_quote": "Exact substring copied from P1", "plain_meaning": "Calm plain-English explanation"}
  ],
  "interpretations_and_concerns": [
    {"passage_id": "P2", "category": "Ambiguity | Harsh Penalty | Conflict | Intimidation Language", "explanation": "Why this warrants review (strictly separated from stated facts)"}
  ],
  "unknown_or_silent": [
    "List of relevant details or user-asked topics that the document is completely silent on"
  ],
  "lawyer_escalation_checklist": [
    "3 specific, actionable questions to ask a qualified legal professional"
  ]
}
"""


def check_ollama_status() -> Tuple[bool, List[str]]:
    """Pings local Ollama endpoint with 1.5-second timeout.
    Returns (is_online, available_models).
    """
    url = f"{OLLAMA_BASE_URL}/api/tags"
    try:
        resp = requests.get(url, timeout=OLLAMA_TIMEOUT_SECONDS)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            return True, models
    except Exception:
        pass
    return False, []


def _get_process_ram_mb() -> float:
    """Returns current process RSS memory in megabytes."""
    try:
        proc = psutil.Process(os.getpid())
        return round(proc.memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return 0.0


def _build_mock_response(
    passages: Dict[str, str],
    user_query: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates realistic schema-compliant triage output for Developer Mock Mode.
    Includes intentional exact quotes AND one intentional paraphrased quote to test the verifier.
    """
    full_text = " ".join(passages.values()).lower()

    # Case 1: Eviction notice (Notice 02)
    if "tenant" in full_text or "evict" in full_text or "security deposit" in full_text or "premises" in full_text:
        # Check passages for real substring candidates
        p1_quote = "FORMAL NOTICE OF LEASE TERMINATION"
        p2_quote = "three (3) calendar days"
        p3_quote = "forfeiture of the entirety of the security deposit"

        # Find actual keys if present
        for pid, txt in passages.items():
            if "lease termination" in txt.lower():
                p1_quote = txt[:min(50, len(txt))]
                break

        return {
            "status": "ANALYZED",
            "calm_overview": (
                "The landlord asserts lease violations (unauthorized sublease/nuisance complaints) "
                "and demands total vacation of the residential premises within 3 days, threatening locks changing "
                "and full forfeiture of the security deposit."
            ),
            "timeline": [
                {
                    "deadline_or_timeframe": "3 calendar days from service",
                    "action_or_consequence": "Complete vacation and surrender of keys, or landlord threatens lock-out",
                    "passage_id": "P2" if "P2" in passages else "P1",
                },
                {
                    "deadline_or_timeframe": "Immediate",
                    "action_or_consequence": "Total forfeiture of security deposit without itemized deductions",
                    "passage_id": "P3" if "P3" in passages else "P2",
                },
            ],
            "stated_facts": [
                {
                    "passage_id": "P1" if "P1" in passages else list(passages.keys())[0],
                    "verbatim_quote": p1_quote if p1_quote in passages.get("P1", "") else list(passages.values())[0][:40],
                    "plain_meaning": "The notice formally asserts termination of the residential tenancy.",
                },
                {
                    "passage_id": "P2" if "P2" in passages else list(passages.keys())[0],
                    "verbatim_quote": "three (3) calendar days" if "three (3) calendar days" in passages.get("P2", "") else list(passages.values())[0][:20],
                    "plain_meaning": "The landlord provides a 3-day deadline before threatening self-help eviction.",
                },
                {
                    # Intentional paraphrase quote to test Substring Verifier!
                    "passage_id": "P3" if "P3" in passages else list(passages.keys())[-1],
                    "verbatim_quote": "The entire security deposit shall be seized immediately without any repair accounting",
                    "plain_meaning": "The landlord declares deposit forfeiture without providing damage itemization.",
                },
            ],
            "interpretations_and_concerns": [
                {
                    "passage_id": "P2" if "P2" in passages else "P1",
                    "category": "Intimidation Language",
                    "explanation": "Threatening lock-outs and unilateral eviction without a court order often violates state tenancy protection statutes.",
                },
                {
                    "passage_id": "P3" if "P3" in passages else "P2",
                    "category": "Harsh Penalty",
                    "explanation": "Automatic blanket forfeiture of deposits without itemized receipts or proof of physical damage is widely restricted under landlord-tenant law.",
                },
            ],
            "unknown_or_silent": [
                "The notice fails to provide an itemized repair estimate or receipt for alleged damages.",
                "The document is silent on any statutory right to cure or contest the alleged lease violation.",
                "No reference to judicial eviction proceedings or formal court filing summons.",
            ],
            "lawyer_escalation_checklist": [
                "Does state/municipal law permit self-help lock-outs without a judicial writ of possession?",
                "What statutory notice period (e.g. 15 or 30 days) is legally mandated before tenancy termination for alleged nuisance?",
                "What formal demand letter should be sent to require escrow deposit accounting within the statutory window?",
            ],
        }

    # Case 2: Employment termination & training bond recovery notice (Notice 01 / default)
    # Check if P1 / P2 / P4 exist
    p1_sample = passages.get("P1", "")
    p2_sample = passages.get("P2", "")
    p3_sample = passages.get("P3", "")
    p4_sample = passages.get("P4", "")

    # Exact substrings from sample notice 01
    q1 = "IMMEDIATE TERMINATION AND RECOVERY NOTICE" if "IMMEDIATE TERMINATION" in p1_sample else (list(passages.values())[0][:50] if passages else "")
    q2 = "Rs. 1,50,000" if "Rs. 1,50,000" in (p2_sample + p3_sample + p4_sample) else "1,50,000"

    # Intentional paraphrase for verification test
    q_paraphrase = "The employee must immediately repay the full training bond within 48 hours without objection"

    return {
        "status": "ANALYZED",
        "calm_overview": (
            "The employer asserts immediate termination of employment for alleged breach of agreement "
            "and demands repayment of a Rs. 1,50,000 training bond within 48 hours, under threat of legal action and industry blacklisting."
        ),
        "timeline": [
            {
                "deadline_or_timeframe": "48 hours from receipt",
                "action_or_consequence": "Remit payment of Rs. 1,50,000 to avoid civil recovery proceedings and reporting to industry verification databases",
                "passage_id": "P4" if "P4" in passages else "P2",
            },
            {
                "deadline_or_timeframe": "30 days cure period",
                "action_or_consequence": "Standard contractual cure timeline mentioned earlier in agreement",
                "passage_id": "P2" if "P2" in passages else "P1",
            },
        ],
        "stated_facts": [
            {
                "passage_id": "P1" if "P1" in passages else list(passages.keys())[0],
                "verbatim_quote": q1 if q1 in passages.get("P1", "") else list(passages.values())[0][:35],
                "plain_meaning": "The notice states the employment relationship is terminated effective immediately.",
            },
            {
                "passage_id": "P3" if "P3" in passages else ("P2" if "P2" in passages else list(passages.keys())[0]),
                "verbatim_quote": "Rs. 1,50,000" if "Rs. 1,50,000" in passages.get("P3", "") else "1,50,000",
                "plain_meaning": "The document specifies a liquidated training cost recovery demand of Rs. 1,50,000.",
            },
            {
                # Intentional paraphrase to demonstrate UNVERIFIED_PARAPHRASE detection!
                "passage_id": "P4" if "P4" in passages else list(passages.keys())[-1],
                "verbatim_quote": q_paraphrase,
                "plain_meaning": "The employer demands unconditional immediate remittance under threat of legal action.",
            },
        ],
        "interpretations_and_concerns": [
            {
                "passage_id": "P2 & P4" if "P4" in passages else "P2",
                "category": "Conflict",
                "explanation": "Contradiction between the 30-day cure period referenced in clause 2 and the abrupt 48-hour payment ultimatum in clause 4.",
            },
            {
                "passage_id": "P4" if "P4" in passages else "P2",
                "category": "Intimidation Language",
                "explanation": "Threatening to flag the employee on industry background verification databases or blacklist them warrants legal scrutiny regarding defamation and restraint of trade.",
            },
            {
                "passage_id": "P3" if "P3" in passages else "P1",
                "category": "Harsh Penalty",
                "explanation": "Demanding full recovery of fixed bond amounts without showing actual, prorated training expenditure or itemized receipts.",
            },
        ],
        "unknown_or_silent": [
            "The document is silent on the designated arbitration seat, venue, or governing court jurisdiction.",
            "No itemized proof of actual training expenses, invoices, or third-party certification costs incurred by the employer.",
            "Silent on employee's statutory right to full and final settlement (e.g. earned unpaid wages, PF/gratuity offset).",
        ],
        "lawyer_escalation_checklist": [
            "Under Section 27 / Section 74 of the Contract Act, is an employment bond enforceable without proof of actual pecuniary loss?",
            "Can an employer legally withhold statutory dues or threaten employment verification reporting for an unadjudicated bond claim?",
            "What structured reply letter should be issued to dispute the 48-hour ultimatum and demand an itemized reconciliation of training costs?",
        ],
    }


def analyze_notice(
    passages: Dict[str, str],
    user_query: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    use_mock: bool = False,
) -> TriageExecutionResult:
    """Performs structured triage analysis on notice passages using either
    local Ollama inference or Developer Mock Mode.
    """
    start_time = time.time()
    initial_ram = _get_process_ram_mb()

    # Step 1: Guardrail Check for Enforceability Queries
    is_refusal, refusal_payload = check_enforceability_query(user_query)
    if is_refusal:
        latency = round((time.time() - start_time) * 1000, 1)
        final_ram = _get_process_ram_mb()
        return TriageExecutionResult(
            success=True,
            data=refusal_payload,
            latency_ms=latency,
            ram_mb=final_ram,
            model_name="Guardrail Pre-Check (Zero-LLM)",
            is_mock=use_mock,
        )

    # Step 2: Guardrail Scan for Prompt Injection
    combined_raw = " ".join(passages.values())
    injection_scan = scan_for_prompt_injection(combined_raw)

    # Step 3: Developer Mock Mode Execution
    if use_mock:
        # Simulate realistic latency (120 - 240 ms)
        time.sleep(0.15)
        latency = round((time.time() - start_time) * 1000, 1)
        final_ram = _get_process_ram_mb()

        # Handle Document is Silent Quick Test
        if user_query and any(keyword in user_query.lower() for keyword in ["arbitration", "venue", "seat", "interest", "silent", "missing"]):
            mock_data = _build_mock_response(passages, user_query)
            mock_data["status"] = "DOCUMENT_IS_SILENT"
            mock_data["unknown_or_silent"].insert(
                0,
                f"DOCUMENT IS ENTIRELY SILENT on user query: '{user_query}'. No clause, appendix, or mention addresses this term."
            )
            return TriageExecutionResult(
                success=True,
                data=mock_data,
                latency_ms=latency,
                ram_mb=final_ram,
                model_name=f"{model_name} (Mock Engine)",
                is_mock=True,
                injection_detected=injection_scan.has_injection,
            )

        mock_data = _build_mock_response(passages, user_query)
        return TriageExecutionResult(
            success=True,
            data=mock_data,
            latency_ms=latency,
            ram_mb=final_ram,
            model_name=f"{model_name} (Mock Engine)",
            is_mock=True,
            injection_detected=injection_scan.has_injection,
        )

    # Step 4: Live Ollama Inference (http://127.0.0.1:11434)
    # Assemble structured prompt with isolation tags
    formatted_passages = "\n\n".join([f"[{pid}]: {text}" for pid, text in passages.items()])
    safe_passages_block = (
        f"<untrusted_notice_text>\n{formatted_passages}\n</untrusted_notice_text>"
    )

    user_prompt = f"Analyze the following legal notice passages:\n{safe_passages_block}"
    if user_query:
        user_prompt += f"\n\nUSER QUESTION: {user_query}\nIf the notice does not answer this question, state that in unknown_or_silent."

    payload = {
        "model": model_name,
        "prompt": user_prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
        },
    }

    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        resp = requests.post(url, json=payload, timeout=60.0)

        latency = round((time.time() - start_time) * 1000, 1)
        final_ram = _get_process_ram_mb()

        if resp.status_code != 200:
            return TriageExecutionResult(
                success=False,
                error_message=f"Ollama returned HTTP error {resp.status_code}: {resp.text}",
                latency_ms=latency,
                ram_mb=final_ram,
                model_name=model_name,
                is_mock=False,
            )

        resp_data = resp.json()
        raw_output = resp_data.get("response", "{}")

        # Parse JSON
        parsed_json = json.loads(raw_output)

        # Validate core schema fields
        for field in ["status", "calm_overview", "timeline", "stated_facts", "interpretations_and_concerns", "unknown_or_silent", "lawyer_escalation_checklist"]:
            if field not in parsed_json:
                parsed_json[field] = [] if field in ["timeline", "stated_facts", "interpretations_and_concerns", "unknown_or_silent", "lawyer_escalation_checklist"] else ""

        return TriageExecutionResult(
            success=True,
            data=parsed_json,
            latency_ms=latency,
            ram_mb=final_ram,
            model_name=model_name,
            is_mock=False,
            injection_detected=injection_scan.has_injection,
        )

    except json.JSONDecodeError as jde:
        latency = round((time.time() - start_time) * 1000, 1)
        final_ram = _get_process_ram_mb()
        return TriageExecutionResult(
            success=False,
            error_message=f"Failed to parse model response as JSON: {str(jde)}",
            latency_ms=latency,
            ram_mb=final_ram,
            model_name=model_name,
            is_mock=False,
        )
    except requests.exceptions.RequestException as req_err:
        latency = round((time.time() - start_time) * 1000, 1)
        final_ram = _get_process_ram_mb()
        return TriageExecutionResult(
            success=False,
            error_message=f"Could not connect to Ollama at {OLLAMA_BASE_URL}: {str(req_err)}. Switch to Developer Mock Mode to test without Ollama.",
            latency_ms=latency,
            ram_mb=final_ram,
            model_name=model_name,
            is_mock=False,
        )
