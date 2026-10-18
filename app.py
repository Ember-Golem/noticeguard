"""NoticeGuard: Local Legal Notice & Termination Letter Triage Agent
A modular, offline Streamlit application designed for local Gemma execution via Ollama
with automatic Developer Mock Mode fallback, loopback-only network isolation,
in-memory document ingestion, and deterministic substring quote verification.
"""
import gc
import os
import sys
import time
from typing import Dict, List, Optional, Any
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core.ingest import ingest_text, ingest_file
from core.guardrails import scan_for_prompt_injection, detect_numeric_conflicts
from core.agent import check_ollama_status, analyze_notice, _get_process_ram_mb
from core.verifier import verify_quotes, EXACT_SUBSTRING_MATCH, UNVERIFIED_PARAPHRASE
from privacy.loopback_check import run_loopback_audit

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NoticeGuard | Local Legal Notice Triage",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Grounded Legal Triage
st.markdown(
    """
    <style>
    /* Metric & Telemetry Bar Styling */
    .telemetry-container {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        background-color: #1e222d;
        border: 1px solid #363c4e;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 14px;
        align-items: center;
    }
    .telemetry-item {
        font-family: monospace;
        font-size: 0.85rem;
        color: #e2e8f0;
    }
    .badge-live {
        background-color: #10b981;
        color: #022c22;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-mock {
        background-color: #f59e0b;
        color: #451a03;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    /* Section Cards */
    .card-grounded {
        background: #111827;
        border-left: 4px solid #10b981;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .card-paraphrase {
        background: #18181b;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .card-concern {
        background: #1e1b2e;
        border-left: 4px solid #8b5cf6;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .card-silent {
        background: #1f2937;
        border-left: 4px solid #64748b;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .passage-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 12px;
        line-height: 1.5;
    }
    .passage-box-cited {
        background: #132238;
        border: 1px solid #38bdf8;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 12px;
        line-height: 1.5;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.15);
    }
    .passage-badge {
        display: inline-block;
        background-color: #0284c7;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 2px 7px;
        border-radius: 4px;
        margin-right: 6px;
    }
    .verbatim-text {
        font-style: italic;
        color: #cbd5e1;
        border-left: 3px solid #64748b;
        padding-left: 8px;
        margin: 6px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "active_passages" not in st.session_state:
    st.session_state.active_passages = {}
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "active_anchors" not in st.session_state:
    st.session_state.active_anchors = {}
if "triage_result" not in st.session_state:
    st.session_state.triage_result = None
if "verified_facts" not in st.session_state:
    st.session_state.verified_facts = []
if "verification_report" not in st.session_state:
    st.session_state.verification_report = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "privacy_audit" not in st.session_state:
    st.session_state.privacy_audit = None
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = "01_bond_recovery_termination_notice.txt"

# ---------------------------------------------------------------------------
# Header & Permanent Calibration Banner
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="background-color: #2e1a05; border: 1px solid #d97706; border-radius: 8px; padding: 12px 16px; margin-bottom: 14px;">
        <span style="font-size: 1.15rem; font-weight: 700; color: #fbbf24;">🛡️ NoticeGuard: Local Legal Notice & Termination Letter Triage</span>
        <div style="margin-top: 4px; color: #fde68a; font-size: 0.88rem; line-height: 1.4;">
            {config.PERMANENT_BOUNDARY_NOTICE}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: Engine Configuration & Document Input
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Engine & Document Input")

# 1. Ollama Connectivity & Mock Mode Check
is_ollama_online, available_models = check_ollama_status()

if is_ollama_online:
    st.sidebar.success("🟢 Local Ollama Online (`127.0.0.1:11434`)")
    model_choices = available_models if available_models else [config.DEFAULT_MODEL]
    selected_model = st.sidebar.selectbox("Local Gemma Model", model_choices, index=0)
    use_mock_mode = st.sidebar.checkbox(
        "🛠️ Force Developer Mock Mode",
        value=False,
        help="Simulate local inference on machines without live Gemma weights loaded.",
    )
else:
    st.sidebar.warning("⚠️ Local Ollama Not Detected at `127.0.0.1:11434`")
    st.sidebar.info("💡 Automatically operating in **Developer Mock Mode** for offline evaluation.")
    use_mock_mode = True
    selected_model = config.DEFAULT_MODEL

st.sidebar.markdown("---")

# 2. Document Ingestion Selector
input_mode = st.sidebar.radio(
    "Select Input Source",
    ["📂 Load Curated Sample Notice", "✍️ Paste Unscripted Text", "📄 Upload File (.txt / .pdf)"],
    index=0,
)

sample_options = {
    "01: Employment Bond Recovery & 48h Demand (Notice 01)": "01_bond_recovery_termination_notice.txt",
    "02: Residential Eviction & Deposit Forfeiture (Notice 02)": "02_tenant_eviction_and_forfeiture_notice.txt",
    "03: Adversarial Prompt Injection Attempt (Notice 03)": "03_adversarial_injection_notice.txt",
}

loaded_sample_path = None

if input_mode == "📂 Load Curated Sample Notice":
    sample_label = st.sidebar.selectbox("Choose Sample Notice", list(sample_options.keys()))
    sample_filename = sample_options[sample_label]
    sample_file_path = os.path.join(os.path.dirname(__file__), "sample_notices", sample_filename)

    if os.path.exists(sample_file_path):
        with open(sample_file_path, "r", encoding="utf-8") as f:
            sample_content = f.read()

        if st.sidebar.button("📥 Load Selected Sample", use_container_width=True) or not st.session_state.raw_text:
            ingest_res = ingest_text(sample_content)
            if ingest_res.success:
                st.session_state.raw_text = ingest_res.raw_text
                st.session_state.active_passages = ingest_res.passages
                st.session_state.active_anchors = ingest_res.anchors
                st.session_state.triage_result = None
                st.session_state.verified_facts = []
                st.session_state.verification_report = None
                st.session_state.selected_sample = sample_filename
                st.rerun()

elif input_mode == "✍️ Paste Unscripted Text":
    pasted_input = st.sidebar.text_area(
        "Paste Legal Notice (1-2 pages, <15,000 chars)",
        value=st.session_state.raw_text,
        height=220,
    )
    if st.sidebar.button("📥 Ingest Pasted Text", use_container_width=True):
        ingest_res = ingest_text(pasted_input)
        if ingest_res.success:
            st.session_state.raw_text = ingest_res.raw_text
            st.session_state.active_passages = ingest_res.passages
            st.session_state.active_anchors = ingest_res.anchors
            st.session_state.triage_result = None
            st.session_state.verified_facts = []
            st.rerun()
        else:
            st.sidebar.error(ingest_res.error_message)

elif input_mode == "📄 Upload File (.txt / .pdf)":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Notice (.txt or .pdf)",
        type=["txt", "pdf"],
        help="Files are processed strictly in memory via io.BytesIO. Zero temporary disk writes.",
    )
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        if st.sidebar.button("📥 Ingest Uploaded File", use_container_width=True):
            ingest_res = ingest_file(file_bytes, uploaded_file.name)
            if ingest_res.success:
                st.session_state.raw_text = ingest_res.raw_text
                st.session_state.active_passages = ingest_res.passages
                st.session_state.active_anchors = ingest_res.anchors
                st.session_state.triage_result = None
                st.session_state.verified_facts = []
                st.rerun()
            else:
                st.sidebar.error(ingest_res.error_message)

# Ensure sample notice 1 is loaded initially if empty
if not st.session_state.active_passages:
    default_sample = os.path.join(
        os.path.dirname(__file__), "sample_notices", "01_bond_recovery_termination_notice.txt"
    )
    if os.path.exists(default_sample):
        with open(default_sample, "r", encoding="utf-8") as f:
            ingest_res = ingest_text(f.read())
            if ingest_res.success:
                st.session_state.raw_text = ingest_res.raw_text
                st.session_state.active_passages = ingest_res.passages
                st.session_state.active_anchors = ingest_res.anchors

# Sidebar: Deterministic Anchors Expander
with st.sidebar.expander("📍 Extracted Regex Anchors", expanded=False):
    anchors = st.session_state.active_anchors
    if anchors:
        st.write("**Notice Timeframes:**")
        if anchors.get("timeframes"):
            for tf in anchors["timeframes"]:
                st.markdown(f"- ⏱️ `{tf}`")
        else:
            st.caption("None extracted")

        st.write("**Currency Figures:**")
        if anchors.get("currencies"):
            for cur in anchors["currencies"]:
                st.markdown(f"- 💰 `{cur}`")
        else:
            st.caption("None extracted")

        st.write("**Calendar Dates:**")
        if anchors.get("dates"):
            for d in anchors["dates"]:
                st.markdown(f"- 📅 `{d}`")
        else:
            st.caption("None extracted")
    else:
        st.caption("No anchors extracted yet.")

# ---------------------------------------------------------------------------
# Top Telemetry & Session Management Bar
# ---------------------------------------------------------------------------
col_tel1, col_tel2, col_tel3, col_tel4, col_tel5 = st.columns([2.5, 2, 1.5, 1.5, 1.5])

with col_tel1:
    if use_mock_mode:
        st.markdown(
            '<div class="telemetry-item">Mode: <span class="badge-mock">🛠️ Developer Mock Mode</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="telemetry-item">Mode: <span class="badge-live">🟢 Live Local Ollama (127.0.0.1)</span></div>',
            unsafe_allow_html=True,
        )

with col_tel2:
    st.markdown(
        f'<div class="telemetry-item">Model: <strong>{selected_model}</strong></div>',
        unsafe_allow_html=True,
    )

with col_tel3:
    last_latency = (
        f"{st.session_state.triage_result.latency_ms:.1f} ms"
        if st.session_state.triage_result
        else "0.0 ms"
    )
    st.markdown(
        f'<div class="telemetry-item">Latency: <strong>{last_latency}</strong></div>',
        unsafe_allow_html=True,
    )

with col_tel4:
    current_ram = _get_process_ram_mb()
    st.markdown(
        f'<div class="telemetry-item">Process RAM: <strong>{current_ram} MB</strong></div>',
        unsafe_allow_html=True,
    )

with col_tel5:
    if st.button("🗑️ Wipe Session RAM", help="Clears session state and triggers garbage collection"):
        st.session_state.triage_result = None
        st.session_state.verified_facts = []
        st.session_state.verification_report = None
        st.session_state.last_query = ""
        gc.collect()
        st.rerun()

st.markdown("<hr style='margin-top: 4px; margin-bottom: 16px;'/>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Prompt Injection Warning Banner
# ---------------------------------------------------------------------------
inj_check = scan_for_prompt_injection(st.session_state.raw_text)
if inj_check.has_injection:
    st.error(
        f"🚨 **Adversarial Prompt Injection Detected in Document!** "
        f"Embedded adversarial signatures detected: `{', '.join(inj_check.matched_patterns)}`. "
        "The document text has been safely quarantined inside `<untrusted_notice_text>` tags to prevent instruction override."
    )

# Numeric / Timeframe Conflicts Check
detected_conflicts = detect_numeric_conflicts(st.session_state.active_passages)

# ---------------------------------------------------------------------------
# Main Split-Screen View: Source Passages (Left) vs Triage Output (Right)
# ---------------------------------------------------------------------------
col_source, col_triage = st.columns([1.0, 1.25])

# Find which passages are cited in the triage output to provide visual highlighting
cited_pids = set()
if st.session_state.triage_result and st.session_state.triage_result.success:
    tdata = st.session_state.triage_result.data
    for item in tdata.get("timeline", []):
        if item.get("passage_id"):
            cited_pids.add(item["passage_id"])
    for item in tdata.get("stated_facts", []):
        if item.get("passage_id"):
            cited_pids.add(item["passage_id"])
    for item in tdata.get("interpretations_and_concerns", []):
        pid = item.get("passage_id", "")
        for part in pid.replace("&", ",").split(","):
            cleaned = part.strip()
            if cleaned:
                cited_pids.add(cleaned)

# ===========================================================================
# LEFT COLUMN: Source Passages & Ground Truth Anchors
# ===========================================================================
with col_source:
    st.subheader("📄 Indexed Source Passages")
    st.caption("Passages deterministically indexed and referenced as [P1], [P2], etc.")

    if detected_conflicts:
        for conf in detected_conflicts:
            st.warning(f"⚠️ **{conf['type']} ({', '.join(conf['passages'])})**: {conf['summary']}")

    passages = st.session_state.active_passages
    if passages:
        for pid, text in passages.items():
            is_cited = pid in cited_pids
            box_class = "passage-box-cited" if is_cited else "passage-box"
            highlight_indicator = " ⭐ (Cited in Triage)" if is_cited else ""

            st.markdown(
                f"""
                <div class="{box_class}">
                    <span class="passage-badge">[{pid}]</span><strong>{highlight_indicator}</strong>
                    <div style="margin-top: 6px; font-size: 0.92rem; color: #f1f5f9;">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No passages loaded. Load a sample notice from the sidebar.")

# ===========================================================================
# RIGHT COLUMN: Quick-Test Buttons & Structured Triage
# ===========================================================================
with col_triage:
    st.subheader("🎯 Triage Analysis & Verification")

    # 4 Quick-Test Buttons for Judges
    st.markdown("**⚡ Quick-Test Buttons for Judges:**")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)

    trigger_triage = False
    run_query = None

    with qcol1:
        if st.button("1. Run Full Triage", use_container_width=True, help="Executes full notice parsing"):
            trigger_triage = True
            run_query = None

    with qcol2:
        if st.button("2. Ask Unanswered Question", use_container_width=True, help="Test Document is Silent detection"):
            trigger_triage = True
            run_query = "What is the designated arbitration seat or governing city for this contract?"

    with qcol3:
        if st.button("3. Enforceability Test", use_container_width=True, help="Test REFUSED_VERDICT guardrail"):
            trigger_triage = True
            run_query = "Is this notice legally enforceable in court?"

    with qcol4:
        if st.button("4. Loopback Audit", use_container_width=True, help="Audit active process network sockets"):
            st.session_state.privacy_audit = run_loopback_audit()

    # Optional custom inquiry input
    custom_query = st.text_input(
        "Ask a specific question about this notice (or leave blank for full triage):",
        value="",
        placeholder="e.g., What is the payment deadline, or what arbitration seat is specified?",
    )
    if custom_query:
        if st.button("🔍 Submit Specific Question"):
            trigger_triage = True
            run_query = custom_query

    # Execute Analysis if triggered
    if trigger_triage:
        with st.spinner("Analyzing notice via offline engine..."):
            exec_res = analyze_notice(
                passages=st.session_state.active_passages,
                user_query=run_query,
                model_name=selected_model,
                use_mock=use_mock_mode,
            )
            st.session_state.triage_result = exec_res
            st.session_state.last_query = run_query or ""

            if exec_res.success and "stated_facts" in exec_res.data:
                # Deterministic quote verification
                verified_items, v_report = verify_quotes(
                    exec_res.data.get("stated_facts", []),
                    st.session_state.active_passages,
                )
                st.session_state.verified_facts = verified_items
                st.session_state.verification_report = v_report
            st.rerun()

    # -----------------------------------------------------------------------
    # Display Triage Output
    # -----------------------------------------------------------------------
    res = st.session_state.triage_result

    if res and res.success:
        data = res.data
        status = data.get("status", "ANALYZED")

        # Status Badge
        if status == "REFUSED_VERDICT":
            st.error("🛑 **Triage Status: REFUSED_VERDICT (Legal Advice Boundary Guardrail Triggered)**")
        elif status == "DOCUMENT_IS_SILENT":
            st.warning("⚠️ **Triage Status: DOCUMENT_IS_SILENT (Target Subject Not Addressed in Text)**")
        else:
            st.success("✅ **Triage Status: ANALYZED (Grounded Document Parsing)**")

        # Calm Overview
        if data.get("calm_overview"):
            st.markdown(
                f"""
                <div style="background-color: #1a2234; border: 1px solid #2563eb; border-radius: 8px; padding: 12px 16px; margin: 12px 0;">
                    <div style="font-weight: 700; color: #60a5fa; margin-bottom: 4px;">📋 Calm Overview & Demand Summary:</div>
                    <div style="color: #e2e8f0; font-size: 0.95rem; line-height: 1.5;">{data['calm_overview']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------------------------------------------------
        # Section 1: Chronological Action & Deadline Timeline
        # -------------------------------------------------------------------
        st.markdown("### ⏱️ Section 1: Chronological Action & Deadline Timeline")
        timeline = data.get("timeline", [])
        if timeline:
            for item in timeline:
                pid = item.get("passage_id", "P?")
                deadline = item.get("deadline_or_timeframe", "Unspecified")
                action = item.get("action_or_consequence", "")
                st.markdown(
                    f"""
                    <div style="background: #1e293b; border-left: 4px solid #38bdf8; padding: 8px 12px; margin-bottom: 6px; border-radius: 4px;">
                        <span class="passage-badge">[{pid}]</span>
                        <strong style="color: #38bdf8;">{deadline}</strong>: 
                        <span style="color: #cbd5e1;">{action}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No specific time-bound milestones detected.")

        # -------------------------------------------------------------------
        # Section 2: Three-Part Grounded Breakdown
        # -------------------------------------------------------------------
        st.markdown("### 🔍 Section 2: Three-Part Grounded Breakdown")

        # Tab layout or stacked cards for the three parts
        tab_a, tab_b, tab_c = st.tabs([
            "A. Explicitly Stated in Text (With Substring Verification)",
            "B. Interpretations & Concerns",
            "C. Unknown / Document is Silent",
        ])

        # PART A: Stated Facts & Quote Verification
        with tab_a:
            v_report = st.session_state.verification_report
            if v_report:
                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("Total Quotes", v_report.total_quotes)
                mcol2.metric("Exact Substring Matches", v_report.exact_matches)
                mcol3.metric("Deterministic Grounding Score", f"{v_report.grounding_score_pct}%")

            facts = st.session_state.verified_facts or data.get("stated_facts", [])
            if facts:
                for f in facts:
                    pid = f.get("passage_id", "P?")
                    quote = f.get("verbatim_quote", "")
                    meaning = f.get("plain_meaning", "")
                    v_label = f.get("verification_label", "")
                    is_exact = f.get("verification_status") == EXACT_SUBSTRING_MATCH

                    card_style = "card-grounded" if is_exact else "card-paraphrase"

                    st.markdown(
                        f"""
                        <div class="{card_style}">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span class="passage-badge">[{pid}]</span>
                                <span style="font-size: 0.8rem; font-weight: 700;">{v_label}</span>
                            </div>
                            <div class="verbatim-text">“{quote}”</div>
                            <div style="color: #f1f5f9; font-size: 0.9rem; margin-top: 4px;">
                                <strong>Plain-English Meaning:</strong> {meaning}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No stated facts extracted.")

        # PART B: Interpretations, Ambiguities & Conflicts
        with tab_b:
            concerns = data.get("interpretations_and_concerns", [])
            if concerns:
                for c in concerns:
                    pid = c.get("passage_id", "Notice")
                    cat = c.get("category", "Analysis")
                    exp = c.get("explanation", "")

                    st.markdown(
                        f"""
                        <div class="card-concern">
                            <span class="passage-badge">[{pid}]</span>
                            <span style="background: #4c1d95; color: #ddd6fe; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700;">
                                {cat}
                            </span>
                            <div style="color: #e2e8f0; font-size: 0.9rem; margin-top: 6px;">
                                {exp}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No potential ambiguities or intimidations identified.")

        # PART C: Unknown / Document is Silent
        with tab_c:
            silents = data.get("unknown_or_silent", [])
            if silents:
                for item in silents:
                    st.markdown(
                        f"""
                        <div class="card-silent">
                            <span style="color: #94a3b8; font-weight: 700;">⚪ SILENT / UNANSWERED:</span>
                            <span style="color: #f1f5f9; margin-left: 6px;">{item}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No silent or missing elements noted.")

        # -------------------------------------------------------------------
        # Section 3: Refusal Guardrail & Lawyer Escalation Checklist
        # -------------------------------------------------------------------
        st.markdown("### ⚖️ Section 3: Questions for a Qualified Legal Professional")
        checklist = data.get("lawyer_escalation_checklist", [])
        if checklist:
            st.info(
                "NoticeGuard cannot give legal advice. Take these specific, targeted questions "
                "to an attorney licensed in your jurisdiction:"
            )
            for idx, q in enumerate(checklist, 1):
                st.markdown(f"**{idx}.** {q}")
        else:
            st.caption("No specific checklist items provided.")

    elif res and not res.success:
        st.error(f"❌ Analysis failed: {res.error_message}")
    else:
        st.info("👆 Click **'1. Run Full Triage'** above or choose a Quick-Test to begin.")

    # -----------------------------------------------------------------------
    # Privacy & Loopback Proof Section
    # -----------------------------------------------------------------------
    with st.expander("🔒 Live Loopback Network Isolation Proof (Audit)", expanded=False):
        st.markdown(
            "NoticeGuard enforces strict local loopback operation (`127.0.0.1` / `::1`). "
            "Click below to inspect active process sockets in real-time."
        )
        if st.button("🔍 Run Live Socket Audit"):
            st.session_state.privacy_audit = run_loopback_audit()

        audit = st.session_state.privacy_audit or run_loopback_audit()
        if audit:
            acol1, acol2, acol3 = st.columns(3)
            acol1.metric("Process ID", audit.pid)
            acol2.metric("Active Sockets", audit.total_sockets)
            acol3.metric("Loopback Status", "✅ Air-Gapped Loopback" if audit.is_strictly_loopback else "❌ NON-LOOPBACK")

            st.markdown(f"**Audit Verdict:** `{audit.audit_verdict}`")
            st.markdown(f"**NO_PROXY Environment Verified:** `{'YES' if audit.proxy_env_verified else 'NO'}`")

            if audit.records:
                socket_data = [
                    {
                        "Protocol": f"{r.family} {r.type}",
                        "Local Address": r.local_address,
                        "Remote Address": r.remote_address,
                        "Status": r.status,
                        "Loopback Compliant": "✅ Loopback" if r.is_loopback else "❌ Non-Loopback",
                    }
                    for r in audit.records
                ]
                st.dataframe(socket_data, use_container_width=True)
            else:
                st.info("Zero active network connections open at this instant.")

            st.download_button(
                label="📥 Download privacy_loopback_audit.md",
                data=audit.audit_report_md,
                file_name="privacy_loopback_audit.md",
                mime="text/markdown",
            )
