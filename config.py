"""Configuration settings for NoticeGuard: Local Legal Notice & Termination Letter Triage Agent.
Enforces loopback-only network isolation, model fallback, and feature flags.
"""
import os

# ---------------------------------------------------------
# Network Isolation: Loopback only
# ---------------------------------------------------------
# Force NO_PROXY to restrict any accidental proxy leakage and keep all traffic local
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

# ---------------------------------------------------------
# Ollama and Agent Settings
# ---------------------------------------------------------
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma4:e4b"
OLLAMA_TIMEOUT_SECONDS = 1.5

# Developer Mock Mode: allows complete UI, guardrail, and verifier testing
# on development machines without requiring a running Ollama daemon
ALLOW_DEV_MOCK_MODE = True

# Target Scope & Input Limits
MAX_DOCUMENT_CHARS = 15000  # Strict limit for short-form notices (1-2 pages)
MIN_DOCUMENT_CHARS = 20

# ---------------------------------------------------------
# Feature Flags for Phased Rollout (Stretch targets)
# ---------------------------------------------------------
ENABLE_STRETCH_OCR = False
ENABLE_STRETCH_HINDI = False

# App metadata
APP_TITLE = "NoticeGuard: Local Legal Notice & Termination Letter Triage Agent"
APP_SUBTITLE = "Air-gapped, privacy-first legal triage powered by local Gemma & deterministic grounding"
APP_VERSION = "1.0.0"

# Calibration boundary text
PERMANENT_BOUNDARY_NOTICE = (
    "⚠️ **PRELIMINARY DOCUMENT TRIAGE ONLY — NOT LEGAL ADVICE**: "
    "NoticeGuard operates strictly offline to assist in parsing deadlines, explicit demands, and ambiguities. "
    "It is not a lawyer, cannot provide legal counsel, and cannot guarantee judicial outcomes. "
    "All factual quotes undergo deterministic substring verification against source passages."
)
