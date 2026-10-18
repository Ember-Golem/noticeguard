"""In-memory ingestion module for NoticeGuard.
Ingests raw text, .txt, and .pdf files completely in memory using io.BytesIO.
Zero temporary files are ever created on disk.
"""
from dataclasses import dataclass, field
import io
import re
from typing import Dict, List, Optional
from config import MAX_DOCUMENT_CHARS, MIN_DOCUMENT_CHARS


@dataclass
class IngestResult:
    """Structured result returned by in-memory document ingestion."""
    success: bool
    raw_text: str = ""
    passages: Dict[str, str] = field(default_factory=dict)
    anchors: Dict[str, List[str]] = field(default_factory=dict)
    error_message: Optional[str] = None
    char_count: int = 0
    passage_count: int = 0


# Compiled regex patterns for deterministic anchor extraction
REGEX_DATES = re.compile(
    r"\b(?:"
    r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"(?:\d{1,2}(?:st|nd|rd|th)?[\s,-]+)?"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"[\s,-]+\d{1,2}(?:st|nd|rd|th)?,?[\s,-]+\d{2,4}|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"[\s,-]+\d{1,2}(?:st|nd|rd|th)?,?[\s,-]+\d{2,4}"
    r")\b",
    re.IGNORECASE,
)

REGEX_CURRENCY = re.compile(
    r"(?:Rs\.?|₹|INR|\$)\s*[\d,]+(?:\.\d{2})?(?:\s*(?:lakhs?|crores?|thousand|million|k))?\b",
    re.IGNORECASE,
)

REGEX_TIMEFRAMES = re.compile(
    r"\b\d+\s*(?:business\s+days?|calendar\s+days?|working\s+days?|days?|hours?|hrs?|weeks?|months?)\b",
    re.IGNORECASE,
)


def extract_anchors(text: str) -> Dict[str, List[str]]:
    """Deterministically extracts dates, currency figures, and notice timeframes from text."""
    if not text:
        return {"dates": [], "currencies": [], "timeframes": []}

    dates = list(dict.fromkeys([m.group(0).strip() for m in REGEX_DATES.finditer(text)]))
    currencies = list(dict.fromkeys([m.group(0).strip() for m in REGEX_CURRENCY.finditer(text)]))
    timeframes = list(dict.fromkeys([m.group(0).strip() for m in REGEX_TIMEFRAMES.finditer(text)]))

    return {
        "dates": dates,
        "currencies": currencies,
        "timeframes": timeframes,
    }


def index_passages(text: str) -> Dict[str, str]:
    """Deterministically splits text by paragraphs into an ordered dictionary {P1: ..., P2: ...}.
    Intelligently identifies explicit P-labels (e.g. 'P1:', '[P1]') while attaching headers.
    """
    if not text or not text.strip():
        return {}

    # Primary split by double line breaks (paragraphs)
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]

    # If the document only used single line breaks, try splitting by numbered clauses or linebreaks
    if len(raw_paras) <= 1 and "\n" in text:
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        if len(lines) > 1:
            raw_paras = lines

    # Check if explicit P-labels exist (e.g. P1:, P2: or [P1])
    explicit_pattern = re.compile(r"^(?:\[?P(\d+)\]?[:\.\s])\s*(.*)", re.DOTALL | re.IGNORECASE)
    has_explicit = any(explicit_pattern.match(p) for p in raw_paras)

    passages: Dict[str, str] = {}
    if has_explicit:
        header_prefix = ""
        for p in raw_paras:
            match = explicit_pattern.match(p)
            if match:
                pid = f"P{match.group(1)}"
                content = match.group(2).strip()
                if header_prefix and pid == "P1":
                    content = f"{header_prefix}\n{content}"
                    header_prefix = ""
                passages[pid] = re.sub(r"[ \t]+", " ", content)
            else:
                # If this is a leading header before P1
                if not passages:
                    header_prefix = p if not header_prefix else f"{header_prefix}\n{p}"
                else:
                    # Trailing unlabelled paragraph: assign next available P index
                    next_idx = len(passages) + 1
                    passages[f"P{next_idx}"] = re.sub(r"[ \t]+", " ", p)
    else:
        idx = 1
        for p in raw_paras:
            normalized = re.sub(r"[ \t]+", " ", p)
            if normalized:
                passages[f"P{idx}"] = normalized
                idx += 1

    return passages


def ingest_text(raw_text: str) -> IngestResult:
    """Ingests raw pasted text in-memory with strict validation."""
    if not raw_text or not raw_text.strip():
        return IngestResult(
            success=False,
            error_message="The provided text is empty. Please enter or upload a legal notice."
        )

    clean_text = raw_text.strip()
    char_count = len(clean_text)

    if char_count < MIN_DOCUMENT_CHARS:
        return IngestResult(
            success=False,
            raw_text=clean_text,
            char_count=char_count,
            error_message=f"Text is too short for notice triage ({char_count} chars). Minimum {MIN_DOCUMENT_CHARS} chars required."
        )

    if char_count > MAX_DOCUMENT_CHARS:
        return IngestResult(
            success=False,
            raw_text=clean_text[:500] + "...",
            char_count=char_count,
            error_message=(
                f"Document exceeds short-form notice limit ({char_count:,} characters found; max {MAX_DOCUMENT_CHARS:,}). "
                "NoticeGuard is scoped for short-form notices and termination letters (1-2 pages)."
            )
        )

    passages = index_passages(clean_text)
    anchors = extract_anchors(clean_text)

    return IngestResult(
        success=True,
        raw_text=clean_text,
        passages=passages,
        anchors=anchors,
        char_count=char_count,
        passage_count=len(passages),
    )


def ingest_file(file_bytes: bytes, filename: str) -> IngestResult:
    """Ingests a file (.txt or .pdf) purely in-memory via io.BytesIO without writing to disk."""
    if not file_bytes:
        return IngestResult(
            success=False,
            error_message=f"Uploaded file '{filename}' is empty (0 bytes)."
        )

    lower_name = filename.lower()

    if lower_name.endswith(".txt"):
        try:
            # Attempt UTF-8 with fallback to Latin-1
            try:
                text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = file_bytes.decode("latin-1")

            # Check for non-printable binary garbage
            if "\x00" in text:
                return IngestResult(
                    success=False,
                    error_message=f"File '{filename}' contains binary data or invalid characters."
                )

            return ingest_text(text)
        except Exception as e:
            return IngestResult(
                success=False,
                error_message=f"Failed to read text file '{filename}': {str(e)}"
            )

    elif lower_name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            return IngestResult(
                success=False,
                error_message="pypdf package is not installed. PDF ingestion unavailable."
            )

        try:
            # In-memory stream strictly in memory
            stream = io.BytesIO(file_bytes)
            reader = PdfReader(stream)

            if len(reader.pages) == 0:
                return IngestResult(
                    success=False,
                    error_message=f"PDF file '{filename}' has no readable pages."
                )

            extracted_pages: List[str] = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    extracted_pages.append(page_text.strip())

            full_text = "\n\n".join(extracted_pages)
            if not full_text.strip():
                return IngestResult(
                    success=False,
                    error_message=f"No selectable text could be extracted from '{filename}'. The PDF may be scanned or image-only."
                )

            return ingest_text(full_text)

        except Exception as e:
            return IngestResult(
                success=False,
                error_message=f"Failed to process PDF file '{filename}': {str(e)}"
            )

    else:
        return IngestResult(
            success=False,
            error_message=f"Unsupported file format '{filename}'. NoticeGuard accepts .txt and .pdf files only."
        )
