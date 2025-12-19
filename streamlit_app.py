import io
import re
from typing import List, Set, Tuple
import fitz  # PyMuPDF

import streamlit as st

# PDF text extraction
import pdfplumber

# NER
import spacy

st.set_page_config(page_title="Transcript Scrubber", layout="wide")
st.title("Transcript PDF Scrubber")

# -------- helpers --------
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

def extract_pdf_text(file_bytes: bytes) -> Tuple[str, List[str]]:
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    full_text = "\n\n".join(pages).strip()
    return full_text, pages

def normalize_list(raw: str) -> List[str]:
    parts = re.split(r"[,\n]+", raw or "")
    cleaned = [p.strip() for p in parts if p.strip()]
    cleaned.sort(key=len, reverse=True)  # longer first helps avoid partial masking
    return cleaned

def build_phrase_regex(phrases: List[str]) -> re.Pattern:
    if not phrases:
        return re.compile(r"(?!x)x")  # match nothing
    escaped = [re.escape(p) for p in phrases]
    # Case-insensitive, avoid matching inside words
    return re.compile(r"(?i)(?<!\w)(" + "|".join(escaped) + r")(?!\w)")

def detect_entities(text: str, nlp, labels=("PERSON", "ORG")) -> Set[str]:
    ents: Set[str] = set()
    # If extremely long transcripts, you might chunk; keep simple for now
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in labels:
            v = ent.text.strip()
            if v:
                ents.add(v)
    return ents

def redact_text(text: str, phrases: List[str], replacement: str) -> str:
    pat = build_phrase_regex(phrases)
    return pat.sub(replacement, text)

def redact_pdf_bytes(
    pdf_bytes: bytes,
    phrases: List[str],
    replacement_text: str = "[REDACTED]",
) -> bytes:
    """
    True redaction: preserves original PDF layout, removes the text visually.
    Note: This is based on text search; it requires selectable text (not scanned images).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # sort longest-first so full names get redacted before first names
    phrases = [p for p in phrases if p and p.strip()]
    phrases.sort(key=len, reverse=True)

    # Optional: de-hyphenate search to catch line breaks like "Site-\nMinder"
    search_flags = 0
    if hasattr(fitz, "TEXT_DEHYPHENATE"):
        search_flags |= fitz.TEXT_DEHYPHENATE

    total_hits = 0

    for page in doc:
        for phrase in phrases:
            # Case-insensitive search if available in your PyMuPDF version
            flags = search_flags
            if hasattr(fitz, "TEXT_IGNORECASE"):
                flags |= fitz.TEXT_IGNORECASE

            rects = page.search_for(phrase, flags=flags)

            # If TEXT_IGNORECASE isn't available, you can optionally try a couple variants:
            if not rects and not hasattr(fitz, "TEXT_IGNORECASE"):
                rects = page.search_for(phrase.lower(), flags=search_flags) + page.search_for(
                    phrase.upper(), flags=search_flags
                )

            for r in rects:
                total_hits += 1
                # Add a redaction annotation (filled box) and optional overlay text
                page.add_redact_annot(r, text=replacement_text, fill=(0, 0, 0))

        # Apply all redactions on the page
        page.apply_redactions()

    out = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    return out

# -------- UI --------
uploaded = st.file_uploader("Upload or drag-and-drop a transcript PDF", type=["pdf"])

with st.sidebar:
    st.header("Scrub Settings")

    mode = st.radio(
        "Mode",
        [
            "Scrub ONLY the names/companies I enter",
            "Auto-detect and scrub ALL names/companies",
        ],
        index=0,
    )

    entity_types = st.multiselect(
        "Auto-detect entity types (auto mode only)",
        ["PERSON", "ORG", "GPE", "LOC"],
        default=["PERSON", "ORG"],
        help="PERSON=people, ORG=companies, GPE=places (cities/countries), LOC=locations",
    )

    replacement = st.text_input("Replacement token", value="[REDACTED]")

    user_terms_raw = st.text_area(
        "Names / companies to scrub (comma or newline separated)",
        placeholder="e.g.\nAlan Ngouenet\nBCG\nAcme Corp",
        height=160,
    )

if not uploaded:
    st.info("Upload a PDF to begin.")
    st.stop()

file_bytes = uploaded.read()

with st.spinner("Extracting text from PDF..."):
    full_text, pages = extract_pdf_text(file_bytes)

if not full_text:
    st.warning(
        "No text could be extracted. If this PDF is scanned (image-only), we’ll need OCR."
    )
    st.stop()

user_terms = normalize_list(user_terms_raw)
targets = list(user_terms)

if mode.startswith("Auto-detect"):
    nlp = load_spacy_model()
    with st.spinner("Detecting names/companies with NER..."):
        detected = detect_entities(full_text, nlp, labels=tuple(entity_types))
    # Sort longest-first so full names get replaced before first names
    targets = sorted(detected, key=len, reverse=True)

with st.spinner("Scrubbing..."):
    redacted_text = redact_text(full_text, targets, replacement.strip() or "[REDACTED]")

with st.spinner("Generating redacted PDF (preserving layout)..."):
    redacted_pdf = redact_pdf_bytes(
        pdf_bytes=file_bytes,
        phrases=targets,
        replacement_text=replacement.strip() or "[REDACTED]",
    )

st.download_button(
    "Download redacted PDF (original layout)",
    data=redacted_pdf,
    file_name="redacted_transcript.pdf",
    mime="application/pdf",
)

# -------- display --------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original (preview)")
    st.text_area("Original", full_text[:20000], height=420)

with col2:
    st.subheader("Redacted (preview)")
    st.text_area("Redacted", redacted_text[:20000], height=420)

with st.expander("What is being scrubbed?"):
    if mode.startswith("Auto-detect"):
        st.write(f"Detected entities ({', '.join(entity_types)}):")
        st.write(targets[:500])  # cap display
    else:
        st.write("Using your provided list:")
        st.write(user_terms if user_terms else "(empty list)")

st.download_button(
    "Download redacted text (.txt)",
    data=redacted_text.encode("utf-8"),
    file_name="redacted_transcript.txt",
    mime="text/plain",
)