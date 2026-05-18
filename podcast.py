"""PDF -> podcast audio.

Two modes:
  * If the PDF already contains a speaker-tagged script (lines like "JAMIE:" /
    "ALEX:"), parse it directly and skip OpenRouter.
  * Otherwise, send the extracted text to OpenRouter to draft a two-host
    dialogue, then synthesize.

Audio backends: edge-tts (free, no key, default) or OpenAI TTS (--tts openai).
One segment per turn, concatenated to MP3.
"""
import argparse
import asyncio
import json
import os
import pathlib
import re
import ssl
import sys

import edge_tts
import edge_tts.communicate
import edge_tts.voices
import requests
from pydub import AudioSegment
from pypdf import PdfReader

# edge_tts hard-codes its SSL context to certifi's bundle, which excludes some
# corporate / egress-proxy CAs. If a system bundle is available (via
# SSL_CERT_FILE), rebuild the context against it so it trusts the local proxy.
_system_ca = os.environ.get("SSL_CERT_FILE") if "SSL_CERT_FILE" in os.environ else None
if _system_ca and os.path.isfile(_system_ca):
    _ctx = ssl.create_default_context(cafile=_system_ca)
    edge_tts.communicate._SSL_CTX = _ctx
    edge_tts.voices._SSL_CTX = _ctx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"

OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "tts-1")
PAUSE_MS = int(os.environ.get("PAUSE_MS", "350"))

# Per-backend default voice pairs (A = JAMIE/host_a, B = ALEX/host_b).
VOICE_DEFAULTS = {
    "edge": {
        "A": os.environ.get("EDGE_VOICE_A", "en-GB-SoniaNeural"),
        "B": os.environ.get("EDGE_VOICE_B", "en-AU-NatashaNeural"),
    },
    "openai": {
        "A": os.environ.get("VOICE_A", "alloy"),
        "B": os.environ.get("VOICE_B", "nova"),
    },
}

# Speaker tags treated as speaker A vs speaker B. Override via env if your
# script uses different names (comma-separated, case-insensitive).
SPEAKERS_A = [s.strip().upper() for s in os.environ.get("SPEAKERS_A", "JAMIE,HOST_A,A").split(",")]
SPEAKERS_B = [s.strip().upper() for s in os.environ.get("SPEAKERS_B", "ALEX,HOST_B,B").split(",")]

# Lines that look like section headers / production cues and should be dropped.
HEADER_RE = re.compile(
    r"^(COLD OPEN|SEGMENT\s+\d+.*|OUTRO|INTRO|PRODUCTION NOTES|NOTES FOR PRODUCTION.*|For TTS:.*)$",
    re.IGNORECASE,
)
# Marker that ends the spoken portion of the script.
END_MARKERS = ("[END]", "NOTES FOR PRODUCTION")


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(f"Missing required env var: {name}")
    return val


def extract_pdf_text(path: pathlib.Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def parse_scripted_pdf(text: str) -> list[dict] | None:
    """Return [{speaker, text}, ...] if the PDF already has speaker tags, else None."""
    speaker_re = re.compile(
        r"^(" + "|".join(re.escape(s) for s in SPEAKERS_A + SPEAKERS_B) + r"):\s*(.*)$",
        re.IGNORECASE,
    )
    speakers_a = set(SPEAKERS_A)

    turns: list[dict] = []
    current: dict | None = None

    def flush() -> None:
        nonlocal current
        if current:
            current["text"] = re.sub(r"\s+", " ", current["text"]).strip()
            # Truncate at end markers if present
            for marker in END_MARKERS:
                idx = current["text"].find(marker)
                if idx != -1:
                    current["text"] = current["text"][:idx].strip()
            if current["text"]:
                turns.append(current)
        current = None

    stopped = False
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln or stopped:
            if stopped:
                break
            continue
        # If a line contains an end marker outside of any speaker turn, stop.
        if any(m in ln for m in END_MARKERS) and current is None:
            stopped = True
            continue
        m = speaker_re.match(ln)
        if m:
            flush()
            tag = m.group(1).upper()
            current = {
                "speaker": "A" if tag in speakers_a else "B",
                "text": m.group(2),
            }
            continue
        if HEADER_RE.match(ln):
            flush()
            continue
        if current is not None:
            current["text"] += " " + ln
    flush()

    return turns if len(turns) >= 4 else None


SCRIPT_PROMPT = """You are a podcast scriptwriter. Turn the document below into a natural two-host dialogue.

HOST_A is curious, asks questions, plays the audience.
HOST_B is the expert who explains.

Rules:
- Output ONLY valid JSON: {"dialogue": [{"speaker": "A"|"B", "text": "..."}]}
- 16-28 turns total. Each line 1-3 sentences, conversational.
- Open with a brief intro by A, close with a brief sign-off.
- Do not invent facts not in the source.

SOURCE DOCUMENT:
---
{text}
---
"""


def generate_script(text: str, api_key: str) -> list[dict]:
    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": SCRIPT_PROMPT.format(text=text)}],
            "response_format": {"type": "json_object"},
        },
        timeout=180,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    data = json.loads(content)
    if isinstance(data, dict) and "dialogue" in data:
        return data["dialogue"]
    if isinstance(data, list):
        return data
    for v in (data.values() if isinstance(data, dict) else []):
        if isinstance(v, list):
            return v
    sys.exit(f"Unexpected script JSON shape: {content[:200]}")


def synthesize_openai(text: str, voice: str, api_key: str, out_path: pathlib.Path) -> None:
    resp = requests.post(
        OPENAI_TTS_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": TTS_MODEL,
            "voice": voice,
            "input": text,
            "response_format": "mp3",
        },
        timeout=180,
    )
    resp.raise_for_status()
    out_path.write_bytes(resp.content)


async def _edge_save(text: str, voice: str, out_path: pathlib.Path) -> None:
    await edge_tts.Communicate(text, voice).save(str(out_path))


def synthesize_edge(text: str, voice: str, out_path: pathlib.Path) -> None:
    asyncio.run(_edge_save(text, voice, out_path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", help="Path to source PDF")
    ap.add_argument("-o", "--output", default="podcast.mp3", help="Output MP3 path")
    ap.add_argument("--script-only", action="store_true", help="Generate JSON script, skip TTS")
    ap.add_argument("--script", help="Use existing JSON script instead of the PDF")
    ap.add_argument("--limit", type=int, help="Only synthesize first N turns (for previewing)")
    ap.add_argument(
        "--force-llm",
        action="store_true",
        help="Ignore speaker tags in PDF and route through OpenRouter to draft a new script",
    )
    ap.add_argument(
        "--tts",
        choices=("edge", "openai"),
        default=os.environ.get("TTS_BACKEND", "edge"),
        help="TTS backend: 'edge' (free, default) or 'openai' (needs OPENAI_API_KEY)",
    )
    args = ap.parse_args()
    voices = VOICE_DEFAULTS[args.tts]

    pdf_path = pathlib.Path(args.pdf)
    out_path = pathlib.Path(args.output)
    script_path = out_path.with_suffix(".json")

    if args.script:
        loaded = json.loads(pathlib.Path(args.script).read_text())
        dialogue = loaded["dialogue"] if isinstance(loaded, dict) and "dialogue" in loaded else loaded
    else:
        if not pdf_path.is_file():
            sys.exit(f"PDF not found: {pdf_path}")
        print(f"Extracting text from {pdf_path}...")
        text = extract_pdf_text(pdf_path)
        if not text:
            sys.exit("No text extracted from PDF (scanned/image-only PDFs need OCR).")
        print(f"  {len(text):,} chars extracted")

        dialogue = None if args.force_llm else parse_scripted_pdf(text)
        if dialogue:
            print(f"  detected scripted PDF: {len(dialogue)} turns parsed directly (no LLM call)")
        else:
            openrouter_key = require_env("OPENROUTER_API_KEY")
            print(f"Generating script via OpenRouter ({OPENROUTER_MODEL})...")
            dialogue = generate_script(text, openrouter_key)
            print(f"  {len(dialogue)} turns generated")

        script_path.write_text(json.dumps({"dialogue": dialogue}, indent=2))
        print(f"  wrote {script_path}")

    if args.limit:
        dialogue = dialogue[: args.limit]
        print(f"  --limit {args.limit}: rendering first {len(dialogue)} turns only")

    if args.script_only:
        return

    openai_key = require_env("OPENAI_API_KEY") if args.tts == "openai" else None
    work = out_path.parent / f"{out_path.stem}_segments"
    work.mkdir(parents=True, exist_ok=True)

    total_chars = sum(len(t["text"]) for t in dialogue)
    backend_label = (
        f"OpenAI TTS ({TTS_MODEL})" if args.tts == "openai" else f"edge-tts ({voices['A']} / {voices['B']})"
    )
    print(f"Synthesizing {len(dialogue)} turns ({total_chars:,} chars) with {backend_label}...")
    segments: list[AudioSegment] = []
    for i, turn in enumerate(dialogue):
        speaker = str(turn["speaker"]).upper()
        voice = voices.get(speaker, voices["A"])
        seg_path = work / f"{i:03d}_{speaker}.mp3"
        preview = turn["text"][:70].replace("\n", " ")
        print(f"  [{i + 1:>3}/{len(dialogue)}] {speaker} ({voice}): {preview}...")
        if not seg_path.exists():  # cheap resume on re-runs
            if args.tts == "openai":
                synthesize_openai(turn["text"], voice, openai_key, seg_path)
            else:
                synthesize_edge(turn["text"], voice, seg_path)
        segments.append(AudioSegment.from_mp3(seg_path))

    pause = AudioSegment.silent(duration=PAUSE_MS)
    combined = AudioSegment.empty()
    for seg in segments:
        combined += seg + pause
    combined.export(out_path, format="mp3")
    print(f"Wrote podcast: {out_path} ({combined.duration_seconds:.1f}s)")


if __name__ == "__main__":
    main()
