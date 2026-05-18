"""PDF -> OpenRouter (script) -> OpenAI TTS (audio) podcast generator."""
import argparse
import json
import os
import pathlib
import sys

import requests
from pydub import AudioSegment
from pypdf import PdfReader

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"

OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "tts-1")
VOICE_A = os.environ.get("VOICE_A", "alloy")
VOICE_B = os.environ.get("VOICE_B", "nova")
PAUSE_MS = int(os.environ.get("PAUSE_MS", "350"))


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(f"Missing required env var: {name}")
    return val


def extract_pdf_text(path: pathlib.Path) -> str:
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()


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
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
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
    for v in data.values() if isinstance(data, dict) else []:
        if isinstance(v, list):
            return v
    sys.exit(f"Unexpected script JSON shape: {content[:200]}")


def synthesize(text: str, voice: str, api_key: str, out_path: pathlib.Path) -> None:
    resp = requests.post(
        OPENAI_TTS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", help="Path to source PDF")
    ap.add_argument("-o", "--output", default="podcast.mp3", help="Output MP3 path")
    ap.add_argument("--script-only", action="store_true", help="Generate JSON script, skip TTS")
    ap.add_argument("--script", help="Use existing JSON script instead of regenerating from PDF")
    args = ap.parse_args()

    pdf_path = pathlib.Path(args.pdf)
    out_path = pathlib.Path(args.output)
    script_path = out_path.with_suffix(".json")

    if args.script:
        dialogue = json.loads(pathlib.Path(args.script).read_text())
        if isinstance(dialogue, dict) and "dialogue" in dialogue:
            dialogue = dialogue["dialogue"]
    else:
        if not pdf_path.is_file():
            sys.exit(f"PDF not found: {pdf_path}")
        openrouter_key = require_env("OPENROUTER_API_KEY")

        print(f"Extracting text from {pdf_path}...")
        text = extract_pdf_text(pdf_path)
        if not text:
            sys.exit("No text extracted from PDF (scanned/image-only PDFs need OCR).")
        print(f"  {len(text):,} chars extracted")

        print(f"Generating script via OpenRouter ({OPENROUTER_MODEL})...")
        dialogue = generate_script(text, openrouter_key)
        script_path.write_text(json.dumps({"dialogue": dialogue}, indent=2))
        print(f"  wrote {script_path} ({len(dialogue)} turns)")

    if args.script_only:
        return

    openai_key = require_env("OPENAI_API_KEY")
    work = out_path.parent / f"{out_path.stem}_segments"
    work.mkdir(parents=True, exist_ok=True)

    print(f"Synthesizing {len(dialogue)} turns with OpenAI TTS ({TTS_MODEL})...")
    segments: list[AudioSegment] = []
    for i, turn in enumerate(dialogue):
        speaker = str(turn["speaker"]).upper()
        voice = VOICE_A if speaker == "A" else VOICE_B
        seg_path = work / f"{i:03d}_{speaker}.mp3"
        preview = turn["text"][:70].replace("\n", " ")
        print(f"  [{i + 1:>2}/{len(dialogue)}] {speaker} ({voice}): {preview}...")
        synthesize(turn["text"], voice, openai_key, seg_path)
        segments.append(AudioSegment.from_mp3(seg_path))

    pause = AudioSegment.silent(duration=PAUSE_MS)
    combined = AudioSegment.empty()
    for seg in segments:
        combined += seg + pause
    combined.export(out_path, format="mp3")
    print(f"Wrote podcast: {out_path} ({combined.duration_seconds:.1f}s)")


if __name__ == "__main__":
    main()
