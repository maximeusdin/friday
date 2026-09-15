"""One-page smoke test: render deposition page 3, transcribe with OpenAI vision."""
import base64, io, os, sys

import fitz

PDF = "data/raw/silvermaster/pdf/FBI File Silvermaster Part 6 late November 1945_text.pdf"
OUT_PNG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/page3.png"
PAGE = int(sys.argv[2]) if len(sys.argv) > 2 else 3  # 1-based pdf page
MODEL = sys.argv[3] if len(sys.argv) > 3 else None

# Load .env without exporting to shell
env = {}
with open(".env") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v.strip().strip('"').strip("'")

doc = fitz.open(PDF)
page = doc[PAGE - 1]
zoom = 1700 / page.rect.width
pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
pix.save(OUT_PNG)
print(f"rendered page {PAGE}: {pix.width}x{pix.height} -> {OUT_PNG}")

from openai import OpenAI

client = OpenAI(api_key=env["OPENAI_API_KEY"])

if MODEL is None:
    names = [m.id for m in client.models.list()]
    cand = [n for n in names if n.startswith(("gpt-5", "gpt-4.1", "gpt-4o"))]
    print("available vision-family models:", sorted(cand))
    sys.exit(0)

PROMPT = """You are transcribing a 1945 typewritten FBI document from a poor-quality scanned photostat.

Rules:
- Transcribe EXACTLY what is typed on the page: preserve original spelling, capitalization, punctuation, and any typist errors. This is a historical source; fidelity to the page outranks readability.
- Do NOT normalize or "correct" personal names. Render names exactly as typed.
- If a character or word is genuinely unreadable, write [illegible].
- If you can read something but are uncertain, wrap your best reading like [?word].
- Preserve paragraph structure. Join words hyphenated across line breaks.
- Transcribe page headers and file numbers (e.g. "NY 65-14603") as they appear.
- Handwritten serial stamps or numbers: render on their own line as [stamp: ...]. Ignore other stray pen marks.
- Output ONLY the transcription. No commentary, no markdown."""

b64 = base64.b64encode(open(OUT_PNG, "rb").read()).decode()
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
        ],
    }],
)
print(f"--- {MODEL} transcription ---")
print(resp.choices[0].message.content)
u = resp.usage
print(f"--- tokens: in={u.prompt_tokens} out={u.completion_tokens}")
