"""Run Surya OCR over the Bentley page images, emitting per-page text files.

Output: <out>/{page:04d}.txt — text lines joined in reading order, the same
shape as the textract/*.txt contract, so a config dir can mount it in the
OCR-vote slot.
"""
import argparse
import os
import sys

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--images", default="data/transcripts/bentley_deposition/images")
ap.add_argument("--out", default="data/transcripts/surya_bentley")
ap.add_argument("--pages", default="2-120")
args = ap.parse_args()

a, b = args.pages.split("-")
pages = [p for p in range(int(a), int(b) + 1)]
os.makedirs(args.out, exist_ok=True)
pending = [p for p in pages if not os.path.exists(os.path.join(args.out, f"{p:04d}.txt"))]
print(f"{len(pending)} pages pending", flush=True)
if not pending:
    sys.exit(0)

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

foundation = FoundationPredictor()
rec = RecognitionPredictor(foundation)
det = DetectionPredictor()

BATCH = 4
for i in range(0, len(pending), BATCH):
    chunk = pending[i:i + BATCH]
    images = [Image.open(os.path.join(args.images, f"{p:04d}.png")) for p in chunk]
    preds = rec(images, det_predictor=det)
    for p, pred in zip(chunk, preds):
        lines = [l.text for l in pred.text_lines]
        tmp = os.path.join(args.out, f"{p:04d}.txt.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        os.replace(tmp, os.path.join(args.out, f"{p:04d}.txt"))
        print(f"page {p}: {len(lines)} lines", flush=True)
print("done")
