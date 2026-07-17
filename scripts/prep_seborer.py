#!/usr/bin/env python3
"""Prep the oscar_seborer dir for ingest: dedup byte-identical files, convert
.doc/.docx -> text PDFs and .jpg/.png -> 1-page image PDFs, so the whole dir is
PDFs that ingest_dir_collection --ocr-auto can route per-file."""
import os, glob, shutil, zipfile, re, subprocess, fitz

D = r"C:\Users\maxim\friday\data\raw\oscar_seborer"
EXC = os.path.join(D, "_excluded")
os.makedirs(EXC, exist_ok=True)

# byte-dup drops (keep the better-named twin) + the docx redundant with its PDF twin
DROP = [
    "Doc 1.PDF", "Doc 2.PDF", "Doc 3.PDF", "Doc 4.PDF", "Doc 5.PDF",
    "Pages from 1222443-0 - Section 3_Fussell_1.pdf",
    "Pages from 1222443-0 - Section 3_Fussell_2.pdf",
    "Pages from 1222443-0 - Section 3_Fussell_3.pdf",
    "Pages from 1222443-0 - Section 3_Kramish.pdf",
    "FBI seborer 1961 Mariam.docx",
    "atomic Oscar Seborer LANL 1956 (LA-UR-19-32293).pdf",
    "Seborer Carr Oscar Seborer (LA-UR-20-22823).doc",
    "Eb569f7e4724a57eaf1aaaf7dbcca9225c9dac7f9_Q59574_R413889_D2331569.docx",
]
for n in DROP:
    p = os.path.join(D, n)
    if os.path.exists(p):
        shutil.move(p, os.path.join(EXC, n)); print("DROP  ", n)
    else:
        print("  (missing, skip)", n)

def docx_text(path):
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    out = []
    for para in re.split(r"</w:p>", xml):
        runs = re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, re.DOTALL)
        line = "".join(runs)
        for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&apos;", "'")):
            line = line.replace(a, b)
        out.append(line)
    return "\n".join(out)

def doc_text(path):
    r = subprocess.run(["antiword", path], capture_output=True, text=True)
    return r.stdout or ""

def text_to_pdf(text, outpath):
    lines = text.replace("\r", "").split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        lines = ["(empty)"]
    LPP = 56
    doc = fitz.open()
    for i in range(0, len(lines), LPP):
        page = doc.new_page(width=612, height=792)
        y = 60
        for ln in lines[i:i+LPP]:
            page.insert_text((54, y), ln[:110], fontsize=9, fontname="cour"); y += 12
    doc.save(outpath); doc.close()

def image_to_pdf(imgpath, outpath):
    img = fitz.open(imgpath)
    pdfbytes = img.convert_to_pdf(); img.close()
    out = fitz.open("pdf", pdfbytes); out.save(outpath); out.close()

DOCS = [
    "FBI seborer 1961.docx", "Lifeat Noah Seborer Mexico.docx", "Nosenko.docx",
    "Oscar Seborer (LA-UR-20-22823).doc",
    "Seborer Noah HK Summary of Noah Seborer FBI File.docx",
    "Seborers Israel.docx", "atomic Trinity Bainbridge Seborer p 20.docx",
    "project hunter.docx", "seborer CIA.docx",
]
print("\n-- converting docs --")
for n in DOCS:
    src = os.path.join(D, n)
    if not os.path.exists(src):
        print("  (missing)", n); continue
    base = os.path.splitext(n)[0]
    txt = doc_text(src) if n.lower().endswith(".doc") else docx_text(src)
    text_to_pdf(txt, os.path.join(D, base + ".pdf"))
    shutil.move(src, os.path.join(EXC, n))
    print(f"  DOC  {n[:52]:52} -> {base}.pdf  ({len(txt)} chars)")

IMAGES = [f for f in os.listdir(D)
          if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png") and os.path.isfile(os.path.join(D, f))]
print("\n-- converting images --")
for n in sorted(IMAGES):
    src = os.path.join(D, n); base = os.path.splitext(n)[0]
    try:
        image_to_pdf(src, os.path.join(D, base + ".pdf"))
        shutil.move(src, os.path.join(EXC, n))
        print(f"  IMG  {n}")
    except Exception as e:
        print(f"  IMG-ERR {n}: {e}")

pdfs = set(os.path.basename(p) for p in glob.glob(os.path.join(D, "*.pdf")))
print(f"\nfinal: {len(pdfs)} PDFs in dir; {len(os.listdir(EXC))} files moved to _excluded")
