import os
import fitz
import pdfplumber
import easyocr
from PIL import Image, ImageEnhance
import io
import numpy as np

# ── Config ──────────────────────────────────────────────
DOCUMENTS_DIR = "./documents"
PROCESSED_DIR = "./processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── EasyOCR readers (initialized once, reused across pages) ──
# Lazy-loaded so we don't pay startup cost for text PDFs
_readers = {}

def get_reader(lang):
    """Get or create an EasyOCR reader for the given language combo."""
    if lang not in _readers:
        print(f"  🔄 Loading EasyOCR model for: {lang} (first time only)...")
        if lang == "ara+eng":
            _readers[lang] = easyocr.Reader(['ar', 'en'], gpu=False)
        elif lang == "ara":
            _readers[lang] = easyocr.Reader(['ar'], gpu=False)
        else:
            _readers[lang] = easyocr.Reader(['en'], gpu=False)
    return _readers[lang]

# ── Document language map ────────────────────────────────
DOCUMENT_LANGUAGES = {
    "EAS - PG Bylaws-2022 July - Stamped.pdf"                                         : "ara+eng",
    "MOT- PG Bylaws- Professional Master- March 2023 Stamped.pdf"                     : "eng",
    "2017-Nov-EMBA Bylaws Approved by SCU.pdf"                                        : "eng",
    "Approved UG Manual Oct2024-UC(27Oct24)_BOT(2Dec24).pdf"                          : "eng",
    "Internship or Service Learning Program - BBA.pdf"                                : "eng",
    "NU Extracurricular - SelfService-signed.pdf"                                     : "eng",
    "NU Graduate Studies Manual 2018.pdf"                                             : "eng",
    "Payment and Services Steps on Selfservice-signed.pdf"                            : "eng",
    "Students Registration Process Guide on PowerCampus SelfService (for students).pdf": "eng",
    "ITCS-PG Bylaws-MSc. Program 2021- Sreachable.pdf"                               : "ara+eng",
    "NU Computer Science Program - v17docx.pdf"                                       : "ara+eng",
    "الخطة الإستراتيجية.pdf"                                                         : "ara",
    "اللائحة الداخلية لوحدة ضمان الجودة للكلية.pdf"                                 : "ara",
}

# Only truly scanned (image-only) PDFs
SCANNED_DOCUMENTS = {
    "EAS - PG Bylaws-2022 July - Stamped.pdf",
    "MOT- PG Bylaws- Professional Master- March 2023 Stamped.pdf",
}

# ── Helper: Check if a page is scanned ──────────────────
def is_scanned_page(page):
    text = page.get_text().strip()
    char_count = len([c for c in text if not c.isspace()])
    return char_count < 30

# ── Helper: Preprocess image for OCR ────────────────────
def preprocess_image(img):
    img = img.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    sharpener = ImageEnhance.Sharpness(img)
    img = sharpener.enhance(2.0)
    return img

# ── Helper: OCR a single page with EasyOCR ──────────────
def ocr_page(page, lang="ara+eng"):
    # Render page to image at 300 DPI
    mat = fitz.Matrix(300/72, 300/72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    img = preprocess_image(img)

    # Convert PIL image to numpy array for EasyOCR
    img_array = np.array(img)

    reader = get_reader(lang)

    # detail=0 returns just text strings
    # paragraph=True groups nearby text into paragraphs
    results = reader.readtext(img_array, detail=1, paragraph=False)

    # EasyOCR returns list of (bbox, text, confidence)
    # Sort by vertical position (top to bottom) then horizontal (right to left for Arabic)
    results.sort(key=lambda r: (r[0][0][1], -r[0][0][0]))  # sort by y, then reverse x for RTL

    # Filter low-confidence results and join
    lines = []
    for (bbox, text, confidence) in results:
        if confidence > 0.3 and text.strip():
            lines.append(text.strip())

    return "\n".join(lines)

# ── Helper: Extract tables via pdfplumber ────────────────
def extract_tables(pdf_path, page_num):
    tables_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num < len(pdf.pages):
                page = pdf.pages[page_num]
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        for row in table:
                            clean_row = [str(cell).strip() for cell in row if cell]
                            if clean_row:
                                tables_text += " | ".join(clean_row) + "\n"
                        tables_text += "\n"
    except Exception as e:
        print(f"    ⚠️  Table extraction error: {e}")
    return tables_text.strip()

# ── Main: Process a single PDF ──────────────────────────
def process_pdf(pdf_path):
    filename   = os.path.basename(pdf_path)
    lang       = DOCUMENT_LANGUAGES.get(filename, "ara+eng")
    is_scanned = filename in SCANNED_DOCUMENTS

    print(f"\nProcessing : {filename}")
    print(f"Language   : {lang}")
    print(f"Type       : {'Fully scanned → EasyOCR' if is_scanned else 'Text/mixed → direct extraction'}")

    full_text = ""
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        print(f"  Page {page_num + 1}/{len(doc)}", end=" ", flush=True)

        if is_scanned or is_scanned_page(page):
            print(f"→ EasyOCR ({lang})")
            text = ocr_page(page, lang=lang)
        else:
            print("→ direct text extraction")
            text = page.get_text()

            # Append any tables found
            tables = extract_tables(pdf_path, page_num)
            if tables:
                text += "\n\n[TABLE]\n" + tables + "\n[/TABLE]"

        full_text += f"\n\n--- Page {page_num + 1} ---\n{text}"

    doc.close()
    return full_text

# ── Main: Process all PDFs ──────────────────────────────
def process_all_documents():
    pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print("❌ No PDF files found in ./documents/")
        return

    print(f"Found {len(pdf_files)} PDF files\n")
    success = 0
    failed  = []

    for pdf_file in sorted(pdf_files):
        pdf_path = os.path.join(DOCUMENTS_DIR, pdf_file)
        try:
            text = process_pdf(pdf_path)

            out_name    = pdf_file.replace(".pdf", ".txt")
            output_path = os.path.join(PROCESSED_DIR, out_name)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"  ✅ Saved  → {out_name}")
            print(f"  📊 {len(text):,} characters extracted")
            success += 1

        except Exception as e:
            print(f"  ❌ Error  : {e}")
            failed.append(pdf_file)

    print(f"\n{'='*50}")
    print(f"✅ Success : {success}/{len(pdf_files)}")
    if failed:
        print(f"❌ Failed  : {len(failed)}")
        for f in failed:
            print(f"    • {f}")

if __name__ == "__main__":
    process_all_documents()