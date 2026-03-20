# Nile University RAG Assistant

A bilingual (Arabic/English) Retrieval-Augmented Generation (RAG) system for Nile University students. Students can ask questions about university regulations, admission requirements, academic programs, and more — in either Arabic or English.

---

## Architecture
```
Documents (PDF) → Ingestion + OCR → ChromaDB Vector Store → RAG Pipeline → FastAPI → Client
```

| Component | Technology |
|---|---|
| Document parsing | PyMuPDF + pdfplumber |
| OCR (scanned PDFs) | EasyOCR (CRAFT + CRNN) |
| Embeddings | intfloat/multilingual-e5-large |
| Vector store | ChromaDB |
| LLM | Qwen2.5:3b via Ollama |
| Language detection | langdetect |
| API | FastAPI |
| Deployment | Docker |

---

## Project Structure
```
university-rag/
├── documents/          # Source PDF files (13 documents)
├── processed/          # Extracted text files
├── chroma_db/          # ChromaDB vector store
├── embedding_model/    # Local copy of multilingual-e5-large
├── ollama_models/      # Local copy of qwen2.5:3b
├── ingest.py           # Document processing pipeline
├── rag_pipeline.py     # RAG pipeline + interactive CLI
├── app.py              # FastAPI application
├── Dockerfile          # Docker configuration
├── start.sh            # Container startup script
└── requirements.txt    # Python dependencies
```

---

## Documents

The system is trained on 13 official Nile University documents covering:

- MSc program bylaws (ITCS, EAS, MOT, EMBA)
- Undergraduate student manual
- Graduate studies manual
- Strategic plan
- Quality assurance bylaws
- Student self-service guides
- CS program catalog

---

## Setup

### Prerequisites

- Python 3.11+
- Ollama installed and running
- Docker (for containerized deployment)

### Local Setup
```bash
# Clone the repo
git clone https://github.com/omaradel227/university-rag
cd university-rag

# Install dependencies
pip install -r requirements.txt

# Pull the LLM
ollama pull qwen2.5:3b

# Process documents (first time only)
python3 ingest.py

# Build vector store and start CLI
python3 rag_pipeline.py
```

### Docker Deployment
```bash
# Save embedding model locally
python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-large')
model.save('./embedding_model')
"

# Copy Ollama models
mkdir -p ollama_models
cp -r ~/.ollama/models ollama_models/

# Build image
docker build -t university-rag .

# Run container
docker run -p 8000:8000 --memory=6g university-rag
```

---

## API

### POST /query
```
POST http://localhost:8000/query
Content-Type: application/json
```

**Request:**
```json
{
    "question": "What are the admission requirements for the MSc program?"
}
```

**Response:**
```json
{
    "question": "What are the admission requirements for the MSc program?",
    "answer": "The admission requirements for the MSc program at Nile University include:\n\n1. A completed application form including a personal statement.\n2. Official transcripts and degrees certified by the granting institution.\n3. A statement of purpose outlining objectives in joining the program.\n4. Documentary evidence of relevant professional experience.\n5. Two recent passport-size photographs.\n6. Photocopy of official ID or passport.\n7. Application fee as announced by the Financial Department.\n8. Three recommendation letters.\n9. A recent Curriculum Vitae.",
    "detected_language": "English",
    "sources": [
        {
            "filename": "MOT- PG Bylaws- Professional Master- March 2023 Stamped.txt",
            "language": "eng",
            "chunk_size": 989
        },
        {
            "filename": "NU Graduate Studies Manual 2018.txt",
            "language": "eng",
            "chunk_size": 922
        }
    ],
    "metrics": {
        "detected_language": "English",
        "retrieval_time_ms": 1615,
        "chunks_retrieved": 5,
        "context_length_chars": 4045,
        "prompt_length_chars": 4391,
        "llm_time_ms": 64596,
        "answer_length": 1003,
        "total_time_ms": 66694
    }
}
```

Arabic queries are also supported:
```json
{
    "question": "ما هي شروط القبول في برنامج الماجستير؟"
}
```

---

## RAG Pipeline

### Document Ingestion

- PyMuPDF extracts embedded text directly from PDFs
- EasyOCR handles genuinely scanned PDFs (CRAFT text detection + CRNN recognition)
- pdfplumber extracts tables
- Arabic PDFs with embedded Unicode text are extracted directly without OCR

### Chunking

- chunk_size: 1000 characters
- chunk_overlap: 100 characters
- Separators: paragraph breaks, newlines, Arabic punctuation

### Retrieval

- Embedding model: `intfloat/multilingual-e5-large` (supports Arabic + English)
- Vector store: ChromaDB (local, persistent)
- Top-K: 5 chunks per query
- Similarity search

### Generation

- Model: `qwen2.5:3b` via Ollama
- Temperature: 0.1
- Language detection via `langdetect` — enforces response language matches question language
- Bilingual system prompt with strict language instruction

---

## Example Output
```
Question: What are the admission requirements for the MSc program?

Detected language : English
Retrieval time    : 4496 ms
Chunks retrieved  : 5
  [1] MOT- PG Bylaws- Professional Master- March 2023 Stamped.txt - 989 chars
  [2] MOT- PG Bylaws- Professional Master- March 2023 Stamped.txt - 959 chars
  [3] NU Graduate Studies Manual 2018.txt - 282 chars
  [4] NU Graduate Studies Manual 2018.txt - 922 chars
  [5] NU Graduate Studies Manual 2018.txt - 577 chars

Answer:
The admission requirements for the MSc program at Nile University include:
1. A completed application form including an applicant's personal statement.
2. Official transcripts and degrees certified by the granting institution.
3. A statement of purpose outlining objectives in joining the program.
4. Documentary evidence of relevant professional experience.
5. Two recent passport-size photographs.
6. Photocopy of official ID or passport.
7. Application fee as announced by the Financial Department of the University.
8. Three recommendation letters.
9. A Curriculum Vitae.


```

---
## Performance

Measured on Apple M2 8GB (CPU inference):

| Metric | Value |
|---|---|
| Retrieval time | ~40-1600 ms |
| LLM time (local Mac) | ~11,000 ms |
| LLM time (Docker CPU) | ~65,000 ms |
| Total chunks in store | 1,868 |
| Documents indexed | 13 |

---

## Evaluation

Evaluated on 20 questions — 10 manually written + 10 auto-generated from documents.
Arabic and English questions both included.
Judge model: qwen2.5:3b via Ollama (local).

| Metric | Score | Description |
|---|---|---|
| Answer Relevancy | 0.8621 | Answers address the question asked |
| Context Recall | 0.9000 | Retrieved chunks contain the needed information |
| Faithfulness | n/a | Requires larger judge model (≥7b) |
| Context Precision | n/a | Requires larger judge model (≥7b) |

**Latency across 20 test questions:**

| | Value |
|---|---|
| Average | 4,643 ms |
| Min | 2,945 ms |
| Max | 7,209 ms |

> Note: Faithfulness and Context Precision metrics require an LLM judge with strong
> instruction-following capability. qwen2.5:3b produces malformed responses for these
> structured scoring tasks. Running evaluation with GPT-4o-mini would yield all 4 scores.
---
## Known Limitations

- LLM runs on CPU — responses take 10-65 seconds depending on environment
- 2 scanned PDFs have OCR artifacts due to pre-existing corrupted text layers in the source files — semantic search still works correctly
- Broad/vague questions perform worse than specific factual questions
- `qwen2.5:3b` occasionally drifts on language — mitigated by explicit language detection and prompt enforcement

---

## Future Improvements

- GPU deployment for faster inference
- Swap Ollama for API-based LLM (OpenAI/Anthropic) to reduce latency
- Add streaming response support
- Expand document base as new university policies are published
- Add conversation history for multi-turn Q&A
