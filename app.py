import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langdetect import detect
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

CHROMA_DIR   = "./chroma_db"
EMBED_MODEL  = "intfloat/multilingual-e5-large"
OLLAMA_MODEL = "qwen2.5:3b"
TOP_K        = 5

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding model...")
    embedding_fn = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError(f"ChromaDB not found at {CHROMA_DIR}. Run rag_pipeline.py first.")

    print("Loading ChromaDB...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_fn,
    )
    print(f"{vectorstore._collection.count()} chunks loaded.")

    print("Loading LLM...")
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.1)

    state["vectorstore"] = vectorstore
    state["llm"]         = llm

    print("System ready.")
    yield

    state.clear()

app = FastAPI(
    title="Nile University RAG Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

class QueryRequest(BaseModel):
    question: str

class SourceItem(BaseModel):
    filename: str
    language: str
    chunk_size: int

class QueryResponse(BaseModel):
    question:          str
    answer:            str
    detected_language: str
    sources:           list[SourceItem]
    metrics: dict

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    metrics     = {}
    total_start = time.time()

    try:
        lang_code = detect(question)
        if lang_code == "ar":
            lang_instruction = "You MUST answer in Arabic only. Do not use any English in your answer."
            lang_label       = "Arabic"
        else:
            lang_instruction = "You MUST answer in English only. Do not use any Arabic in your answer."
            lang_label       = "English"
    except:
        lang_instruction = "Answer in the same language as the question."
        lang_label       = "unknown"

    metrics["detected_language"] = lang_label

    retrieval_start = time.time()
    retriever = state["vectorstore"].as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    retrieved = retriever.invoke(question)
    metrics["retrieval_time_ms"] = round((time.time() - retrieval_start) * 1000)
    metrics["chunks_retrieved"]  = len(retrieved)

    if not retrieved:
        raise HTTPException(status_code=404, detail="No relevant information found.")

    context_parts = []
    for i, doc in enumerate(retrieved):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)
    metrics["context_length_chars"] = len(context)

    prompt = f"""You are an academic assistant for Nile University.
{lang_instruction}
Use only the information provided in the context below. If you cannot find the answer, say so clearly in the correct language.

Context:
{context}

Question: {question}

Answer ({lang_label}):"""

    metrics["prompt_length_chars"] = len(prompt)

    llm_start      = time.time()
    answer         = state["llm"].invoke(prompt)
    metrics["llm_time_ms"]   = round((time.time() - llm_start) * 1000)
    metrics["answer_length"] = len(answer.strip())
    metrics["total_time_ms"] = round((time.time() - total_start) * 1000)

    sources = [
        SourceItem(
            filename=doc.metadata.get("source", "unknown"),
            language=doc.metadata.get("language", "unknown"),
            chunk_size=len(doc.page_content),
        )
        for doc in retrieved
    ]

    return QueryResponse(
        question=question,
        answer=answer.strip(),
        detected_language=lang_label,
        sources=sources,
        metrics=metrics,
    )