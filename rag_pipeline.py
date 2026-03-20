import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langdetect import detect
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

PROCESSED_DIR = "./processed"
CHROMA_DIR    = "./chroma_db"
EMBED_MODEL   = "intfloat/multilingual-e5-large"
OLLAMA_MODEL  = "qwen2.5:3b"
TOP_K         = 5

DOCUMENT_LANGUAGES = {
    "EAS - PG Bylaws-2022 July - Stamped.txt"                                          : "ara+eng",
    "MOT- PG Bylaws- Professional Master- March 2023 Stamped.txt"                      : "eng",
    "2017-Nov-EMBA Bylaws Approved by SCU.txt"                                         : "eng",
    "Approved UG Manual Oct2024-UC(27Oct24)_BOT(2Dec24).txt"                           : "eng",
    "Internship or Service Learning Program - BBA.txt"                                 : "eng",
    "NU Extracurricular - SelfService-signed.txt"                                      : "eng",
    "NU Graduate Studies Manual 2018.txt"                                              : "eng",
    "Payment and Services Steps on Selfservice-signed.txt"                             : "eng",
    "Students Registration Process Guide on PowerCampus SelfService (for students).txt": "eng",
    "ITCS-PG Bylaws-MSc. Program 2021- Sreachable.txt"                                : "ara+eng",
    "NU Computer Science Program - v17docx.txt"                                        : "ara+eng",
    "الخطة الإستراتيجية.txt"                                                          : "ara",
    "اللائحة الداخلية لوحدة ضمان الجودة للكلية.txt"                                  : "ara",
}

def load_documents():
    docs = []
    txt_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".txt")]

    if not txt_files:
        print("No processed .txt files found. Run ingest.py first.")
        return []

    print(f"Loading {len(txt_files)} documents...")

    for txt_file in sorted(txt_files):
        path = os.path.join(PROCESSED_DIR, txt_file)
        lang = DOCUMENT_LANGUAGES.get(txt_file, "ara+eng")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        docs.append(Document(
            page_content=content,
            metadata={
                "source"  : txt_file,
                "language": lang,
                "path"    : path,
            }
        ))
        print(f"  {txt_file} ({lang}) - {len(content):,} chars")

    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "،", " ", ""],
    )

    chunks = []
    for doc in docs:
        split = splitter.split_documents([doc])
        for chunk in split:
            chunk.metadata.update(doc.metadata)
        chunks.extend(split)

    print(f"{len(docs)} documents -> {len(chunks)} chunks")
    return chunks

def build_vector_store(chunks, force_rebuild=False):
    embedding_fn = HuggingFaceEmbeddings(
    model_name="./embedding_model",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

    if os.path.exists(CHROMA_DIR) and not force_rebuild:
        print(f"Loading existing ChromaDB from {CHROMA_DIR}")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embedding_fn,
        )
        print(f"{vectorstore._collection.count()} chunks in store")
        return vectorstore

    print(f"Building ChromaDB vector store...")
    print(f"Embedding model : {EMBED_MODEL}")
    print(f"Chunks to embed : {len(chunks)}")
    print(f"This may take a few minutes on first run...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=CHROMA_DIR,
    )
    vectorstore.persist()
    print(f"Vector store built and saved to {CHROMA_DIR}")
    print(f"{vectorstore._collection.count()} chunks stored")
    return vectorstore

def query_rag(question, vectorstore, verbose=False):
    metrics = {}
    total_start = time.time()

    try:
        lang_code = detect(question)
        if lang_code == "ar":
            lang_instruction = "You MUST answer in Arabic only. Do not use any English in your answer."
            lang_label = "Arabic"
        else:
            lang_instruction = "You MUST answer in English only. Do not use any Arabic in your answer."
            lang_label = "English"
    except:
        lang_instruction = "Answer in the same language as the question."
        lang_label = "unknown"

    metrics["detected_language"] = lang_label

    retrieval_start = time.time()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    retrieved = retriever.invoke(question)
    metrics["retrieval_time_ms"] = round((time.time() - retrieval_start) * 1000)
    metrics["chunks_retrieved"]  = len(retrieved)
    metrics["chunk_sizes"]       = [len(doc.page_content) for doc in retrieved]
    metrics["sources"]           = [doc.metadata.get("source", "unknown") for doc in retrieved]

    if not retrieved:
        return "لم أجد معلومات كافية للإجابة.", [], metrics

    context_parts = []
    for i, doc in enumerate(retrieved):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)
    metrics["context_length_chars"] = len(context)

    if verbose:
        print(f"Detected language : {lang_label}")
        print(f"Retrieval time    : {metrics['retrieval_time_ms']} ms")
        print(f"Chunks retrieved  : {len(retrieved)}")
        for i, doc in enumerate(retrieved):
            print(f"  [{i+1}] {doc.metadata.get('source')} - {len(doc.page_content)} chars")

    prompt = f"""You are an academic assistant for Nile University.
{lang_instruction}
Use only the information provided in the context below. If you cannot find the answer, say so clearly in the correct language.

Context:
{context}

Question: {question}

Answer ({lang_label}):"""

    metrics["prompt_length_chars"] = len(prompt)

    llm_start = time.time()
    llm       = OllamaLLM(model=OLLAMA_MODEL, temperature=0.1)
    answer    = llm.invoke(prompt)
    metrics["llm_time_ms"]   = round((time.time() - llm_start) * 1000)
    metrics["answer_length"] = len(answer.strip())
    metrics["total_time_ms"] = round((time.time() - total_start) * 1000)

    return answer.strip(), retrieved, metrics

def main():
    print("=" * 60)
    print("Nile University RAG Assistant")
    print("=" * 60)

    docs = load_documents()
    if not docs:
        return

    chunks      = chunk_documents(docs)
    vectorstore = build_vector_store(chunks)

    print("\nSystem ready. Type your question (or 'exit' to quit).")
    print("Arabic and English questions both supported.\n")

    while True:
        question = input("Question: ").strip()

        if not question:
            continue
        if question.lower() in ["exit", "quit", "خروج"]:
            print("Goodbye.")
            break

        print("\nSearching...\n")
        answer, sources, metrics = query_rag(question, vectorstore, verbose=True)

        print(f"\nAnswer:\n{answer}")

        print(f"\nSources:")
        for doc in sources:
            print(f"  - {doc.metadata.get('source', 'unknown')}")

        print(f"\nPerformance:")
        print(f"  Language detected : {metrics['detected_language']}")
        print(f"  Retrieval time    : {metrics['retrieval_time_ms']} ms")
        print(f"  LLM time          : {metrics['llm_time_ms']} ms")
        print(f"  Total time        : {metrics['total_time_ms']} ms")
        print(f"  Chunks retrieved  : {metrics['chunks_retrieved']}")
        print(f"  Context length    : {metrics['context_length_chars']} chars")
        print(f"  Answer length     : {metrics['answer_length']} chars")
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()