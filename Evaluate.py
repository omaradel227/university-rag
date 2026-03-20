import os
import time
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langdetect import detect

CHROMA_DIR   = "./chroma_db"
EMBED_MODEL  = "./embedding_model"
OLLAMA_MODEL = "qwen2.5:3b"
TOP_K        = 5
RESULTS_FILE = "./evaluation_results.json"

MANUAL_TEST_SET = [
    {
        "question": "What is the minimum GPA required to graduate from the MSc program?",
        "ground_truth": "Students must maintain a cumulative GPA of 3.0 or higher to graduate from the MSc program at Nile University."
    },
    {
        "question": "What are the admission requirements for the MSc program?",
        "ground_truth": "Admission requirements include a completed application form, official transcripts, a statement of purpose, two passport-size photographs, a photocopy of official ID or passport, application fee, three recommendation letters, and a curriculum vitae."
    },
    {
        "question": "How many credit hours are required to graduate from the MSc program?",
        "ground_truth": "Students must complete 36 credit hours to graduate from the MSc program."
    },
    {
        "question": "What is the TOEFL score required for admission?",
        "ground_truth": "A TOEFL ITP score of 500 or an IELTS score of 6 is required from applicants who did not receive their prior degrees from an English-speaking institution."
    },
    {
        "question": "What happens if a student's GPA falls below 3.0?",
        "ground_truth": "A student whose GPA falls below 3.0 is put on academic probation and given one semester to correct this. If the GPA remains below 3.0 at the end of the probationary period, the student will not be eligible to receive the academic degree."
    },
    {
        "question": "How many credit hours can be transferred from another institution?",
        "ground_truth": "Up to 9 credit hours may be transferred from another accredited institution towards the ITCS Master's degree, with a grade of B or better, and students cannot transfer more than 25% of the required credit hours."
    },
    {
        "question": "What are the MSc tracks available at the ITCS school?",
        "ground_truth": "The four MSc tracks available are: Informatics (MSITCS-INF), Information Security (MSITCS-IS), Software Engineering (MSITCS-SWE), and Wireless Technologies (MSITCS-WT)."
    },
    {
        "question": "What is the grading scale at Nile University?",
        "ground_truth": "The grading scale includes A+ and A (4.0, Excellent), A- (3.7), B+ (3.3, Very Good), B (3.0, Good), B- (2.7), C+ (2.3), C (2.0), and F (0.0, Fail)."
    },
    {
        "question": "ما هو الحد الأدنى للمعدل التراكمي للتخرج؟",
        "ground_truth": "يجب أن يحافظ الطالب على معدل تراكمي لا يقل عن 3.0 للتخرج من برنامج الماجستير في جامعة النيل."
    },
    {
        "question": "ما هي مسارات الماجستير المتاحة في كلية تكنولوجيا المعلومات؟",
        "ground_truth": "المسارات المتاحة هي: المعلوماتية، وأمن المعلومات، وهندسة البرمجيات، وتكنولوجيا الاتصالات اللاسلكية."
    },
]

def generate_questions_from_docs(vectorstore, llm, n=10):
    print("Auto-generating questions from documents...")
    collection = vectorstore._collection
    results    = collection.get(limit=50)
    chunks     = results["documents"]

    import random
    selected = random.sample(chunks, min(n, len(chunks)))

    generated = []
    for i, chunk in enumerate(selected):
        if len(chunk.strip()) < 100:
            continue

        prompt = f"""Read the following text from a university document and generate one specific factual question that can be answered from this text. Also provide the answer.

Text:
{chunk[:800]}

Respond in this exact format:
QUESTION: <your question here>
ANSWER: <the answer here>"""

        try:
            response = llm.invoke(prompt)
            lines    = response.strip().split("\n")
            question = ""
            answer   = ""
            for line in lines:
                if line.startswith("QUESTION:"):
                    question = line.replace("QUESTION:", "").strip()
                elif line.startswith("ANSWER:"):
                    answer = line.replace("ANSWER:", "").strip()

            if question and answer:
                generated.append({
                    "question":     question,
                    "ground_truth": answer,
                    "auto":         True
                })
                print(f"  [{i+1}] Generated: {question[:80]}...")
        except Exception as e:
            print(f"  [{i+1}] Failed to generate: {e}")

        if len(generated) >= n:
            break

    return generated

def run_rag(question, vectorstore, llm):
    try:
        lang_code = detect(question)
        if lang_code == "ar":
            lang_instruction = "You MUST answer in Arabic only."
            lang_label       = "Arabic"
        else:
            lang_instruction = "You MUST answer in English only."
            lang_label       = "English"
    except:
        lang_instruction = "Answer in the same language as the question."
        lang_label       = "unknown"

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    retrieved = retriever.invoke(question)

    if not retrieved:
        return "", [], 0

    context_parts = []
    for i, doc in enumerate(retrieved):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an academic assistant for Nile University.
{lang_instruction}
Use only the information provided in the context below. If you cannot find the answer, say so clearly.

Context:
{context}

Question: {question}

Answer ({lang_label}):"""

    start  = time.time()
    answer = llm.invoke(prompt)
    latency = round((time.time() - start) * 1000)

    return answer.strip(), retrieved, latency

def main():
    print("=" * 60)
    print("Nile University RAG Evaluation")
    print("=" * 60)

    print("\nLoading embedding model...")
    embedding_fn = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Loading ChromaDB...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_fn,
    )
    print(f"{vectorstore._collection.count()} chunks loaded.")

    print("Loading LLM...")
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.1)

    auto_questions = generate_questions_from_docs(vectorstore, llm, n=10)
    test_set       = MANUAL_TEST_SET + auto_questions
    print(f"\nTotal test questions: {len(test_set)} ({len(MANUAL_TEST_SET)} manual + {len(auto_questions)} auto-generated)")

    print("\nRunning RAG pipeline on all questions...")
    questions     = []
    answers       = []
    contexts      = []
    ground_truths = []
    latencies     = []

    for i, item in enumerate(test_set):
        question    = item["question"]
        ground_truth = item["ground_truth"]
        print(f"  [{i+1}/{len(test_set)}] {question[:70]}...")

        answer, retrieved, latency = run_rag(question, vectorstore, llm)
        context_texts = [doc.page_content for doc in retrieved]

        questions.append(question)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(ground_truth)
        latencies.append(latency)

        print(f"    Latency: {latency} ms | Answer: {answer[:60]}...")

    print("\nRunning RAGAS evaluation...")
    ragas_dataset = Dataset.from_dict({
        "question":   questions,
        "answer":     answers,
        "contexts":   contexts,
        "ground_truth": ground_truths,
    })

    ragas_llm        = LangchainLLMWrapper(ChatOllama(model=OLLAMA_MODEL, temperature=0.1))
    ragas_embeddings = LangchainEmbeddingsWrapper(embedding_fn)

    results = evaluate( 
        dataset=ragas_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    def safe_score(val):
        if isinstance(val, list):
            valid = [v for v in val if v is not None]
            return sum(valid) / len(valid) if valid else 0.0
        return val if val is not None else 0.0

    print(f"  Faithfulness      : {safe_score(results['faithfulness']):.4f}")
    print(f"  Answer Relevancy  : {safe_score(results['answer_relevancy']):.4f}")
    print(f"  Context Precision : {safe_score(results['context_precision']):.4f}")
    print(f"  Context Recall    : {safe_score(results['context_recall']):.4f}")
    print(f"\nLatency:")
    print(f"  Average : {round(sum(latencies) / len(latencies))} ms")
    print(f"  Min     : {min(latencies)} ms")
    print(f"  Max     : {max(latencies)} ms")

    output = {
        "summary": {
            "faithfulness":      safe_score(results["faithfulness"]),
            "answer_relevancy":  safe_score(results["answer_relevancy"]),
            "context_precision": safe_score(results["context_precision"]),
            "context_recall":    safe_score(results["context_recall"]),
            "latency_avg_ms":    round(sum(latencies) / len(latencies)),
            "latency_min_ms":    min(latencies),
            "latency_max_ms":    max(latencies),
            "total_questions":   len(test_set),
        },
        "per_question": [
            {
                "question":      questions[i],
                "answer":        answers[i],
                "ground_truth":  ground_truths[i],
                "latency_ms":    latencies[i],
                "context_sources": [
                    doc.metadata.get("source", "unknown")
                    for doc in run_rag(questions[i], vectorstore, llm)[1]
                ] if i < 3 else []
            }
            for i in range(len(questions))
        ]
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nFull results saved to {RESULTS_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()