import math
import os
import ssl
import time
import warnings
from collections import Counter
from difflib import SequenceMatcher

import certifi
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from httpx import Client
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm

load_dotenv()

# Ignore warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

try:
    import evaluate
except ImportError:
    evaluate = None

try:
    from bert_score import score as bert_score
except ImportError:
    bert_score = None

ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


# ============================================================================
# Evaluation metrics functions
# ============================================================================
def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0.0


def is_similar(text1, text2, threshold=0.25):
    """Determine if a retrieved document chunk (text1) is relevant to the reference answer (text2).

    Because document chunks are long (1000-2000 chars) while reference answers are short (30-100 words),
    standard full-string SequenceMatcher ratio collapses below 0.1.
    We check substring containment or key token recall to evaluate retrieval relevance accurately.
    """
    t1_lower = str(text1).lower()
    t2_lower = str(text2).lower()
    if t2_lower in t1_lower or t1_lower in t2_lower:
        return True
    tokens2 = [tok for tok in t2_lower.split() if len(tok) > 2]
    if not tokens2:
        return False
    matches = sum(1 for tok in tokens2 if tok in t1_lower)
    return (matches / len(tokens2)) >= threshold


def compute_precision_at_k(relevant_at_k):
    return safe_divide(sum(relevant_at_k), len(relevant_at_k))


def compute_recall_at_k(relevant_at_k, total_relevant):
    return safe_divide(sum(relevant_at_k), total_relevant)


def compute_mrr(relevant_at_k):
    try:
        first_relevant_rank = next(i for i, r in enumerate(relevant_at_k, 1) if r)
        return 1.0 / first_relevant_rank
    except StopIteration:
        return 0.0


def compute_dcg(relevances):
    return sum((2**int(rel) - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevances))


def compute_ndcg(relevant_at_k):
    dcg = compute_dcg(relevant_at_k)
    idcg = compute_dcg(sorted(relevant_at_k, reverse=True))
    return safe_divide(dcg, idcg)


def compute_rouge_l(reference, candidate):
    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(reference, candidate)["rougeL"].fmeasure
    # Fallback LCS F1
    matcher = SequenceMatcher(None, reference, candidate)
    match_len = sum(triple.size for triple in matcher.get_matching_blocks())
    p = safe_divide(match_len, len(candidate))
    r = safe_divide(match_len, len(reference))
    return safe_divide(2 * p * r, p + r)


def compute_bleu(reference, candidate):
    if evaluate is not None:
        try:
            bleu = evaluate.load("bleu")
            return bleu.compute(predictions=[candidate], references=[[reference]])["bleu"]
        except Exception:
            pass
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    if not ref_tokens or not cand_tokens:
        return 0.0
    smooth = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smooth)


def compute_bert_score(reference, candidate):
    if bert_score is not None:
        try:
            _, _, f1 = bert_score([candidate], [reference], lang="en")
            return f1.mean().item()
        except Exception:
            pass
    # Robust fallback: token/character sequence similarity
    return SequenceMatcher(None, reference.lower(), candidate.lower()).ratio()


def compute_exact_match(reference, candidate):
    return int(candidate.strip().lower() == reference.strip().lower())


def compute_f1(reference, candidate):
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    common = Counter(ref_tokens) & Counter(cand_tokens)
    num_common = sum(common.values())

    precision = safe_divide(num_common, len(cand_tokens))
    recall = safe_divide(num_common, len(ref_tokens))

    return safe_divide(2 * precision * recall, precision + recall)


# ============================================================================
# RAG system setup and retrieval functions
# ============================================================================
def load_vector_db(vector_db_path="vector_db"):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
    return vector_db


def probabilities_from_distances(distances, temperature: float = 0.25):
    """Convert Chroma vector distance scores into a well-calibrated probability distribution.

    Chroma's distance metric (e.g. squared L2) produces numbers with large magnitudes
    (e.g., 300+). Directly taking math.exp(-distance) causes floating point underflow
    where the top candidate gets 1.0 probability and all others collapse to 0.0.

    To produce a scale-invariant, well-calibrated distribution:
    1. Normalize distances into relative similarities in [0.0, 1.0]:
       similarity_i = 1.0 - (d_i - d_min) / (d_max - d_min)
       where 1.0 is the best match and 0.0 is the farthest in the candidate pool.
    2. Apply temperature-scaled softmax:
       p_i = exp((similarity_i - max_sim) / temperature) / sum_j exp((similarity_j - max_sim) / temperature)
    """
    if not distances:
        return []
    if len(distances) == 1:
        return [1.0]

    d_min = min(distances)
    d_max = max(distances)
    span = d_max - d_min

    if span < 1e-8:
        return [1.0 / len(distances)] * len(distances)

    similarities = [1.0 - (d - d_min) / span for d in distances]
    sim_max = max(similarities)
    weights = [math.exp((s - sim_max) / temperature) for s in similarities]
    total = sum(weights) or 1.0
    return [w / total for w in weights]


def select_documents_rag(vector_db, question: str, retrieval: str = "top_k", k: int = 5, p: float = 0.85, candidate_pool: int = 20, temperature: float = 0.25):
    """Retrieve documents using either Top-k (fixed count) or Top-p (nucleus probability threshold)."""
    pairs = vector_db.similarity_search_with_score(question, k=candidate_pool)
    if not pairs:
        return []

    probabilities = probabilities_from_distances([score for _, score in pairs], temperature=temperature)
    selected = []
    cumulative = 0.0

    for index, ((doc, dist), prob) in enumerate(zip(pairs, probabilities), start=1):
        if retrieval == "top_k" and index > k:
            break
        cumulative += prob
        selected.append(doc)
        if retrieval == "top_p" and cumulative >= p:
            break

    return selected


def setup_llm(api_key="", model_name="openai/gpt-oss-20b"):
    groq_api_key = api_key or os.getenv("GROQ_API_KEY", "").strip()
    if groq_api_key and groq_api_key != "API_KEY":
        from langchain_groq import ChatGroq
        return ChatGroq(
            groq_api_key=groq_api_key,
            model=model_name,
            http_client=Client(verify=ssl_context)
        )
    else:
        from langchain_ollama import ChatOllama
        chat_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
        return ChatOllama(model=chat_model, temperature=0)


def setup_prompt():
    template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
    return ChatPromptTemplate.from_template(template)


def load_evaluation_dataset(file_path, sample_size=5, random_state=42):
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(file_path, encoding="ISO-8859-1")

    # Handle UTF-8 BOM if present in column names
    col_map = {c: "Question" for c in df.columns if "question" in c.lower()}
    col_map.update({c: "Answer" for c in df.columns if "answer" in c.lower()})
    df = df.rename(columns=col_map)

    if sample_size and len(df) > sample_size:
        df = df.head(sample_size) if random_state is None else df.sample(n=sample_size, random_state=random_state)

    return df[["Question", "Answer"]].apply(
        lambda row: {"question": str(row["Question"]), "answer": str(row["Answer"])}, axis=1
    ).tolist()


# ============================================================================
# Main evaluation function
# ============================================================================
def evaluate_rag_system(evaluation_dataset, vector_db, llm, prompt, retrieval_mode="top_k", k=5, p=0.85, temperature=0.25):
    """Evaluate the RAG system under a specified retrieval strategy."""
    metrics = {
        "question": [], "reference_answer": [], "generated_answer": [],
        "retrieval_mode": [], "num_retrieved_docs": [],
        "precision_at_k": [], "recall_at_k": [], "mrr": [], "ndcg": [],
        "rouge_l": [], "bleu": [], "bert_score": [], "exact_match": [],
        "f1": [], "response_time": []
    }

    print(f"\n--- Running Evaluation: {retrieval_mode.upper()} (k={k}, p={p}) ---")
    for sample in tqdm(evaluation_dataset, desc=f"Evaluating {retrieval_mode}"):
        question = sample["question"]
        reference_answer = sample["answer"]

        # Retrieve relevant documents
        start_time = time.time()
        if hasattr(vector_db, "get_relevant_documents"):
            # Backwards compatibility for raw retriever objects
            relevant_documents = vector_db.get_relevant_documents(question)
        else:
            relevant_documents = select_documents_rag(
                vector_db=vector_db,
                question=question,
                retrieval=retrieval_mode,
                k=k,
                p=p,
                temperature=temperature,
            )
        retrieval_time = time.time() - start_time

        # Generate answer
        context = "\n\n".join([doc.page_content for doc in relevant_documents])
        chain = prompt | llm
        start_gen = time.time()
        try:
            generated_answer = chain.invoke({"context": context, "question": question})
            generated_answer_text = (
                generated_answer.content
                if hasattr(generated_answer, "content")
                else str(generated_answer)
            )
        except Exception as e:
            generated_answer_text = f"Error: {e}"
        generation_time = time.time() - start_gen

        response_time = retrieval_time + generation_time

        # Extract relevance matching for retrieval metrics
        relevant_at_k = [is_similar(doc.page_content, reference_answer) for doc in relevant_documents[:5]]
        if not relevant_at_k:
            relevant_at_k = [False]

        # Calculate metrics
        metrics["question"].append(question)
        metrics["reference_answer"].append(reference_answer)
        metrics["generated_answer"].append(generated_answer_text)
        metrics["retrieval_mode"].append(retrieval_mode)
        metrics["num_retrieved_docs"].append(len(relevant_documents))
        metrics["precision_at_k"].append(compute_precision_at_k(relevant_at_k))
        metrics["recall_at_k"].append(compute_recall_at_k(relevant_at_k, len(relevant_documents)))
        metrics["mrr"].append(compute_mrr(relevant_at_k))
        metrics["ndcg"].append(compute_ndcg(relevant_at_k))

        metrics["rouge_l"].append(compute_rouge_l(reference_answer, generated_answer_text))
        metrics["bleu"].append(compute_bleu(reference_answer, generated_answer_text))
        metrics["bert_score"].append(compute_bert_score(reference_answer, generated_answer_text))
        metrics["exact_match"].append(compute_exact_match(reference_answer, generated_answer_text))
        metrics["f1"].append(compute_f1(reference_answer, generated_answer_text))
        metrics["response_time"].append(response_time)

    return metrics


def print_evaluation_summary(metrics, label="Evaluation"):
    print(f"\n==================== {label.upper()} SUMMARY ====================")
    summary_keys = [
        "num_retrieved_docs", "precision_at_k", "recall_at_k", "mrr", "ndcg",
        "rouge_l", "bleu", "bert_score", "exact_match", "f1", "response_time"
    ]
    for key in summary_keys:
        if key in metrics and metrics[key]:
            val = np.mean(metrics[key])
            fmt = f"{val:.4f}" if "time" not in key else f"{val:.2f}s"
            print(f"  Average {key.replace('_', ' ').title():<25}: {fmt}")
    print(f"  Processed Samples            : {len(metrics['precision_at_k'])}")
    print("================================================================")


# ============================================================================
# Main execution: Compare Top-k vs Top-p
# ============================================================================
if __name__ == "__main__":
    # Auto-load configurations with sensible defaults
    vector_db_path = os.getenv("VECTOR_DB_PATH", "vector_db_eng_hornbill")
    if not os.path.exists(vector_db_path) and os.path.exists("vector_db"):
        vector_db_path = "vector_db"

    evaluation_dataset_path = os.getenv("EVAL_DATASET_PATH", "eng_book.csv")
    if not os.path.exists(evaluation_dataset_path) and os.path.exists("science_book_ques.csv"):
        evaluation_dataset_path = "science_book_ques.csv"

    groq_api_key = os.getenv("GROQ_API_KEY", "")
    model_name = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")

    print(f"Loading Vector DB from: {vector_db_path}")
    print(f"Loading Evaluation Dataset from: {evaluation_dataset_path}")
    print(f"LLM Model: {model_name}")

    # Setup
    vector_db = load_vector_db(vector_db_path)
    llm = setup_llm(groq_api_key, model_name)
    prompt = setup_prompt()

    # Load 5 representative evaluation questions for comparative benchmark
    dataset_sample_size = int(os.getenv("EVAL_SAMPLE_SIZE", "5"))
    evaluation_dataset = load_evaluation_dataset(evaluation_dataset_path, sample_size=dataset_sample_size, random_state=42)

    # 1. Run Top-k Evaluation (fixed k=5)
    metrics_top_k = evaluate_rag_system(
        evaluation_dataset, vector_db, llm, prompt,
        retrieval_mode="top_k", k=5, p=0.85
    )
    df_top_k = pd.DataFrame(metrics_top_k)
    df_top_k.to_csv("evaluation_results_top_k.csv", index=False)
    print_evaluation_summary(metrics_top_k, "Top-k (k=5)")

    # 2. Run Top-p Evaluation (nucleus cumulative probability cutoff p=0.85)
    metrics_top_p = evaluate_rag_system(
        evaluation_dataset, vector_db, llm, prompt,
        retrieval_mode="top_p", k=5, p=0.85
    )
    df_top_p = pd.DataFrame(metrics_top_p)
    df_top_p.to_csv("evaluation_results_top_p.csv", index=False)
    print_evaluation_summary(metrics_top_p, "Top-p (p=0.85)")

    # 3. Create Side-by-Side Comparison Summary
    comparison = {
        "Metric": [
            "Avg Retrieved Docs", "Precision@k", "Recall@k", "MRR", "NDCG",
            "ROUGE-L", "BLEU", "BERT/Semantic Score", "Exact Match", "Token F1", "Response Time (s)"
        ],
        "Top-k (k=5)": [
            round(float(np.mean(metrics_top_k["num_retrieved_docs"])), 2),
            round(float(np.mean(metrics_top_k["precision_at_k"])), 4),
            round(float(np.mean(metrics_top_k["recall_at_k"])), 4),
            round(float(np.mean(metrics_top_k["mrr"])), 4),
            round(float(np.mean(metrics_top_k["ndcg"])), 4),
            round(float(np.mean(metrics_top_k["rouge_l"])), 4),
            round(float(np.mean(metrics_top_k["bleu"])), 4),
            round(float(np.mean(metrics_top_k["bert_score"])), 4),
            round(float(np.mean(metrics_top_k["exact_match"])), 4),
            round(float(np.mean(metrics_top_k["f1"])), 4),
            round(float(np.mean(metrics_top_k["response_time"])), 2),
        ],
        "Top-p (p=0.85)": [
            round(float(np.mean(metrics_top_p["num_retrieved_docs"])), 2),
            round(float(np.mean(metrics_top_p["precision_at_k"])), 4),
            round(float(np.mean(metrics_top_p["recall_at_k"])), 4),
            round(float(np.mean(metrics_top_p["mrr"])), 4),
            round(float(np.mean(metrics_top_p["ndcg"])), 4),
            round(float(np.mean(metrics_top_p["rouge_l"])), 4),
            round(float(np.mean(metrics_top_p["bleu"])), 4),
            round(float(np.mean(metrics_top_p["bert_score"])), 4),
            round(float(np.mean(metrics_top_p["exact_match"])), 4),
            round(float(np.mean(metrics_top_p["f1"])), 4),
            round(float(np.mean(metrics_top_p["response_time"])), 2),
        ],
    }
    df_comp = pd.DataFrame(comparison)
    df_comp.to_csv("evaluation_results_comparison.csv", index=False)
    print("\n================== TOP-K vs TOP-P COMPARISON ==================")
    print(df_comp.to_string(index=False))
    print("================================================================")
    print("\nSaved detailed results to:")
    print("  - evaluation_results_top_k.csv")
    print("  - evaluation_results_top_p.csv")
    print("  - evaluation_results_comparison.csv")
