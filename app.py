from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal
import math
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

app = FastAPI()

# Function to load vector database
def load_vector_db(vector_db_path):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
    return vector_db

# Initialize vector database
vector_db_path = os.getenv("VECTOR_DB_PATH", "vector_db")
vector_db = load_vector_db(vector_db_path)
print(f"Vector database '{vector_db_path}' loaded successfully\n")

CANDIDATE_POOL = 20

groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
use_groq = bool(groq_api_key) and groq_api_key != "API_KEY"

if use_groq:
    from langchain_groq import ChatGroq

    chat_model = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model=chat_model,
    )
    print(f"Using Groq model '{chat_model}' for answers\n")
else:
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
    llm = ChatOllama(model=chat_model, temperature=0)
    print(f"GROQ_API_KEY not set. Using local Ollama model '{chat_model}'\n")


def probabilities_from_distances(distances, temperature: float = 0.25):
    """Turn Chroma vector distances into a normalized probability distribution.

    Chroma's distance metric (e.g. squared L2) can produce values with large magnitudes
    (e.g. 300+). Directly computing math.exp(-distance) causes floating-point underflow
    where the first item receives 1.0 probability and all others receive 0.0.

    To produce a scale-invariant, well-calibrated probability distribution:
    1. Normalize distances into relative similarities in [0.0, 1.0]:
       similarity_i = 1.0 - (d_i - d_min) / (d_max - d_min)
       (where closest match = 1.0, farthest in candidate pool = 0.0)
    2. Apply temperature-scaled softmax:
       p_i = exp((sim_i - max_sim) / temperature) / sum(...)
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
    return [weight / total for weight in weights]


def select_documents(question: str, retrieval: str, k: int, p: float, temperature: float = 0.25):
    """Top-k keeps a fixed count. Top-p keeps the closest chunks until their probabilities add up to p.

    Probabilities are calculated over the same candidate pool in both modes,
    so the numbers can be compared.
    """
    pairs = vector_db.similarity_search_with_score(question, k=CANDIDATE_POOL)
    if not pairs:
        return []

    probabilities = probabilities_from_distances([score for _, score in pairs], temperature=temperature)
    selected = []
    cumulative = 0.0

    for index, ((doc, distance), probability) in enumerate(zip(pairs, probabilities), start=1):
        if retrieval == "top_k" and index > k:
            break
        cumulative += probability
        selected.append({
            "document": doc,
            "distance": float(distance),
            "probability": probability,
            "cumulative_probability": cumulative,
        })
        if retrieval == "top_p" and cumulative >= p:
            break

    return selected

# Define the template for responding with document context
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

class Query(BaseModel):
    question: str
    retrieval: Literal["top_k", "top_p"] = "top_k"
    k: int = Field(default=5, ge=1, le=20)
    p: float = Field(default=0.85, gt=0, le=1)
    temperature: float = Field(default=0.25, gt=0, le=2.0)

class DocumentInfo(BaseModel):
    page: str
    link: str
    snippet: str
    probability: float
    cumulative_probability: float
    distance: float

class Response(BaseModel):
    answer: str
    retrieval: str
    k: int
    p: float
    retrieved_documents: List[DocumentInfo]

def answer_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(getattr(block, "text", block)))
        return "\n".join(part for part in parts if part)
    return str(content)

@app.post("/ask", response_model=Response)
async def ask_question(query: Query):
    try:
        selected = select_documents(query.question, query.retrieval, query.k, query.p, query.temperature)

        doc_info = []
        for item in selected:
            doc = item["document"]
            doc_info.append(DocumentInfo(
                page=str(doc.metadata.get("page", "N/A")),
                link=doc.metadata.get("source", "N/A"),
                snippet=doc.page_content,
                probability=round(item["probability"], 4),
                cumulative_probability=round(item["cumulative_probability"], 4),
                distance=round(item["distance"], 4),
            ))

        context = "\n\n".join(item["document"].page_content for item in selected)
        formatted_prompt = prompt.format(context=context, question=query.question)
        response = llm.invoke(formatted_prompt)
        answer = answer_text(response.content if hasattr(response, "content") else response)

        return Response(
            answer=answer,
            retrieval=query.retrieval,
            k=query.k,
            p=query.p,
            retrieved_documents=doc_info,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)