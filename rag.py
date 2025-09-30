import os
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from huggingface_hub import InferenceClient

PICKLE_FILE = "embeddings.pkl"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"

HF_TOKEN = os.environ.get("HF_TOKEN", "")
client = InferenceClient(provider="novita", api_key=HF_TOKEN)

app = FastAPI()

# Load chunks
with open(PICKLE_FILE, "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]

# Build TF-IDF matrix
vectorizer = TfidfVectorizer()
chunk_embeddings = vectorizer.fit_transform(chunks)   # sparse matrix

def retrieve_context(query: str, top_k: int = 3):
    query_vec = vectorizer.transform([query])
    sims = (chunk_embeddings @ query_vec.T).toarray().ravel()
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_idx]

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_bot(req: QueryRequest):
    texts = retrieve_context(req.query)
    context = "\n".join(texts)

    messages = [
        {"role": "system", "content": "You are an edubot, that does not use bold texts. Use the following context to answer FAQs clearly and naturally."},
        {"role": "system", "content": context},
        {"role": "user", "content": req.query}
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        return {completion.choices[0].message["content"]}
    except Exception as e:
        return {"error": str(e)}