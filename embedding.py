import re
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from pathlib import Path

PICKLE_FILE = "embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

app = FastAPI()
embedder = SentenceTransformer(MODEL_NAME)

class PathRequest(BaseModel):
    path: str

def clean_text(text: str) -> str:
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return clean_text(text)

def process_text_and_save(text: str):
    splitter = CharacterTextSplitter(separator=" ", chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    with open(PICKLE_FILE, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)
    return {"message": "Embeddings and chunks saved successfully", "chunks_count": len(chunks)}

@app.post("/upload")
async def upload_file(request: PathRequest):
    path = Path(request.path)
    if not path.exists() or not path.suffix == ".pdf":
        return {"error": "Invalid path or not a .pdf file."}
    text = extract_text_from_pdf(path)
    return process_text_and_save(text)