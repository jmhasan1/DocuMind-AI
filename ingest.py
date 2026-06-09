# ingest.py
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer

def load_pdf(path):
    doc = fitz.open(path)
    return " ".join([page.get_text() for page in doc])

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")

# Run once to ingest your PDFs
text = load_pdf("your_document.pdf")
chunks = chunk_text(text)
embeddings = model.encode(chunks).tolist()
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)