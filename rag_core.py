# rag_core.py
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Explicitly load the .env file
load_dotenv()


embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")
llm = Groq()  # set GROQ_API_KEY in env

def rag_answer(query, chat_history=[]):
    # 1. Embed the query
    query_embedding = embed_model.encode([query]).tolist()
    # 2. Retrieve top-3 chunks
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    context = "\n\n".join(results["documents"][0])
    # 3. Build prompt with memory
    messages = chat_history + [
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant", messages=messages
    )
    return response.choices[0].message.content