# rag_core.py
from groq import Groq
from langchain_openai import ChatOpenAI
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Explicitly load the .env file
load_dotenv()


embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")
llm = Groq()  # set GROQ_API_KEY in env
llm_gpt=ChatOpenAI(model="gpt-4o-mini-2024-07-18")

# Processing the Query and Vector Similarity Search
# This function accepts two inputs: the active query from the user, and an optional chat_history list containing prior turns of the conversation to handle memory tracking.
def rag_answer(query, chat_history=[]):
    # 1. Embed the query
    query_embedding = embed_model.encode([query]).tolist()          # Turns the incoming plain-text question into a 384-dimensional dense vector array. To find relevant documents, we must search vectors using vectors
    # 2. Retrieve top-3 chunks
    results = collection.query(query_embeddings=query_embedding, n_results=3)       # ChromaDB calculates the cosine distance between your question's vector and every chunk vector sitting in your local database. It instantly returns the top 3 closest, most semantically relevant text blocks
    context = "\n\n".join(results["documents"][0])
    # 3. Build prompt with memory (Prompt Construction with Memory Insertion)
    messages = chat_history + [                                                             # This constructs a unified conversation layout for the model. If chat_history holds previous exchanges, they are appended first.
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}            # We inject the extracted raw document pieces explicitly into the text string. The LLM doesn't need to guess or hallucinate—the source reference answers are pinned straight to the prompt input
    ]
    response = llm_gpt.chat.completions.create(
        model="gpt-4o-mini-2024-07-18", messages=messages
    )
    return response.choices[0].message.content                                              # Extracts the generated clean string reply from the JSON payload returned by Groq's/ChatGPT's API response.

    
