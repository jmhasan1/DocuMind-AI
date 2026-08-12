# ingest.py
import fitz                                                     # PyMuPDF # programmatically read and parse PDF structures.
import chromadb                                                 # Imports our local vector database engine.
from sentence_transformers import SentenceTransformer           # Imports the specialized class that downloads and manages pre-trained AI embedding models.

def load_pdf(path):                                     # Defines a function that accepts the string file path of your PDF.
    doc = fitz.open(path)                               # PyMuPDF Opens the PDF file and creates a document object that allows us to programmatically access its contents.(pages, metadata, and images.)
    return " ".join([page.get_text() for page in doc])  # Returns the concatenated text from all pages (Glues all those individual page strings together into one massive, continuous string, separated by spaces, and returns it to the caller).

# We are chunking , as LLMs have context window limitations. We will break the text into smaller chunks to ensure that we can process it effectively.
def chunk_text(text, chunk_size=500, overlap=50):       
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

# Initializing Models and Databases
embed_model = SentenceTransformer("all-MiniLM-L6-v2")       # This model turns text into a 384-dimensional vector array. It's tiny (~90MB) so can run on low ower GPUs. It is also very fast and has good semantic understanding. It is a great choice for semantic search and retrieval tasks.
client = chromadb.PersistentClient(path="./chroma_db")      # Tells ChromaDB to initialize a database instance that saves everything to your local hard drive in a folder called chroma_db.
collection = client.get_or_create_collection("documents")   # Creates a collection (like a table in a relational database) called "documents" where we will store our text chunks and their corresponding embeddings.

# Run once to ingest your PDFs
text = load_pdf("data/your_document.pdf")
chunks = chunk_text(text)
embeddings = embed_model.encode(chunks).tolist()            # This passes your text chunks to the sentence-transformer AI model. The model mathematically reads the context of the words and yields numerical vector arrays representing their meaning. .tolist() converts the resulting NumPy array into standard Python lists so ChromaDB can digest it.
# This uploads your processed data into your local database table.
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]          # Creates unique IDs for each chunk so we can reference them later.  
)