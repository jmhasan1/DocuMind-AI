
import gradio as gr
from agent import agent_loop               # Pulls in your ReAct decision-making loop that controls tool invocation and Groq/ChatGPT API calls.
from ingest import load_pdf, chunk_text, collection, embed_model                # Imports your document parser, chunk splitter, active ChromaDB collection instance, and your local text embedder model straight into the application server context.


# Defines the function triggered whenever a user sends a message in the chat UI. It receives the user's latest input (message) and previous chat history (history)
def chat(message, history):
    """
    Cleans Gradio's internal history tracking by stripping out front-end 
    metadata before delivering the payload array to the Groq SDK.
    """
    clean_history = []                  # Initializes an empty list to store sanitized history data.
    
    for turn in history:            # Loopig through each turn in the chathistory.
        # Extract only the strict schema keys that Groq permits
        if isinstance(turn, dict) and "role" in turn and "content" in turn: # Checks whether the current turn is a dictionary and contains standard schema keys ("role" and "content").
            clean_history.append({                                          # Extracts only the required keys, removing extra frontend metadata so the payload is compatible with the target API (e.g., Groq).
                "role": turn["role"],
                "content": turn["content"]
            })
            
    # Execute the core agent loop with the validated, sterile history payload
    response = agent_loop(message, clean_history)                           # Passes the user's input and sanitized message history to the main agent loop to generate an answer.
    
    return response

def ingest_file(file):
    if file is None:                    # Checks if the user clicked the ingestion button without selecting a file.
        return "⚠️ Please select a valid PDF file first!"
        
    text = load_pdf(file.name)             # Extracts raw text from the uploaded PDF using its file system path.
    chunks = chunk_text(text)              # Splits the raw text into smaller text passages.(chunks)
    embeddings = embed_model.encode(chunks).tolist()            # Generates vector embeddings for each chunk and converts the output to a standard Python list format.
    
    # Use explicit document hashes to completely eliminate indexing collisions
    collection.add(                 # Calls ChromaDB to insert documents into the vector collection.
        documents=chunks, 
        embeddings=embeddings,
        ids=[f"ui_{i}_{hash(chunks[i])}" for i in range(len(chunks))]   # Generates unique string IDs for each chunk using index positions and string hashing to prevent key collisions.
    )
    return f"✅ Successfully processed, vectorized, and indexed {len(chunks)} chunks into ChromaDB!"


# Building the frontend layout grid
with gr.Blocks(title="DocuMind AI") as demo:                        # Initializes a Gradio layout object named demo with the browser title set to "DocuMind AI".
    gr.Markdown("# 🧠 DocuMind AI — Agentic RAG Assistant")
    
    with gr.Row():                                          # Creates a horizontal row container to split elements side-by-side.
        with gr.Column(scale=1):                            # Creates a left-hand column spanning 1 unit of available grid width.
            file_upload = gr.File(label="Upload Reference PDF", file_types=[".pdf"])
            ingest_btn = gr.Button("🚀 Ingest Document", variant="primary")
            status = gr.Textbox(label="Pipeline System Status", interactive=False)
            
            ingest_btn.click(ingest_file, inputs=file_upload, outputs=status)
            
        with gr.Column(scale=3):
            # Removed the type parameter to match your current Gradio package version
            gr.ChatInterface(fn=chat)

if __name__ == "__main__":
    demo.launch()