
import gradio as gr
from agent import agent_loop
from ingest import load_pdf, chunk_text, collection, embed_model


def chat(message, history):
    """
    Cleans Gradio's internal history tracking by stripping out front-end 
    metadata before delivering the payload array to the Groq SDK.
    """
    clean_history = []
    
    for turn in history:
        # Extract only the strict schema keys that Groq permits
        if isinstance(turn, dict) and "role" in turn and "content" in turn:
            clean_history.append({
                "role": turn["role"],
                "content": turn["content"]
            })
            
    # Execute the core agent loop with the validated, sterile history payload
    response = agent_loop(message, clean_history)
    
    return response

def ingest_file(file):
    if file is None:
        return "⚠️ Please select a valid PDF file first!"
        
    text = load_pdf(file.name)
    chunks = chunk_text(text)
    embeddings = embed_model.encode(chunks).tolist()
    
    # Use explicit document hashes to completely eliminate indexing collisions
    collection.add(
        documents=chunks, 
        embeddings=embeddings,
        ids=[f"ui_{i}_{hash(chunks[i])}" for i in range(len(chunks))]
    )
    return f"✅ Successfully processed, vectorized, and indexed {len(chunks)} chunks into ChromaDB!"

# Building the frontend layout grid
with gr.Blocks(title="DocuMind AI") as demo:
    gr.Markdown("# 🧠 DocuMind AI — Agentic RAG Assistant")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="Upload Reference PDF", file_types=[".pdf"])
            ingest_btn = gr.Button("🚀 Ingest Document", variant="primary")
            status = gr.Textbox(label="Pipeline System Status", interactive=False)
            
            ingest_btn.click(ingest_file, inputs=file_upload, outputs=status)
            
        with gr.Column(scale=3):
            # Removed the type parameter to match your current Gradio package version
            gr.ChatInterface(fn=chat)

if __name__ == "__main__":
    demo.launch()