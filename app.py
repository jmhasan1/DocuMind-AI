# Gradio app for DocuMind AI
# app.py
import gradio as gr
from agent import agent_loop
from ingest import load_pdf, chunk_text, collection, embed_model

chat_history = []

def chat(message, history):
    global chat_history
    response = agent_loop(message, chat_history)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})
    return response

def ingest_file(file):
    text = load_pdf(file.name)
    chunks = chunk_text(text)
    embeddings = embed_model.encode(chunks).tolist()
    collection.add(documents=chunks, embeddings=embeddings,
                   ids=[f"c_{i}" for i in range(len(chunks))])
    return f"✅ Ingested {len(chunks)} chunks"

with gr.Blocks(title="DocuMind AI") as demo:
    gr.Markdown("## DocuMind AI — Agentic RAG Assistant")
    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="Upload PDF")
            ingest_btn = gr.Button("Ingest Document")
            status = gr.Textbox(label="Status")
            ingest_btn.click(ingest_file, file_upload, status)
        with gr.Column(scale=3):
            gr.ChatInterface(chat)

demo.launch()