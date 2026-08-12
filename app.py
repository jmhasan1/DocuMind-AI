import gradio as gr

from agent import agent_loop
from ingest import load_pdf, chunk_text, collection, embed_model


def chat(message, history, provider):
    """Gradio ChatInterface callback with provider selection."""
    clean_history = []

    for turn in history or []:
        if isinstance(turn, dict):
            role = turn.get("role")
            content = turn.get("content")

            if role in {"user", "assistant"} and isinstance(content, str):
                clean_history.append(
                    {"role": role, "content": content}
                )

    try:
        return agent_loop(
            user_message=message,
            chat_history=clean_history,
            provider=provider,
        )
    except Exception as exc:
        # Show an actionable error in the UI instead of a long Gradio traceback.
        return f"⚠️ {type(exc).__name__}: {exc}"


def ingest_file(file):
    if file is None:
        return "⚠️ Please select a valid PDF file first!"

    text = load_pdf(file.name)
    chunks = chunk_text(text)

    if not chunks:
        return "⚠️ No extractable text was found in the PDF."

    embeddings = embed_model.encode(chunks).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"ui_{i}_{hash(chunks[i])}" for i in range(len(chunks))],
    )

    return (
        f"✅ Successfully processed, vectorized, and indexed "
        f"{len(chunks)} chunks into ChromaDB!"
    )


with gr.Blocks(title="DocuMind AI") as demo:
    gr.Markdown("# 🧠 DocuMind AI — Agentic RAG Assistant")

    with gr.Row():
        with gr.Column(scale=1):
            provider = gr.Dropdown(
                choices=["groq", "openai"],
                value="groq",
                label="LLM Provider",
                info="Switch between Groq and OpenAI without changing the agent code.",
            )

            file_upload = gr.File(
                label="Upload Reference PDF",
                file_types=[".pdf"],
            )
            ingest_btn = gr.Button("🚀 Ingest Document", variant="primary")
            status = gr.Textbox(
                label="Pipeline System Status",
                interactive=False,
            )

            ingest_btn.click(
                ingest_file,
                inputs=file_upload,
                outputs=status,
            )

        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=chat,
                additional_inputs=[provider],
                title="Chat with your documents",
                description="Choose Groq or OpenAI, then ask questions about your indexed PDFs.",
            )


if __name__ == "__main__":
    demo.launch()