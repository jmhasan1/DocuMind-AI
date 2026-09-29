import gradio as gr #Gradio provides the web interface

from agent import agent_loop             #This connects the ChatInterface to the agent loop
from ingest import ingest_document       #This connects the upload UI to the document ingestion pipeline 


def chat(message, history, provider):               # This function is the bridge between the ChatInterface and the agent.
    """Gradio ChatInterface callback with provider selection."""
    clean_history = []                     # to create a normalized conversation-history structure.

    for turn in history or []:               # The code safely handles the possibility that: history == None
        if isinstance(turn, dict):           # Defensive Programming - Only process history entries if they are dictionary objects.
            role = turn.get("role")          # It extracts the role & content from each turn in the history.
            content = turn.get("content")

            if role in {"user", "assistant"} and isinstance(content, str):       # Filtering Valid Messages - only two rules: role must be either "user" or "assistant", and content must be a string. Anything else is ignored.
                clean_history.append({"role": role, "content": content})

    try:
        # Calling the Agent
        return agent_loop(                  # can be considered as the API boundary inside the application.
            user_message=message,           # app.py provides the user message, the cleaned chat history, and the selected provider to the agent loop.The agent provides:final answer.
            chat_history=clean_history,
            provider=provider,              # UI allows user to select between different LLM providers (Groq or OpenAI) for the agent loop.
        )
    except Exception as exc:
        return f"⚠️ {type(exc).__name__}: {exc}"        # For production, I would not expose raw exception messages to users.


# Responsibilty - take the uploaded PDF and send it into the ingestion pipeline.
def ingest_file(file):
    if file is None:
        return "⚠️ Please select a valid PDF file first!"

    try:
        report = ingest_document(file.name)             # app.py only knows app.py only knows.It doesn't know what happens internally.
    except Exception as exc:
        return f"⚠️ {type(exc).__name__}: {exc}"        #This protects the UI from ingestion failures.

    if report["status"] == "empty":                     # Handling an Empty PDF.
        return "⚠️ No extractable text was found in the PDF."

    return (                
        f"✅ Indexed {report['chunks_indexed']} chunks from "
        f"{report['filename']} ({report['pages_with_text']} text pages). "
        f"Document ID: {report['document_id']}"
    )
    # Succesful Ingeston Response - The ingestion layer returns a structured report. 
    # app.py converts that structured backend result into a human-readable UI message

# Building the Gradio Application
with gr.Blocks(title="DocuMind AI") as demo:        # This creates the main Gradio application.
    gr.Markdown("# 🧠 DocuMind AI — Agentic RAG Assistant")

    with gr.Row():              # creates a horizontal layout.
        with gr.Column(scale=1): # creates a vertical layout for the left side of the UI.The scale values control the relative width.
            provider = gr.Dropdown(                 # This creates LLM Provider selection dropdown.
                choices=["groq", "openai"],
                value="groq",
                label="LLM Provider",
                info="Switch between Groq and OpenAI without changing the agent code.",
            )

            file_upload = gr.File(                  # This restricts the UI to PDF files.
                label="Upload Reference PDF",
                file_types=[".pdf"],
            )
            ingest_btn = gr.Button("🚀 Ingest Document", variant="primary")         # This is the user's explicit trigger for ingestion.(Upload ≠ ingestion)
            status = gr.Textbox(
                label="Pipeline System Status",         # This is read-only from the user's perspective.Status Textbox.
                interactive=False,
            )

            ingest_btn.click(                  # When ingest_btn is clicked, call ingest_file() using file_upload as input and put its return value into status.
                ingest_file,                   # This is effectively an event-driven callback.
                inputs=file_upload,
                outputs=status,
            )

        with gr.Column(scale=3):
            gr.ChatInterface(                   # This is the core conversational UI.
                fn=chat,                        # It means : User submits message --> Gradio calls chat().
                additional_inputs=[provider],   # Normally the chat callback receives the conversation information. But the callback also excepts provider, because its signature is:chat(message, history, provider).
                title="Chat with your documents",
                description="Choose Groq or OpenAI, then ask questions about your indexed PDFs.",
            )


if __name__ == "__main__":
    demo.launch()