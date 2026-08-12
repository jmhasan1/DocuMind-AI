"""Agentic orchestration for DocuMind AI.

The agent uses LangChain's provider-neutral ChatModel interface, so the
same agent loop works with OpenAI ChatOpenAI or Groq ChatGroq.
"""
import numexpr as ne

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from llm_factory import get_llm
from rag_core import rag_answer


def _history_to_messages(chat_history):
    """Convert Gradio's role/content history into LangChain messages."""
    messages = []

    for turn in chat_history or []:
        if not isinstance(turn, dict):
            continue

        role = turn.get("role")
        content = turn.get("content")

        # Ignore non-text frontend payloads.
        if not isinstance(content, str) or not content.strip():
            continue

        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    return messages


def _build_tools(provider, chat_history):
    """Create request-scoped tools so concurrent users cannot change providers."""

    @tool
    def rag_search(query: str) -> str:
        """Search the uploaded documents and answer a factual question from them."""
        return rag_answer(
            query,
            chat_history=chat_history,
            provider=provider,
        )

    @tool
    def calculate(expression: str) -> str:
        """Safely evaluate a mathematical expression."""
        try:
            result = ne.evaluate(expression).item()
            return str(result)
        except Exception:
            return f"Error: '{expression}' is not a valid mathematical expression."

    @tool
    def summarize_doc(topic: str) -> str:
        """Summarize the uploaded document or a requested section/topic."""
        return rag_answer(
            f"Summarize everything about: {topic}",
            chat_history=chat_history,
            provider=provider,
        )

    return [rag_search, calculate, summarize_doc]


def agent_loop(user_message, chat_history=None, provider="groq"):
    """Run the agentic RAG loop using the selected LLM provider."""
    tools = _build_tools(provider, chat_history or [])

    llm = get_llm(provider)
    llm_with_tools = llm.bind_tools(
        tools,
        tool_choice="auto",
    )

    system_prompt = SystemMessage(
        content=(
            "You are DocuMind AI, a precise document-grounded assistant. "
            "Use rag_search for factual questions about uploaded documents. "
            "Use calculate for arithmetic. Use summarize_doc when the user "
            "asks for a document or section summary. You may call multiple "
            "tools when necessary. After receiving tool results, answer the "
            "user clearly and do not invent facts that are not supported by "
            "the retrieved document context."
        )
    )

    messages = [system_prompt]
    messages.extend(_history_to_messages(chat_history))
    messages.append(HumanMessage(content=user_message))

    # First model pass: decide whether a tool is required.
    response = llm_with_tools.invoke(messages)

    if not response.tool_calls:
        return response.content

    # Add the model's tool-call message to the conversation.
    messages.append(response)

    # Execute every requested tool. This also supports providers/models that
    # return multiple tool calls in one assistant message.
    tool_map = {tool.name: tool for tool in tools}

    for tool_call in response.tool_calls:
        name = tool_call["name"]
        args = tool_call.get("args", {})
        selected_tool = tool_map.get(name)

        if selected_tool is None:
            result = f"Error: unknown tool '{name}'."
        else:
            try:
                result = selected_tool.invoke(args)
            except Exception as exc:
                result = f"Tool '{name}' failed: {type(exc).__name__}: {exc}"

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )

    # Second pass: synthesize the final answer from tool results.
    final = llm.invoke(messages)
    return final.content