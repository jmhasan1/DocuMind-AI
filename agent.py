"""Agentic orchestration for DocuMind AI.

The agent uses LangChain's provider-neutral ChatModel interface, so the
same agent loop works with OpenAI ChatOpenAI or Groq ChatGroq.
"""
import numexpr as ne                    # numexpr powers the calculator tool.

from langchain_core.messages import (       # These are the four message types used by the agent.
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool           # This converts ordinary Python functions into LLM-callable tools.

from llm_factory import get_llm                 # agent.py delegates LLM creation → llm_factory.py
from rag_core import rag_answer                 # agent.py delegates Document reasoning/retrieval → rag_core.py

def _history_to_messages(chat_history):         # First helper function -- Its job is to convert the history coming from Gradio into LangChain message objects.
    """Convert Gradio's role/content history into LangChain messages."""
    messages = []                               # Initilizie output.

    for turn in chat_history or []:             # Iterate through history.Also, or [] - missing history doesn't crash the application.
        if not isinstance(turn, dict):          # Invalid frontend objects are ignored.
            continue

        role = turn.get("role")                 # The code extracts role & content.
        content = turn.get("content")

        # Ignore non-text frontend payloads. (This prevents things like:None,empty strings, non-text frontend payloads from reaching the LLM)
        if not isinstance(content, str) or not content.strip():
            continue

        # This creates a clean boundary between UI representation & LLM representation.                    
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    return messages

# The Tool Factory -- this function creates the tools available to the agent.
# Also note - The agent.py handles tool selection, while rag_core.py handles RAG execution.
def _build_tools(provider, chat_history):
    """Create request-scoped tools so concurrent users cannot change providers."""
    # Above doctring means tools are created for each request, rather than being global objects.This is a good concurrency-aware design.
    
    # The LLM uses the tool name and description to understand when it should call it.
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
            result = ne.evaluate(expression).item()             # numexpr Instead of eval() -- because arbitrary Python eval() can execute dangerous code.
            return str(result)
        except Exception:
            return f"Error: '{expression}' is not a valid mathematical expression."
        # error handling is important because tool failures become observations that the LLM can reason about.

    # Summarization Reuses RAG.This is a retrieval-grounded summarization strategy.
    # That's a reasonable approach for large documents because we don't necessarily want to feed the entire PDF to the LLM.
    @tool
    def summarize_doc(topic: str) -> str:
        """Summarize the uploaded document or a requested section/topic."""
        return rag_answer(
            f"Summarize everything about: {topic}",
            chat_history=chat_history,
            provider=provider,
        )

    return [rag_search, calculate, summarize_doc]


# This is the main orchestration function called by app.py.
def agent_loop(user_message, chat_history=None, provider="groq"):
    """Run the agentic RAG loop using the selected LLM provider."""
    tools = _build_tools(provider, chat_history or [])

    llm = get_llm(provider)             # This delegates model creation to llm_factory.py.
    llm_with_tools = llm.bind_tools(    # Binds the tools.The model can now produce structured tool calls.
        tools,
        tool_choice="auto",             # Let the model decide whether it needs a tool.
    )

    # This is effectively the policy layer of the agent.
    # It defines: WHEN to use RAG ; WHEN to calculate; WHEN to summarize; HOW to behave after tool execution ; WHAT not to hallucinate
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

    # Conversational reasoning.Intially System Reasoning, then Conversation history is inserted., then Human Message.
    messages = [system_prompt]
    messages.extend(_history_to_messages(chat_history))
    messages.append(HumanMessage(content=user_message))

    # First model pass: decide whether a tool is required.
    response = llm_with_tools.invoke(messages)

    if not response.tool_calls:             # If no tools are reqd.This avoids unnecessary retrieval and another LLM call.
        return response.content

    # Add the model's tool-call message to the conversation.
    messages.append(response)                   # The conversation now records: User -> AI -> "I want to call rag_search with query X" (or other tool)

    # Execute every requested tool. This also supports providers/models that
    # return multiple tool calls in one assistant message.
    tool_map = {tool.name: tool for tool in tools}

    for tool_call in response.tool_calls:       # tool_calls originates from the LangChain AIMessage returned by the chat model after tool binding.
        name = tool_call["name"]
        args = tool_call.get("args", {})
        selected_tool = tool_map.get(name)

        #  Unknown Tool Handling : If the model requests something unexpected.
        if selected_tool is None: 
            result = f"Error: unknown tool '{name}'."
        else:
            try:
                result = selected_tool.invoke(args) # Executing the Tool
            except Exception as exc:
                result = f"Tool '{name}' failed: {type(exc).__name__}: {exc}" 
            # Tool Failure Handling - this prevents one failed tool from crashing the entire agent.

        messages.append(
            ToolMessage(                # Very Important - This is how the result is returned to the LLM.
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )

    # Second pass: synthesize the final answer from tool results.
    final = llm.invoke(messages)
    return final.content
    # This is where the final natural-language answer is generated.


    #  mportant Architectural Detail: This Is a Two-Pass Agent. The current implementation is not an unlimited iterative agent loop.