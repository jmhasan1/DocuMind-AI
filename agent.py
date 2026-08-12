# agent.py
# This script is the brain of your application. It implements a ReAct (Reasoning and Acting) orchestration loop using Groq's/ 
# ChatGPT's native function-calling. Instead of hardcoding when to use RAG or when to calculate, you give the LLM structural 
# choices and let it decide dynamically based on the user's intent.

import json
from groq import Groq
from langchain_openai import ChatOpenAI

from rag_core import rag_answer

# The tools list is a collection of JSON schemas that describe your local Python functions to the LLM. The LLM doesn't see 
# your actual code; it only sees these descriptions to understand what capabilities are at its disposal.

tools = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search uploaded documents to answer factual questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_doc",
            "description": "Summarize the entire document or a section",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"}
                },
                "required": ["topic"]
            }
        }
    }
]

    
import numexpr as ne
from rag_core import rag_answer

# A generic router function that takes a tool name string and its structured dictionary arguments and executes the corresponding Python function.
def run_tool(name, args):
    if name == "rag_search":
        return rag_answer(args["query"])
        
    elif name == "calculate":
        expression = args["expression"]
        try:
            # Safe evaluation using numexpr
            result = ne.evaluate(expression).item()
            return str(result)
        except Exception:
            # Fallback if the LLM hallucinated a non-math string into the tool
            return f"Error: '{expression}' is not a valid mathematical expression."
            
    elif name == "summarize_doc":
        return rag_answer(f"Summarize everything about: {args['topic']}")

# The Agentic Core Loop
# This function takes the incoming UI message alongside ongoing message records.
def agent_loop(user_message, chat_history):
    llm_gpt = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
    
    # Define a strict system prompt to anchor model formatting
    system_prompt = {
        "role": "system",
        "content": (
            "You are a precise AI Assistant equipped with specialized tools. "
            "When invoking a function tool, you MUST output ONLY the valid JSON object "
            "arguments matching the tool schema parameters. Never wrap your function "
            "arguments in extra text tags, XML tags, or markdown codeblocks."
        )
    }
    
    # Reinforce the system prompt at the very beginning of the thread arrays
    messages = [system_prompt] + chat_history + [{"role": "user", "content": user_message}]
    
    response = llm_gpt.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        tools=tools,            # Passes our dictionary of tool metadata to the LLM so it can reason about when to call them
        tool_choice="auto"      # grants our LLM the autonomy to decide when to call a tool based on the user's intent
    )
    
    msg = response.choices[0].message
    
    # If LLM called a tool
    if msg.tool_calls:                          # Evaluates whether the LLM decided it needed an external tool to fulfill the request. If yes, it stops text generation and returns a structural instruction inside tool_calls
        tool_call = msg.tool_calls[0]
        tool_result = run_tool(
            tool_call.function.name,            # Extracts the tool name string from the LLM's tool_call instruction
            json.loads(tool_call.function.arguments)        # Decodes the LLM-generated string arguments into a functional Python dictionary safely. Both parameters pass directly into our run_tool router.
        )
        
        # Feed tool result back and get final answer
        messages.append(msg)                            # Appends the LLM’s tool invocation request to the conversation tracker.
        messages.append({                               # Appends the output returned by our local Python function execution. We must supply the matching tool_call_id so the model knows which tool request this specific result satisfies
            "role": "tool", 
            "content": tool_result, 
            "tool_call_id": tool_call.id
        })

        # Sends the full conversation history—including the original prompt, the tool request, and the tool's execution result—back to Groq/ChatGPT for final answer generation.
        final = llm_gpt.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages
        )
        return final.choices[0].message.content
        
    return msg.content