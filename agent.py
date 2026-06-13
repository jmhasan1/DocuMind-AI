# agent.py
import json
from groq import Groq
from rag_core import rag_answer

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


def agent_loop(user_message, chat_history):
    llm = Groq()
    
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
    
    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    msg = response.choices[0].message
    
    # If LLM called a tool
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        tool_result = run_tool(
            tool_call.function.name,
            json.loads(tool_call.function.arguments)
        )
        
        # Feed tool result back and get final answer
        messages.append(msg)
        messages.append({
            "role": "tool", 
            "content": tool_result, 
            "tool_call_id": tool_call.id
        })
        
        final = llm.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=messages
        )
        return final.choices[0].message.content
        
    return msg.content