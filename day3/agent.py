"""
Assignment 3 – LLM-Based Agent
Uses the new Google GenAI SDK to decide which tool to call.
Falls back to a simulated keyword router if an error occurs.
"""

import re
import json
import datetime
# Assuming tools.py exists in the same directory and exports a dictionary called TOOLS
from tools import TOOLS

# ── LLM System Prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a routing agent. Given a user query, decide which tool to call.
Available tools: calculator, weather, summarizer, none.
Reply ONLY with a JSON object: {"tool": "<tool_name>", "argument": "<extracted_arg>"}
- calculator: math expressions (argument = the expression)
- weather: city name lookup (argument = city name only)
- summarizer: text to summarise (argument = the raw text)
- none: greetings or unknown requests (argument = "")
No preamble. JSON only."""


def llm_decide(query: str) -> dict:
    """Call Gemini via the new Google GenAI API to decide tool + argument."""
    try:
        from google import genai
        from google.genai import types
        
        # ⚠️ Hardcoded key as requested. Keep this safe!
        api_key = "AIzaSyBOtj8holygdCdSUfDxxu1Qb-mRIGtcbd0"
        
        # Initialize the new client
        client = genai.Client(api_key=api_key)

        # Generate the response using gemini-2.0-flash and strictly enforce JSON
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                max_output_tokens=200,
            )
        )
        
        raw = response.text.strip()
        return json.loads(raw)
        
    except ImportError:
        print("[INFO] 'google-genai' package not installed — using simulated LLM.")
        return _simulated_llm(query)
    except Exception as e:
        print(f"[WARN] LLM error: {e} — using simulated LLM.")
        return _simulated_llm(query)


def _simulated_llm(query: str) -> dict:
    """Keyword-based fallback when the API call fails."""
    q = query.lower()
    
    # Calculator Fallback: Extract ONLY the math, ignore words
    if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", q) or any(w in q for w in ["calculate", "compute"]):
        # The regex now forces the match to START with a digit (\d)
        match = re.search(r"\d+[\d\s\+\-\*\/\(\)\.]+", q)
        expr = match.group(0).strip() if match else "0"
        return {"tool": "calculator", "argument": expr}
        
    # Weather Fallback
    if "weather" in q:
        city = re.sub(r"(?i)weather\s*(in|for|at)?\s*", "", query).strip()
        return {"tool": "weather", "argument": city}
        
    # Summarizer Fallback
    if any(w in q for w in ["summarize", "summarise", "summary"]):
        arg = re.sub(r"(?i)^(summarize|summarise|summary of)\s*", "", query).strip()
        return {"tool": "summarizer", "argument": arg}
        
    return {"tool": "none", "argument": ""}

# ── Logger ────────────────────────────────────────────────────────────────────

LOG_FILE = "agent_log.txt"


def log(entry: dict):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Main Loop ─────────────────────────────────────────────────────────────────

def run_agent():
    print("=== Gemini LLM-Based Agent ===")
    print(f"Logs → {LOG_FILE}")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Agent: Goodbye!")
            break

        decision  = llm_decide(user_input)
        tool_name = decision.get("tool", "none")
        argument  = decision.get("argument", user_input)

        if tool_name in TOOLS:
            try:
                result = TOOLS[tool_name](argument)
            except Exception as e:
                result = f"Tool execution failed: {e}"
        else:
            result = "I'm not sure how to help with that."

        log({
            "timestamp": datetime.datetime.now().isoformat(),
            "input":     user_input,
            "tool":      tool_name,
            "argument":  argument,
            "output":    str(result),
        })

        print(f"Agent [{tool_name}]: {result}\n")


if __name__ == "__main__":
    run_agent()