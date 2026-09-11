"""
Run this to SEE what gets sent to the LLM — without calling the API.

Usage:
    python debug_prompts.py
"""

from src.agents.agents import writer_prompt, critic_prompt
from src.utils.llm_debug import print_chain_request, print_agent_request

TOPIC = "The impact of AI on the job market in 2026"

# Example payload similar to what the pipeline builds after search + scrape
SAMPLE_RESEARCH = """SEARCH RESULTS:
Title: Example Study
URL: https://example.com/ai-jobs
Snippet: AI is changing entry-level hiring patterns.

DETAILED SCRAPED CONTENT:
AI adoption varies by industry. Entry-level roles in routine tasks are most exposed.

SOURCE URLS TO CITE:
https://example.com/ai-jobs
"""

SAMPLE_REPORT = "## Sample Report\nAI is affecting entry-level jobs..."

print("\n" + "#" * 70)
print("# ARCHITECTURE IN THIS PROJECT")
print("#" * 70)
print("""
Step 1: Search AGENT  (tool: web_search)   -> LangGraph agent loop
Step 2: Reader AGENT  (tool: scrape_url)   -> LangGraph agent loop
Step 3: Writer CHAIN  (prompt | llm)       -> single LLM call
Step 4: Critic CHAIN  (prompt | llm)       -> single LLM call

Agents and chains are separate steps in pipeline.py (not nested inside each other).
Chains are NOT between the two agents — they run after both agents finish.
""")

print_agent_request(
    "Step 1 - Search Agent",
    user_input=f"Find recent, reliable and detailed information about: {TOPIC}",
    system_prompt=(
        "You are a research search agent. "
        "You MUST call the web_search tool for every query..."
    ),
)

print_agent_request(
    "Step 2 - Reader Agent",
    user_input=(
        f"Topic: {TOPIC}\n\n"
        "Scrape these URLs for deeper content (call scrape_url for each):\n"
        "- https://example.com/ai-jobs\n\n"
        f"Full search context:\n{SAMPLE_RESEARCH}"
    ),
    system_prompt=(
        "You are a research reader agent. "
        "You MUST call the scrape_url tool on the most relevant http/https URLs..."
    ),
)

print_chain_request(
    "Step 3 - Writer Chain",
    writer_prompt,
    {"topic": TOPIC, "research": SAMPLE_RESEARCH},
)

print_chain_request(
    "Step 4 - Critic Chain",
    critic_prompt,
    {"report": SAMPLE_REPORT},
)

print("Done. These are the exact message shapes sent to Gemini.")
print("Run with DEBUG_LLM=1 to print this during the full pipeline:\n")
print("  DEBUG_LLM=1 python main.py")
