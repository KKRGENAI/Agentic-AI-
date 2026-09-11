import os
import re

from langchain_core.messages import ToolMessage

from src.agents.agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
    writer_prompt,
    critic_prompt,
)
from src.utils.llm_debug import (
    print_agent_conversation,
    print_agent_request,
    print_chain_request,
)


def _message_text(content) -> str:
    """Normalize Gemini/LangChain message content to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                parts.append(getattr(block, "text") or "")
        return "\n".join(p for p in parts if p)
    return str(content)


def _tool_outputs(messages, tool_name: str | None = None) -> str:
    """Collect ToolMessage contents (optionally filtered by tool name)."""
    matched = []
    all_tools = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        text = _message_text(msg.content)
        all_tools.append(text)
        if tool_name is None or getattr(msg, "name", None) in (None, tool_name):
            matched.append(text)
    chunks = matched if matched else all_tools
    return "\n----\n".join(chunks)


def _extract_urls(text: str, limit: int = 5) -> list[str]:
    """Pull unique http(s) URLs from text, in order of appearance."""
    if not text:
        return []
    found = re.findall(r"https?://[^\s\]\)\"'<>]+", text)
    cleaned = []
    seen = set()
    for url in found:
        url = url.rstrip(".,;:)")
        if url not in seen:
            seen.add(url)
            cleaned.append(url)
        if len(cleaned) >= limit:
            break
    return cleaned


def run_research_pipeline(topic: str, debug: bool | None = None) -> dict:

    if debug is None:
        debug = os.getenv("DEBUG_LLM", "").lower() in {"1", "true", "yes"}

    state = {}

    # search agent working
    print("\n" + " =" * 50)
    print("step 1 - search agent is working ...")
    print("=" * 50)

    search_user_input = f"Find recent, reliable and detailed information about: {topic}"
    if debug:
        print_agent_request(
            "Step 1 - Search Agent",
            user_input=search_user_input,
            system_prompt=(
                "You are a research search agent. "
                "You MUST call the web_search tool for every query. "
                "Do not answer from memory alone. "
                "In your final reply, list the best findings as bullet points and "
                "ALWAYS include the full Title and URL for every source you cite. "
                "Never drop or rewrite URLs."
            ),
        )

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", search_user_input)]
    })

    search_messages = search_result["messages"]
    if debug:
        print_agent_conversation("Step 1 - Search Agent", search_messages)
    # Prefer raw Tavily tool output so URLs are never dropped by the LLM summary
    raw_search = _tool_outputs(search_messages, tool_name="web_search")
    search_summary = _message_text(search_messages[-1].content)

    if raw_search.strip():
        state["search_results"] = (
            f"{raw_search}\n\n"
            f"Agent summary:\n{search_summary}"
        )
    else:
        state["search_results"] = search_summary

    urls = _extract_urls(state["search_results"])
    state["urls"] = urls

    print("\n search result ", state["search_results"])
    print("\n extracted URLs:", urls)

    # step 2 - reader agent
    print("\n" + " =" * 50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("=" * 50)

    reader_agent = build_reader_agent()
    if urls:
        url_block = "\n".join(f"- {u}" for u in urls[:3])
        reader_prompt = (
            f"Topic: {topic}\n\n"
            f"Scrape these URLs for deeper content (call scrape_url for each):\n"
            f"{url_block}\n\n"
            f"Full search context:\n{state['search_results']}"
        )
    else:
        reader_prompt = (
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results']}"
        )

    if debug:
        print_agent_request(
            "Step 2 - Reader Agent",
            user_input=reader_prompt,
            system_prompt=(
                "You are a research reader agent. "
                "You MUST call the scrape_url tool on the most relevant http/https URLs "
                "provided in the user message (up to 3 URLs). "
                "Do not invent URLs. If no URL is present, say so clearly. "
                "After scraping, briefly summarize the extracted content and keep the source URLs."
            ),
        )

    reader_result = reader_agent.invoke({
        "messages": [("user", reader_prompt)]
    })

    reader_messages = reader_result["messages"]
    if debug:
        print_agent_conversation("Step 2 - Reader Agent", reader_messages)
    raw_scrape = _tool_outputs(reader_messages, tool_name="scrape_url")
    reader_summary = _message_text(reader_messages[-1].content)

    if raw_scrape.strip():
        state["scraped_content"] = (
            f"{raw_scrape}\n\n"
            f"Reader summary:\n{reader_summary}"
        )
    else:
        state["scraped_content"] = reader_summary

    print("\nscraped content: \n", state["scraped_content"])

    # step 3 - writer chain
    print("\n" + " =" * 50)
    print("step 3 - Writer is drafting the report ...")
    print("=" * 50)

    sources_line = "\n".join(state.get("urls") or []) or "(none found)"
    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}\n\n"
        f"SOURCE URLS TO CITE:\n{sources_line}"
    )

    writer_inputs = {
        "topic": topic,
        "research": research_combined,
    }
    if debug:
        print_chain_request("Step 3 - Writer Chain", writer_prompt, writer_inputs)

    state["report"] = writer_chain.invoke(writer_inputs)

    print("\n Final Report\n", state["report"])

    # critic report
    print("\n" + " =" * 50)
    print("step 4 - critic is reviewing the report ")
    print("=" * 50)

    critic_inputs = {"report": state["report"]}
    if debug:
        print_chain_request("Step 4 - Critic Chain", critic_prompt, critic_inputs)

    state["feedback"] = critic_chain.invoke(critic_inputs)

    print("\n critic report \n", state["feedback"])

    return state
