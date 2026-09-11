from __future__ import annotations

from langchain_core.messages import BaseMessage


def _message_text(content) -> str:
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


def print_chain_request(step_name: str, prompt_template, inputs: dict) -> None:
    """Show the exact messages a chain sends to the LLM (after placeholders are filled)."""
    messages = prompt_template.invoke(inputs).to_messages()

    print("\n" + "=" * 70)
    print(f"LLM REQUEST -> {step_name} (chain)")
    print("=" * 70)
    print(f"Input variables: {list(inputs.keys())}")
    print("-" * 70)

    for i, msg in enumerate(messages, start=1):
        role = getattr(msg, "type", msg.__class__.__name__)
        print(f"\n[{i}] {role.upper()}")
        print("-" * 40)
        print(_message_text(msg.content))

    print("\n" + "=" * 70 + "\n")


def print_agent_request(step_name: str, user_input: str, system_prompt: str | None = None) -> None:
    """Show what we send when an agent is first invoked."""
    print("\n" + "=" * 70)
    print(f"LLM REQUEST -> {step_name} (agent - initial turn)")
    print("=" * 70)
    print("-" * 70)

    if system_prompt:
        print("\n[1] SYSTEM")
        print("-" * 40)
        print(system_prompt)

    print(f"\n[{2 if system_prompt else 1}] HUMAN")
    print("-" * 40)
    print(user_input)
    print("\n" + "=" * 70 + "\n")


def print_agent_conversation(step_name: str, messages) -> None:
    """Show the full agent message history (every LLM round-trip)."""
    print("\n" + "=" * 70)
    print(f"LLM CONVERSATION -> {step_name} (agent - full history)")
    print("=" * 70)

    for i, msg in enumerate(messages, start=1):
        role = getattr(msg, "type", msg.__class__.__name__)
        print(f"\n[{i}] {role.upper()}")
        print("-" * 40)

        content = _message_text(msg.content)
        if content:
            # Keep output readable in terminal
            preview = content if len(content) <= 2500 else content[:2500] + "\n... [truncated]"
            print(preview)

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            print("\nTool calls:")
            for call in tool_calls:
                print(f"  - {call.get('name')}: {call.get('args')}")

    print("\n" + "=" * 70 + "\n")
