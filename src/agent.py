"""
EconSafeBench LangGraph Agent Wrapper
======================================
Provides a unified interface to run any of the 8 evaluated models
as a ReAct agent with custom economic tools.

Usage:
    state = SceneDState()
    tools = make_scene_d_tools(state)
    response = run_agent(
        task_prompt = case["task_prompt"],
        model_alias = "gpt-5.4",
        tools       = tools,
        seed        = 42,
    )
    result = judge_scene_d(dim, state, case)
"""

from __future__ import annotations
import os
from typing import Any
from langchain_core.messages import HumanMessage

# ── Model registry ─────────────────────────────────────────────────────────────
# Models are routed through an OpenAI-compatible endpoint configured by
# OPENAI_API_KEY and optionally OPENAI_BASE_URL.

MODEL_REGISTRY: dict[str, str] = {
    "gpt-5.4":          "gpt-5.4",
    "gpt-5.4-mini":     "gpt-5.4-mini",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "o3-mini":          "o3-mini",
    "deepseek-v3.2":    "deepseek-v3.2",
    "deepseek-r1":      "deepseek-r1",   # excluded from analysis (no function calling)
    "qwen3-max":        "qwen3-max",
    "glm-5":            "glm-5",          # GLM-5 (Zhipu AI / Tsinghua)
    "qwen2.5-7b":       "qwen2.5-7b",   # local vLLM, --served-model
}

DEFAULT_MODELS = list(MODEL_REGISTRY.keys())

# ── Locally-hosted models (vLLM or similar OpenAI-compatible server) ──────────
# Models listed here override the global OPENAI_API_KEY / OPENAI_BASE_URL so
# that API-based and locally-served models can be mixed in the same run
# without switching env vars between calls. Add new local models here only —
# nothing else in this file needs to change.
LOCAL_MODEL_ENDPOINTS: dict[str, str] = {
    "qwen2.5-7b": "http://localhost:8000/v1",
}
LOCAL_MODEL_API_KEY = "local-vllm-no-auth"  # vLLM does not validate this by default


def _build_llm(model_alias: str, seed: int = 0, temperature: float = 0.0):
    """Build a LangChain LLM instance for the given model alias."""
    model_str = MODEL_REGISTRY.get(model_alias, model_alias)

    # All models go through the configured OpenAI-compatible endpoint.
    from langchain_openai import ChatOpenAI
    # o3-mini is a reasoning model: temperature=0 causes unstable tool-calling;
    # use temperature=1 (the recommended default for o-series models).
    # DeepSeek-R1 does not support function calling at all — excluded upstream.
    effective_temp = 1.0 if model_alias == "o3-mini" else temperature

    kwargs: dict[str, Any] = dict(
        model=model_str,
        temperature=effective_temp,
        api_key=(LOCAL_MODEL_API_KEY if model_alias in LOCAL_MODEL_ENDPOINTS
                 else os.environ["OPENAI_API_KEY"]),
        base_url=LOCAL_MODEL_ENDPOINTS.get(model_alias, os.environ.get("OPENAI_BASE_URL")),
        max_tokens=1500,
        # Hard per-request timeout so a stalled HTTP call raises instead of
        # hanging the whole run forever. Tune via LG_REQUEST_TIMEOUT.
        timeout=float(os.environ.get("LG_REQUEST_TIMEOUT", "90")),
        # Disable the SDK's own retry/backoff — run_agent() governs retries,
        # otherwise the two stack and a dead endpoint hangs for minutes.
        max_retries=0,
    )
    # seed supported by GPT/o-series only
    if model_alias not in ("deepseek-v3.2", "deepseek-r1",
                           "gemini-2.5-flash", "qwen2.5-7b", "qwen3-max"):
        kwargs["extra_body"] = {"seed": seed}

    # Qwen3 defaults to thinking mode on some providers; disable it for
    # non-streaming calls when the endpoint supports this flag.
    if model_alias == "qwen3-max":
        kwargs["extra_body"] = {"enable_thinking": False}

    return ChatOpenAI(**kwargs)

def run_agent(
    task_prompt: str,
    model_alias: str,
    tools: list,
    seed: int = 0,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> str:
    """
    Run a ReAct agent for one eval case.

    The agent receives task_prompt as a HumanMessage and is expected to
    call one or more tools to complete the economic task. Tool call
    parameters are recorded in the shared state object and used by the
    judgment layer.

    Returns the agent's final text response (last AI message content).
    """
    import time
    from langgraph.prebuilt import create_react_agent
    from langgraph.errors import GraphRecursionError

    llm = _build_llm(model_alias, seed=seed, temperature=temperature)
    agent = create_react_agent(llm, tools)

    # Cap the ReAct loop. Without this it defaults to 25 steps; a model whose
    # tool-call signalling the proxy mishandles (e.g. gemini via OpenAI-compat)
    # can spin to the limit per period — slow and burns huge quota. One real
    # decision needs ~1–2 model calls, so a small cap is plenty.
    # gemini-2.5-flash needs more steps to complete tool calls in complex prompts
    # (Scene A RV+DC combined). Other models are fine at 8.
    default_limit = int(os.environ.get("LG_RECURSION_LIMIT", "8"))
    recursion_limit = 12 if model_alias == "gemini-2.5-flash" else default_limit
    invoke_cfg = {"recursion_limit": recursion_limit}

    # Small delay to avoid rate limiting on aggregation APIs
    time.sleep(0.5)

    for attempt in range(max_retries):
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=task_prompt)]},
                config=invoke_cfg,
            )
            # Extract last AI message
            messages = result.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    # Skip tool messages
                    if msg.__class__.__name__ not in ("ToolMessage",):
                        return str(msg.content)
            return ""
        except GraphRecursionError as exc:
            # Looping past the cap won't fix itself on retry — fail fast so the
            # period is excluded (never silently scored) by the caller.
            print(f"    [recursion-limit {recursion_limit} hit] {exc!r} — not retrying")
            return "__ERROR__: recursion limit exceeded"
        except Exception as exc:
            wait = 2 ** attempt
            print(f"    [retry {attempt+1}/{max_retries}] {exc!r} — wait {wait}s")
            time.sleep(wait)

    return "__ERROR__: max retries exceeded"


def run_agent_dry(task_prompt: str, tools: list) -> str:
    """Dry-run: invoke tools with dummy values without calling the API."""
    # Return a canned response that mentions the task
    return f"[dry run] Task received: {task_prompt[:80]}..."
