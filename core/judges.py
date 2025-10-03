"""Judge helper functions extracted from the Streamlit app."""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st
from openai import AsyncOpenAI

from core.api_clients import openai_structured_json, openrouter_json
from core.pricing import _to_openrouter_model_id, _to_native_model_id, _get_provider_from_model_id

_CONFIG: Dict[str, Any] = {}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "final_choice_model": {"type": "string", "description": "One of: mistral, gpt5, third"},
        "final_label": {"type": "string"},
        "judge_rationale": {"type": "string"},
    },
    "required": ["final_choice_model", "final_label", "judge_rationale"],
    "additionalProperties": False,
}

JUDGE_INSTRUCTIONS = (
    "You are a neutral judge... Return ONLY JSON with: final_choice_model, final_label, judge_rationale."
)

PRUNER_SCHEMA = {
    "type": "object",
    "properties": {
        "kept_context_keys": {
            "type": "array",
            "description": "An array of essential context keys to keep.",
            "items": {
                "type": "string",
                "enum": ["instruction", "summary", "user_messages", "agent_responses", "tool_logs"],
            },
        },
        "action": {
            "type": "string",
            "enum": ["general_answer", "kb_lookup", "tool_call"],
            "description": "The best next action to take.",
        },
        "prune_rationale": {
            "type": "string",
            "description": "Briefly explain why you chose these keys and this action.",
        },
    },
    "required": ["kept_context_keys", "action", "prune_rationale"],
    "additionalProperties": False,
}

PRUNER_INSTRUCTIONS = (
    "You are an expert AI assistant that analyzes conversational context to plan the next step.\n"
    "Your goal is to identify the minimum essential context needed to answer the user's `new_question` and decide on the correct `action`.\n"
    "AVAILABLE CONTEXT KEYS: ['instruction', 'summary', 'user_messages', 'agent_responses', 'tool_logs']"
)

BASELINE_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["general_answer", "kb_lookup", "tool_call"]},
        "rationale": {"type": "string", "description": "Explain why this action is appropriate when all context is available."}
    },
    "required": ["action", "rationale"],
    "additionalProperties": False
}

BASELINE_ACTION_INSTRUCTIONS = (
    "You are an expert AI assistant that evaluates the entire conversation context to choose the next action.\n"
    "Do NOT prune any context keys. Consider every field in `context` plus the `new_question` to select the best action.\n"
    "Return JSON with `action` (general_answer, kb_lookup, or tool_call) and a short `rationale`."
)


def configure(context: Dict[str, Any]) -> None:
    """Store judge-related configuration."""
    _CONFIG.clear()
    _CONFIG.update(context)




async def run_judge_flexible(payload: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
    """Run the judge model with flexible routing between OpenRouter and OpenAI."""
    api_routing_mode = _CONFIG.get("API_ROUTING_MODE", "openrouter")
    openai_api_key = _CONFIG.get("OPENAI_API_KEY")
    if model is None:
        model = st.session_state.get('judge_model', _CONFIG.get("OPENAI_MODEL", "openai/gpt-5-mini"))

    provider = _get_provider_from_model_id(model)

    if api_routing_mode == "openrouter" or provider in {"mistralai", "anthropic", "meta-llama", "deepseek"}:
        return await openrouter_json(
            _to_openrouter_model_id(model),
            JUDGE_INSTRUCTIONS,
            payload,
            "judge",
            JUDGE_SCHEMA,
        )
    if provider == "openai" and openai_api_key:
        native_model = _to_native_model_id(model)
        return await openai_structured_json(
            AsyncOpenAI(api_key=openai_api_key),
            native_model,
            JUDGE_INSTRUCTIONS,
            payload,
        )
    return await openrouter_json(
        _to_openrouter_model_id(model),
        JUDGE_INSTRUCTIONS,
        payload,
        "judge",
        JUDGE_SCHEMA,
    )


async def run_judge_openai(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience wrapper to force an OpenAI judge model."""
    model = st.session_state.get('judge_model', _CONFIG.get("OPENAI_MODEL", "openai/gpt-5-mini"))
    return await run_judge_flexible(payload, model)


async def run_judge_ollama(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience wrapper to force an OpenRouter judge model."""
    model = st.session_state.get('judge_model', _CONFIG.get("OPENROUTER_MODEL", "mistralai/mistral-small-3.2-24b-instruct"))
    return await run_judge_flexible(payload, model)






async def run_pruner(payload: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
    """Run the pruning model with flexible routing."""
    api_routing_mode = _CONFIG.get("API_ROUTING_MODE", "openrouter")
    openai_api_key = _CONFIG.get("OPENAI_API_KEY")
    if model is None:
        model = st.session_state.get('pruner_model', _CONFIG.get("OPENAI_MODEL", "openai/gpt-5-mini"))

    provider = _get_provider_from_model_id(model)

    if api_routing_mode == "openrouter" or provider in {"mistralai", "anthropic", "meta-llama", "deepseek"}:
        return await openrouter_json(
            _to_openrouter_model_id(model),
            PRUNER_INSTRUCTIONS,
            payload,
            "pruner",
            PRUNER_SCHEMA,
        )
    if provider == "openai" and openai_api_key:
        native_model = _to_native_model_id(model)
        return await openai_structured_json(
            AsyncOpenAI(api_key=openai_api_key),
            native_model,
            PRUNER_INSTRUCTIONS,
            payload,
        )
    return await openrouter_json(
        _to_openrouter_model_id(model),
        PRUNER_INSTRUCTIONS,
        payload,
        "pruner",
        PRUNER_SCHEMA,
    )


async def run_action_without_pruning(payload: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
    """Predict the next action using the full context without pruning."""
    api_routing_mode = _CONFIG.get("API_ROUTING_MODE", "openrouter")
    openai_api_key = _CONFIG.get("OPENAI_API_KEY")
    if model is None:
        model = st.session_state.get('pruner_model', _CONFIG.get("OPENAI_MODEL", "openai/gpt-5-mini"))

    provider = _get_provider_from_model_id(model)

    if api_routing_mode == "openrouter" or provider in {"mistralai", "anthropic", "meta-llama", "deepseek"}:
        return await openrouter_json(
            _to_openrouter_model_id(model),
            BASELINE_ACTION_INSTRUCTIONS,
            payload,
            "baseline_action",
            BASELINE_ACTION_SCHEMA,
        )
    if provider == "openai" and openai_api_key:
        native_model = _to_native_model_id(model)
        return await openai_structured_json(
            AsyncOpenAI(api_key=openai_api_key),
            native_model,
            BASELINE_ACTION_INSTRUCTIONS,
            payload,
        )
    return await openrouter_json(
        _to_openrouter_model_id(model),
        BASELINE_ACTION_INSTRUCTIONS,
        payload,
        "baseline_action",
        BASELINE_ACTION_SCHEMA,
    )
