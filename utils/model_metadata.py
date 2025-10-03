"""Utilities for loading model metadata for UI selections."""

from typing import Dict, Any, Tuple

from core.pricing import get_all_available_models, fetch_gemini_models_from_linkup
from utils.model_discovery import fetch_openrouter_models_for_ui, fetch_openai_models

DEFAULT_OPENROUTER_MODEL_METADATA: Dict[str, Dict[str, str]] = {
    "mistralai/mistral-small-3.2-24b-instruct": {'context': '131,072', 'input_cost': '$0.06', 'output_cost': '$0.18'},
    "deepseek/deepseek-v3.1-terminus": {'context': '128,000', 'input_cost': '$0.14', 'output_cost': '$0.28'},
}

OLLAMA_MODEL_METADATA: Dict[str, Dict[str, str]] = {
    "mistral:latest": {'context': '4,096', 'local_info': 'General Purpose'},
    "llama3:8b": {'context': '8,192', 'local_info': 'Next-Gen Llama'},
    "mistral-small:24b-instruct-2501-q4_K_M": {'context': '32,768', 'local_info': 'Mistral Small Quantized'},
    "Custom...": {'context': 'N/A', 'local_info': 'User Defined'}
}


def _parse_pricing(pricing: Any) -> Tuple[str, str]:
    """Normalize pricing information into display-friendly strings."""
    if isinstance(pricing, dict):
        prompt = pricing.get('prompt')
        completion = pricing.get('completion')
        input_cost = f"${prompt * 1_000_000:.2f}" if prompt is not None else "Unknown"
        output_cost = f"${completion * 1_000_000:.2f}" if completion is not None else "Unknown"
        return input_cost, output_cost

    if isinstance(pricing, str):
        text = pricing.split(' per', 1)[0]
        if '/' in text:
            parts = text.split('/', 1)
            return parts[0].strip(), parts[1].strip()
        cleaned = text.strip()
        return cleaned or "Unknown", "Unknown"

    return "Unknown", "Unknown"


def _coerce_context(entry: Dict[str, Any]) -> str:
    context = entry.get('context')
    if context:
        return str(context)
    context_length = entry.get('context_length') or entry.get('max_context_length')
    if context_length:
        try:
            return f"{int(context_length):,}"
        except (ValueError, TypeError):
            return str(context_length)
    return "Unknown"


def _build_metadata(entries: Dict[str, Dict[str, Any]], fallbacks: Dict[str, Dict[str, str]] = None) -> Dict[str, Dict[str, str]]:
    metadata: Dict[str, Dict[str, str]] = {}

    for entry in entries.values():
        model_id = entry.get('id') or entry.get('model_id') or entry.get('name')
        if not model_id:
            continue
        context = _coerce_context(entry)
        input_cost, output_cost = _parse_pricing(entry.get('pricing'))
        metadata[model_id] = {
            'context': context,
            'input_cost': input_cost,
            'output_cost': output_cost
        }

    if fallbacks:
        for model_id, fallback in fallbacks.items():
            metadata.setdefault(model_id, fallback)

    return metadata


def load_model_metadata() -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]], Dict[str, Dict[str, str]], list]:
    """Load model metadata for OpenRouter, OpenAI, and Gemini along with the available model list."""
    openrouter_raw = fetch_openrouter_models_for_ui()
    openrouter_metadata = _build_metadata(openrouter_raw, DEFAULT_OPENROUTER_MODEL_METADATA)

    openai_raw = fetch_openai_models()
    openai_metadata = _build_metadata(openai_raw)
    if not openai_metadata:
        openai_metadata['openai/gpt-5-mini'] = {'context': '128,000', 'input_cost': '$0.25', 'output_cost': '$2.00'}

    gemini_raw = fetch_gemini_models_from_linkup()
    # Convert keys to match OpenRouter-style IDs if missing
    gemini_processed = {}
    for key, value in gemini_raw.items():
        model_id = value.get('id') or f"google/{key}"
        gemini_processed[model_id] = {
            'id': model_id,
            'context_length': value.get('context_length'),
            'pricing': value.get('pricing')
        }
    gemini_metadata = _build_metadata(gemini_processed)
    if not gemini_metadata:
        gemini_metadata['google/gemini-2.5-flash'] = {'context': '1,000,000', 'input_cost': '$0.35', 'output_cost': '$1.05'}

    available_models = get_all_available_models()
    return openrouter_metadata, openai_metadata, gemini_metadata, available_models

