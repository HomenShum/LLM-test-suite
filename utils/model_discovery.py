"""
Model discovery functions for UI display.
Extracted from streamlit_test_v5.py to reduce main file size.

This module contains:
- OpenRouter model fetching for UI
- OpenAI model fetching for UI
- Third model display name logic
"""

import streamlit as st
import httpx
from typing import Dict, Any, Optional
from core.pricing import (
    fetch_openrouter_pricing,
    fetch_openai_models_from_linkup,
    _to_openrouter_model_id
)


# Default models configuration
OPENROUTER_MODEL = "mistralai/mistral-large-2411"
OPENAI_MODEL = "gpt-5-mini"
THIRD_MODEL_KIND = "None"  # "None", "Gemini", or "Ollama"
THIRD_MODEL = ""


@st.cache_data(ttl=60 * 60 * 24 * 30)  # Cache for 30 days
def fetch_openrouter_models_for_ui() -> Dict[str, Dict[str, Any]]:
    """Fetches models from OpenRouter that support structured outputs, with metadata for UI display."""
    try:
        url = "https://openrouter.ai/api/v1/models"
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        models_data = response.json().get("data", [])
        
        # Get pricing map
        pricing_map = fetch_openrouter_pricing()
        
        models = {}
        for model in models_data:
            model_id = model.get("id", "")
            
            # Filter for models that support structured outputs
            # (This is a heuristic - adjust based on actual API capabilities)
            if not any(keyword in model_id.lower() for keyword in ["gpt", "claude", "gemini", "mistral", "llama"]):
                continue
            
            # Get pricing
            pricing = pricing_map.get(model_id, {"prompt": 0, "completion": 0})
            
            # Format pricing for display
            prompt_price = pricing.get("prompt", 0)
            completion_price = pricing.get("completion", 0)
            
            # Convert to dollars per million tokens
            prompt_per_m = prompt_price * 1_000_000
            completion_per_m = completion_price * 1_000_000
            
            # Get context length
            context_length = model.get("context_length", 0)
            context_str = f"{context_length:,}" if context_length > 0 else "Unknown"
            
            models[model_id] = {
                "id": model_id,
                "name": model.get("name", model_id),
                "context": context_str,
                "pricing": f"${prompt_per_m:.2f}/${completion_per_m:.2f} per 1M tokens",
                "description": model.get("description", "")
            }
        
        return models
    
    except Exception as e:
        st.warning(f"Failed to fetch OpenRouter models: {e}")
        return _get_fallback_openrouter_models()


def _get_fallback_openrouter_models() -> Dict[str, Dict[str, Any]]:
    """Fallback models if API fetch fails."""
    pricing_map = fetch_openrouter_pricing()
    
    fallback_models = {
        "mistralai/mistral-large-2411": {
            "id": "mistralai/mistral-large-2411",
            "name": "Mistral Large 2411",
            "context": "128,000",
            "pricing": "Unknown",
            "description": "Mistral's largest model"
        },
        "openai/gpt-5-mini": {
            "id": "openai/gpt-5-mini",
            "name": "GPT-5 Mini",
            "context": "128,000",
            "pricing": "Unknown",
            "description": "OpenAI's efficient model"
        },
        "google/gemini-2.5-flash": {
            "id": "google/gemini-2.5-flash",
            "name": "Gemini 2.5 Flash",
            "context": "1,000,000",
            "pricing": "Unknown",
            "description": "Google's fast model"
        }
    }
    
    # Update pricing if available
    for model_id, model_data in fallback_models.items():
        pricing = pricing_map.get(model_id)
        if pricing:
            prompt_per_m = pricing.get("prompt", 0) * 1_000_000
            completion_per_m = pricing.get("completion", 0) * 1_000_000
            model_data["pricing"] = f"${prompt_per_m:.2f}/${completion_per_m:.2f} per 1M tokens"
    
    return fallback_models


@st.cache_data(ttl=60 * 60 * 24 * 30)  # Cache for 30 days
def fetch_openai_models() -> Dict[str, Dict[str, Any]]:
    """Returns OpenAI models discovered via Linkup API (cached for 30 days)."""
    models = fetch_openai_models_from_linkup()
    
    # Format for UI display
    ui_models = {}
    for model_id, model_data in models.items():
        pricing = model_data.get("pricing", {})
        prompt_price = pricing.get("prompt", 0)
        completion_price = pricing.get("completion", 0)
        
        # Convert to dollars per million tokens
        prompt_per_m = prompt_price * 1_000_000
        completion_per_m = completion_price * 1_000_000
        
        context_length = model_data.get("context_length", 0)
        context_str = f"{context_length:,}" if context_length > 0 else "Unknown"
        
        ui_models[model_id] = {
            "id": model_data.get("id", model_id),
            "name": model_data.get("name", model_id),
            "context": context_str,
            "pricing": f"${prompt_per_m:.2f}/${completion_per_m:.2f} per 1M tokens"
        }
    
    return ui_models


def get_third_model_display_name() -> str:
    """Dynamically returns the name of the configured third model."""
    third_kind = st.session_state.get('third_model_kind', THIRD_MODEL_KIND)
    third_model = st.session_state.get('third_model', THIRD_MODEL)
    
    if third_kind == "None" or not third_model:
        return "None"
    elif third_kind == "Gemini":
        return f"Gemini ({third_model})"
    elif third_kind == "Ollama":
        return f"Ollama ({third_model})"
    else:
        return third_model


def _normalize_ollama_root(url: str) -> str:
    """Normalize Ollama base URL."""
    u = (url or "").rstrip("/")
    if not u:
        u = "http://localhost:11434"
    return u

