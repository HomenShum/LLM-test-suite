"""
Pricing and model discovery functions.
Extracted from streamlit_test_v5.py to reduce main file size.

This module contains:
- OpenRouter pricing cache management
- Model discovery from various providers
- Custom price lookup functions
"""

import json
import os
import time
import streamlit as st
import httpx
from typing import Dict, Any, Optional, List
from pathlib import Path


# Pricing cache configuration
_PRICING_CACHE_DIR = Path("pricing_cache")
_OPENROUTER_PRICING_FILE = _PRICING_CACHE_DIR / "openrouter_pricing.json"
_PRICING_CACHE_TTL_DAYS = 30


def _load_pricing_from_disk() -> Optional[Dict[str, Dict[str, float]]]:
    """Load OpenRouter pricing from disk cache if it exists and is not expired."""
    if not _OPENROUTER_PRICING_FILE.exists():
        return None
    
    try:
        with open(_OPENROUTER_PRICING_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except Exception:
        # If there's any error reading the cache, return None to trigger a fresh fetch
        return None


def _save_pricing_to_disk(pricing_map: Dict[str, Dict[str, float]]) -> None:
    """Save OpenRouter pricing to disk cache."""
    _PRICING_CACHE_DIR.mkdir(exist_ok=True)
    try:
        with open(_OPENROUTER_PRICING_FILE, 'w', encoding='utf-8') as f:
            json.dump(pricing_map, f, indent=2)
    except Exception:
        # Non-critical error, just log it
        pass


def fetch_openrouter_pricing(force_refresh: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Fetches OpenRouter pricing with 30-day disk cache.
    Returns a dict mapping model IDs to pricing info.

    Args:
        force_refresh: If True, bypass cache and fetch fresh data from API
    """
    # Try to load from disk cache first (unless force refresh)
    if not force_refresh:
        cached = _load_pricing_from_disk()
        if cached is not None:
            return cached

    # Only fetch from API if explicitly requested via force_refresh
    # This prevents auto-fetching on Streamlit Cloud deployment
    if not force_refresh:
        # Return empty dict if no cache exists and not forcing refresh
        return {}

    # Fetch fresh pricing from OpenRouter API (only when force_refresh=True)
    try:
        url = "https://openrouter.ai/api/v1/models"
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        models_data = response.json().get("data", [])

        pricing_map = {}
        for model in models_data:
            model_id = model.get("id")
            pricing = model.get("pricing", {})

            if model_id and pricing:
                pricing_map[model_id] = {
                    "prompt": float(pricing.get("prompt", 0)),
                    "completion": float(pricing.get("completion", 0))
                }

        # Save to disk cache
        _save_pricing_to_disk(pricing_map)

        return pricing_map

    except Exception as e:
        st.warning(f"Failed to fetch OpenRouter pricing: {e}")
        return {}


def _to_openrouter_model_id(model: str, provider: str = None) -> str:
    """
    Convert a model name to OpenRouter format.
    
    Examples:
        "gpt-5-mini" -> "openai/gpt-5-mini"
        "gemini-2.5-flash" -> "google/gemini-2.5-flash"
    """
    if "/" in model:
        return model  # Already in correct format
    
    # Auto-detect provider if not specified
    if provider is None:
        if model.startswith("gpt"):
            provider = "openai"
        elif model.startswith("gemini"):
            provider = "google"
        elif model.startswith("claude"):
            provider = "anthropic"
        elif model.startswith("mistral"):
            provider = "mistralai"
        else:
            provider = "unknown"
    
    return f"{provider}/{model}"


def _to_native_model_id(model: str) -> str:
    """
    Convert OpenRouter format to native model ID.
    
    Examples:
        "openai/gpt-5-mini" -> "gpt-5-mini"
        "google/gemini-2.5-flash" -> "gemini-2.5-flash"
    """
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def _get_provider_from_model_id(model: str) -> str:
    """
    Extract provider from model ID.
    
    Examples:
        "openai/gpt-5-mini" -> "openai"
        "google/gemini-2.5-flash" -> "google"
    """
    if "/" in model:
        return model.split("/", 1)[0]
    
    # Auto-detect from model name
    if model.startswith("gpt"):
        return "openai"
    elif model.startswith("gemini"):
        return "google"
    elif model.startswith("claude"):
        return "anthropic"
    elif model.startswith("mistral"):
        return "mistralai"
    else:
        return "unknown"


def custom_openrouter_price_lookup(provider: str, model: str) -> Optional[Dict[str, float]]:
    """Unified custom price lookup that uses 30-day cached pricing for all providers."""
    pricing_map = fetch_openrouter_pricing()

    # Convert to OpenRouter format if needed
    model_id = _to_openrouter_model_id(model, provider)

    pricing_data = pricing_map.get(model_id)

    if not pricing_data:
        return None

    # Convert from OpenRouter format (prompt/completion) to cost tracker format (input_per_mtok_usd/output_per_mtok_usd)
    # OpenRouter pricing is already in USD per token, multiply by 1M to get per million tokens
    return {
        "input_per_mtok_usd": pricing_data.get("prompt", 0) * 1_000_000,
        "output_per_mtok_usd": pricing_data.get("completion", 0) * 1_000_000
    }


def custom_gemini_price_lookup(provider: str, model: str) -> Optional[Dict[str, float]]:
    """Custom price lookup for Gemini models using 30-day cached pricing."""
    return custom_openrouter_price_lookup("google", model)


def custom_openai_price_lookup(provider: str, model: str) -> Optional[Dict[str, float]]:
    """Custom price lookup for OpenAI models using 30-day cached pricing."""
    return custom_openrouter_price_lookup("openai", model)


# Ollama model metadata (local models have no API pricing)
OLLAMA_MODEL_METADATA = {
    "mistral:latest": {'context': '4,096', 'local_info': 'General Purpose'},
    "llama2:latest": {'context': '4,096', 'local_info': 'General Purpose'},
    "codellama:latest": {'context': '16,384', 'local_info': 'Code Generation'},
}


@st.cache_data(ttl=60 * 60 * 24 * 30)  # Cache for 30 days
def _fetch_models_from_openrouter(provider_filter: str = None) -> Dict[str, Dict[str, Any]]:
    """
    Fetch models from OpenRouter API with accurate pricing.
    
    Args:
        provider_filter: Optional provider to filter by (e.g., "openai", "google")
    
    Returns:
        Dict mapping model IDs to model metadata
    """
    try:
        url = "https://openrouter.ai/api/v1/models"
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        models_data = response.json().get("data", [])
        
        models = {}
        for model in models_data:
            model_id = model.get("id", "")
            
            # Apply provider filter if specified
            if provider_filter and not model_id.startswith(f"{provider_filter}/"):
                continue
            
            # Extract pricing
            pricing = model.get("pricing", {})
            prompt_price = float(pricing.get("prompt", 0))
            completion_price = float(pricing.get("completion", 0))
            
            # Extract context length
            context_length = model.get("context_length", 0)
            
            models[model_id] = {
                "id": model_id,
                "name": model.get("name", model_id),
                "context_length": context_length,
                "pricing": {
                    "prompt": prompt_price,
                    "completion": completion_price
                },
                "description": model.get("description", "")
            }
        
        return models
    
    except Exception as e:
        st.warning(f"Failed to fetch models from OpenRouter: {e}")
        return {}


@st.cache_data(ttl=60 * 60 * 24 * 30)  # Cache for 30 days
def fetch_gemini_models_from_linkup(force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Fetch Gemini models from Linkup API with pricing.
    Falls back to default models if API fails or if not forcing refresh.

    Args:
        force_refresh: If True, attempt to fetch from Linkup API. If False, use defaults.
    """
    # Don't auto-fetch on startup - only use defaults unless explicitly requested
    if not force_refresh:
        return _get_default_gemini_models()

    try:
        url = "https://api.linkup.so/v1/models"
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        data = response.json()

        return _parse_gemini_models_from_linkup(data)

    except Exception as e:
        # Silently fall back to default models (Linkup API is optional)
        # Use st.info instead of st.warning to reduce noise
        # st.info(f"Using default Gemini models (Linkup API unavailable)")
        return _get_default_gemini_models()


def _get_default_gemini_models() -> Dict[str, Dict[str, Any]]:
    """Fallback default Gemini 2.5 models - fetches live pricing from OpenRouter."""
    pricing_map = fetch_openrouter_pricing()
    
    default_models = {
        "gemini-2.5-flash": {
            "id": "google/gemini-2.5-flash",
            "name": "Gemini 2.5 Flash",
            "context_length": 1000000,
            "pricing": pricing_map.get("google/gemini-2.5-flash", {"prompt": 0, "completion": 0})
        },
        "gemini-2.5-pro": {
            "id": "google/gemini-2.5-pro",
            "name": "Gemini 2.5 Pro",
            "context_length": 2000000,
            "pricing": pricing_map.get("google/gemini-2.5-pro", {"prompt": 0, "completion": 0})
        }
    }
    
    return default_models


def _parse_gemini_models_from_linkup(data: dict) -> Dict[str, Dict[str, Any]]:
    """
    Parse Gemini models from Linkup API response.
    """
    models = {}
    pricing_map = fetch_openrouter_pricing()
    
    for model in data.get("models", []):
        model_id = model.get("id", "")
        
        # Only include Gemini models
        if not model_id.startswith("gemini"):
            continue
        
        # Get pricing from OpenRouter cache
        openrouter_id = f"google/{model_id}"
        pricing = pricing_map.get(openrouter_id, {"prompt": 0, "completion": 0})
        
        models[model_id] = {
            "id": openrouter_id,
            "name": model.get("name", model_id),
            "context_length": model.get("context_length", 0),
            "pricing": pricing
        }
    
    return models if models else _get_default_gemini_models()


@st.cache_data(ttl=60 * 60 * 24 * 30)  # Cache for 30 days
def fetch_openai_models_from_linkup(force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Fetch OpenAI models from Linkup API with pricing.
    Falls back to default models if API fails or if not forcing refresh.

    Args:
        force_refresh: If True, attempt to fetch from Linkup API. If False, use defaults.
    """
    # Don't auto-fetch on startup - only use defaults unless explicitly requested
    if not force_refresh:
        return _get_default_openai_models()

    try:
        url = "https://api.linkup.so/v1/models"
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        data = response.json()

        return _parse_openai_models_from_linkup(data)

    except Exception as e:
        # Silently fall back to default models (Linkup API is optional)
        # Use st.info instead of st.warning to reduce noise
        # st.info(f"Using default OpenAI models (Linkup API unavailable)")
        return _get_default_openai_models()


def _get_default_openai_models() -> Dict[str, Dict[str, Any]]:
    """Fallback default GPT-5 models - fetches live pricing from OpenRouter."""
    pricing_map = fetch_openrouter_pricing()

    default_models = {
        "gpt-5-mini": {
            "id": "openai/gpt-5-mini",
            "name": "GPT-5 Mini",
            "context_length": 128000,
            "pricing": pricing_map.get("openai/gpt-5-mini", {"prompt": 0, "completion": 0})
        },
        "gpt-5": {
            "id": "openai/gpt-5",
            "name": "GPT-5",
            "context_length": 128000,
            "pricing": pricing_map.get("openai/gpt-5", {"prompt": 0, "completion": 0})
        }
    }

    return default_models


def _parse_openai_models_from_linkup(data: dict) -> Dict[str, Dict[str, Any]]:
    """
    Parse OpenAI models from Linkup API response.
    """
    models = {}
    pricing_map = fetch_openrouter_pricing()

    for model in data.get("models", []):
        model_id = model.get("id", "")

        # Only include GPT models
        if not model_id.startswith("gpt"):
            continue

        # Get pricing from OpenRouter cache
        openrouter_id = f"openai/{model_id}"
        pricing = pricing_map.get(openrouter_id, {"prompt": 0, "completion": 0})

        models[model_id] = {
            "id": openrouter_id,
            "name": model.get("name", model_id),
            "context_length": model.get("context_length", 0),
            "pricing": pricing
        }

    return models if models else _get_default_openai_models()


def get_all_available_models() -> List[str]:
    """
    Returns a list of all available model IDs across all providers.
    """
    all_models = []

    # Get OpenRouter models
    openrouter_models = _fetch_models_from_openrouter()
    all_models.extend(openrouter_models.keys())

    # Get Gemini models
    gemini_models = fetch_gemini_models_from_linkup()
    all_models.extend([m["id"] for m in gemini_models.values()])

    # Get OpenAI models
    openai_models = fetch_openai_models_from_linkup()
    all_models.extend([m["id"] for m in openai_models.values()])

    # Get Ollama models
    all_models.extend(OLLAMA_MODEL_METADATA.keys())

    return sorted(set(all_models))

