"""
Vision model discovery and caching for Test 6.

This module fetches and caches vision-capable models from OpenRouter API.
Cache is stored locally with 30-day TTL to avoid repeated API calls.
"""

import json
import httpx
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import streamlit as st


# Cache configuration
_VISION_CACHE_DIR = Path("pricing_cache")
_VISION_MODELS_FILE = _VISION_CACHE_DIR / "openrouter_vision_models.json"
_VISION_CACHE_TTL_DAYS = 30


def _load_vision_models_from_disk() -> Optional[Dict[str, Any]]:
    """Load OpenRouter vision models from disk cache if it exists and is not expired."""
    if not _VISION_MODELS_FILE.exists():
        return None
    
    try:
        with open(_VISION_MODELS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if cache is expired
        cached_time = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
        if datetime.now() - cached_time > timedelta(days=_VISION_CACHE_TTL_DAYS):
            return None
        
        return data.get("models", {})
    except Exception:
        return None


def _save_vision_models_to_disk(models: Dict[str, Any]) -> None:
    """Save OpenRouter vision models to disk cache."""
    _VISION_CACHE_DIR.mkdir(exist_ok=True)
    try:
        data = {
            "cached_at": datetime.now().isoformat(),
            "models": models
        }
        with open(_VISION_MODELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def fetch_openrouter_vision_models(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Fetches OpenRouter vision-capable models with 30-day disk cache.
    Returns a dict mapping model IDs to model info.

    Args:
        force_refresh: If True, bypass cache and fetch fresh data from API
    
    Returns:
        Dict mapping model IDs to model metadata (name, pricing, context_length, etc.)
    """
    # Try to load from disk cache first (unless force refresh)
    if not force_refresh:
        cached = _load_vision_models_from_disk()
        if cached is not None:
            return cached

    # Fetch from OpenRouter API
    try:
        url = "https://openrouter.ai/api/v1/models"
        
        async def fetch_models():
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
        
        import asyncio
        loop = asyncio.get_event_loop()
        data = loop.run_until_complete(fetch_models())
        
        # Filter for vision-capable models (input_modalities includes "image")
        vision_models = {}
        
        for model in data.get("data", []):
            model_id = model.get("id", "")
            input_modalities = model.get("architecture", {}).get("input_modalities", [])
            
            # Check if model supports image input
            if "image" in input_modalities:
                pricing = model.get("pricing", {})
                
                vision_models[model_id] = {
                    "id": model_id,
                    "name": model.get("name", model_id),
                    "context_length": model.get("context_length", 0),
                    "pricing": {
                        "prompt": float(pricing.get("prompt", 0)),
                        "completion": float(pricing.get("completion", 0)),
                        "image": float(pricing.get("image", 0))  # Some models have separate image pricing
                    },
                    "input_modalities": input_modalities,
                    "output_modalities": model.get("architecture", {}).get("output_modalities", []),
                    "provider": model_id.split("/")[0] if "/" in model_id else "unknown"
                }
        
        # Save to disk cache
        _save_vision_models_to_disk(vision_models)
        
        return vision_models
    
    except Exception as e:
        st.warning(f"Failed to fetch OpenRouter vision models: {e}")
        # Return default fallback models
        return _get_default_vision_models()


def _get_default_vision_models() -> Dict[str, Any]:
    """
    Fallback default vision models if API fetch fails.
    Uses latest GPT-5, Gemini 2.5, Claude 3.5, and Llama 3.2 vision models.
    """
    return {
        "openai/gpt-5-mini": {
            "id": "openai/gpt-5-mini",
            "name": "GPT-5 Mini",
            "context_length": 128000,
            "pricing": {"prompt": 0.0, "completion": 0.0, "image": 0.0},
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
            "provider": "openai"
        },
        "openai/gpt-5-nano": {
            "id": "openai/gpt-5-nano",
            "name": "GPT-5 Nano",
            "context_length": 128000,
            "pricing": {"prompt": 0.0, "completion": 0.0, "image": 0.0},
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
            "provider": "openai"
        },
        "google/gemini-2.5-flash-lite": {
            "id": "google/gemini-2.5-flash-lite",
            "name": "Gemini 2.5 Flash Lite",
            "context_length": 1000000,
            "pricing": {"prompt": 0.0, "completion": 0.0, "image": 0.0},
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
            "provider": "google"
        },
        "google/gemini-2.5-flash": {
            "id": "google/gemini-2.5-flash",
            "name": "Gemini 2.5 Flash",
            "context_length": 1000000,
            "pricing": {"prompt": 0.0, "completion": 0.0, "image": 0.0},
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
            "provider": "google"
        },
        "anthropic/claude-sonnet-4.5": {
            "id": "anthropic/claude-sonnet-4.5",
            "name": "Claude Sonnet 4.5",
            "context_length": 1000000,
            "pricing": {"prompt": 0.0, "completion": 0.0, "image": 0.0},
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
            "provider": "anthropic"
        },
        "anthropic/claude-3.5-sonnet": {
            "id": "anthropic/claude-3.5-sonnet",
            "name": "Claude 3.5 Sonnet",
            "context_length": 200000,
            "pricing": {"prompt": 0.0, "completion": 0.0, "image": 0.0},
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
            "provider": "anthropic"
        },
        "meta-llama/llama-3.2-90b-vision-instruct": {
            "id": "meta-llama/llama-3.2-90b-vision-instruct",
            "name": "Llama 3.2 90B Vision",
            "context_length": 128000,
            "pricing": {"prompt": 0.0, "completion": 0.0, "image": 0.0},
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
            "provider": "meta-llama"
        }
    }


def get_vision_models_by_provider(provider: str = None) -> Dict[str, Any]:
    """
    Get vision models, optionally filtered by provider.
    
    Args:
        provider: Optional provider filter (e.g., "openai", "google", "anthropic", "meta-llama")
    
    Returns:
        Dict of vision models matching the filter
    """
    all_models = fetch_openrouter_vision_models()
    
    if provider is None:
        return all_models
    
    return {
        model_id: model_info
        for model_id, model_info in all_models.items()
        if model_info.get("provider") == provider
    }


def get_recommended_vision_models() -> Dict[str, str]:
    """
    Get recommended vision models for Test 6.
    
    Returns:
        Dict mapping model categories to recommended model IDs
    """
    all_models = fetch_openrouter_vision_models()
    
    # Try to find the best models from each provider
    recommendations = {}
    
    # OpenAI: Prefer gpt-5-nano or gpt-5-mini
    openai_models = [m for m in all_models.keys() if m.startswith("openai/")]
    if "openai/gpt-5-nano" in openai_models:
        recommendations["openai"] = "openai/gpt-5-nano"
    elif "openai/gpt-5-mini" in openai_models:
        recommendations["openai"] = "openai/gpt-5-mini"
    elif openai_models:
        recommendations["openai"] = openai_models[0]
    
    # Google: Prefer gemini-2.5-flash-lite
    google_models = [m for m in all_models.keys() if m.startswith("google/")]
    if "google/gemini-2.5-flash-lite" in google_models:
        recommendations["google"] = "google/gemini-2.5-flash-lite"
    elif "google/gemini-2.5-flash" in google_models:
        recommendations["google"] = "google/gemini-2.5-flash"
    elif google_models:
        recommendations["google"] = google_models[0]
    
    # Anthropic: Prefer claude-sonnet-4.5 (latest), then claude-3.5-sonnet
    anthropic_models = [m for m in all_models.keys() if m.startswith("anthropic/")]
    if "anthropic/claude-sonnet-4.5" in anthropic_models:
        recommendations["anthropic"] = "anthropic/claude-sonnet-4.5"
    elif "anthropic/claude-3.5-sonnet" in anthropic_models:
        recommendations["anthropic"] = "anthropic/claude-3.5-sonnet"
    elif anthropic_models:
        recommendations["anthropic"] = anthropic_models[0]
    
    # Meta: Prefer llama-3.2-90b-vision-instruct
    meta_models = [m for m in all_models.keys() if m.startswith("meta-llama/")]
    if "meta-llama/llama-3.2-90b-vision-instruct" in meta_models:
        recommendations["meta-llama"] = "meta-llama/llama-3.2-90b-vision-instruct"
    elif meta_models:
        recommendations["meta-llama"] = meta_models[0]
    
    return recommendations


def get_all_vision_model_ids() -> List[str]:
    """Get list of all available vision model IDs."""
    return list(fetch_openrouter_vision_models().keys())


def get_vision_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed info for a specific vision model.
    
    Args:
        model_id: The model ID (e.g., "openai/gpt-5-nano")
    
    Returns:
        Model info dict or None if not found
    """
    all_models = fetch_openrouter_vision_models()
    return all_models.get(model_id)

