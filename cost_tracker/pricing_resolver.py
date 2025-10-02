# cost_tracker/pricing_resolver.py
import os
import re
import json
import requests
from typing import Optional

LINKUP_API_KEY = os.getenv("LINKUP_API_KEY", "")
LINKUP_SEARCH_URL = "https://api.linkup.so/v1/search"
LINKUP_FETCH_URL = "https://api.linkup.so/v1/fetch"

# Known pricing page URLs
PRICING_URLS = {
    "OpenAI": "https://openai.com/api/pricing/",
    "Google": "https://ai.google.dev/pricing",
    "Anthropic": "https://www.anthropic.com/pricing",
}

# Look for "$X per 1M tokens" patterns (input/prompt vs output/completion)
P_IN  = re.compile(r"(input|prompt)[^$]{0,60}\$?([0-9]+(?:\.[0-9]+)?)\s*/\s*(?:1m|million)\s*tokens", re.I)
P_OUT = re.compile(r"(output|completion)[^$]{0,60}\$?([0-9]+(?:\.[0-9]+)?)\s*/\s*(?:1m|million)\s*tokens", re.I)

def _first(text: str, pat: re.Pattern) -> Optional[float]:
    """Extract first match of pricing pattern from text."""
    m = pat.search(text)
    return float(m.group(2)) if m else None

def _extract_model_section(content: str, model: str) -> Optional[str]:
    """
    Extract the section of content that mentions the specific model.

    Args:
        content: Full page content
        model: Model identifier (e.g., "gpt-4o", "gemini-1.5-pro")

    Returns:
        Section of text around the model mention, or None if not found
    """
    # Normalize model name for searching
    model_normalized = model.lower().replace("-", "").replace("_", "").replace(".", "")
    content_lower = content.lower()

    # Try to find the model name in the content
    model_variations = [
        model.lower(),
        model.lower().replace("-", " "),
        model.lower().replace("_", " "),
        model_normalized
    ]

    for variation in model_variations:
        idx = content_lower.find(variation)
        if idx != -1:
            # Extract a window around the model mention (500 chars before and after)
            start = max(0, idx - 500)
            end = min(len(content), idx + 500)
            return content[start:end]

    return None

def _fetch_pricing_page(provider: str) -> Optional[str]:
    """
    Fetch pricing page content using Linkup Fetch API.

    Args:
        provider: Provider name (e.g., "OpenAI", "Google", "Anthropic")

    Returns:
        Page content as string, or None if fetch fails
    """
    if not LINKUP_API_KEY or provider not in PRICING_URLS:
        return None

    url = PRICING_URLS[provider]
    headers = {
        "Authorization": f"Bearer {LINKUP_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "url": url,
        "includeRawHtml": False,
        "renderJs": True,  # Enable JS rendering for dynamic pricing pages
        "extractImages": False
    }

    try:
        r = requests.post(LINKUP_FETCH_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()

        # Extract content from response
        content = data.get("content", "") or data.get("text", "")
        return str(content) if content else None

    except Exception as e:
        print(f"Warning: Linkup fetch failed for {provider} pricing page: {e}")
        return None

def linkup_price_lookup(provider: str, model: str) -> Optional[dict]:
    """
    Use Linkup API to find pricing for a model.

    Strategy:
    1. Try fetching the official pricing page directly (faster, more accurate)
    2. Fall back to search API if fetch fails

    Args:
        provider: Provider name (e.g., "OpenAI", "Anthropic", "OpenRouter", "Google")
        model: Model identifier

    Returns:
        dict with input_per_mtok_usd and output_per_mtok_usd, or None if not found
    """
    if not LINKUP_API_KEY:
        return None

    # Strategy 1: Try fetching the pricing page directly
    if provider in PRICING_URLS:
        page_content = _fetch_pricing_page(provider)
        if page_content:
            # Search for model-specific pricing in the page
            model_section = _extract_model_section(page_content, model)
            if model_section:
                inp = _first(model_section, P_IN)
                out = _first(model_section, P_OUT)

                if inp is not None or out is not None:
                    return {
                        "input_per_mtok_usd": float(inp or 0.0),
                        "output_per_mtok_usd": float(out or 0.0)
                    }

    # Strategy 2: Fall back to search API

    # Build search query based on provider
    if provider == "Google":
        # For Google/Gemini, search their pricing page
        q = f"Google Gemini {model} pricing API cost per million tokens"
        include_domains = ["ai.google.dev", "cloud.google.com"]
    elif provider == "OpenAI":
        # For OpenAI, search their pricing page
        q = f"OpenAI {model} pricing API cost per million tokens"
        include_domains = ["openai.com"]
    elif provider == "Anthropic":
        # For Anthropic, search their pricing page
        q = f"Anthropic Claude {model} pricing API cost per million tokens"
        include_domains = ["anthropic.com"]
    else:
        # Generic search for other providers
        q = f"{provider} {model} pricing tokens per 1M input output cost"
        include_domains = []

    headers = {
        "Authorization": f"Bearer {LINKUP_API_KEY}",
        "Content-Type": "application/json"
    }

    # Correct Linkup API payload format
    payload = {
        "q": q,
        "depth": "standard",
        "outputType": "sourcedAnswer",
        "includeImages": False,
        "includeInlineCitations": False,
        "includeSources": True
    }

    # Add domain filtering if specified
    if include_domains:
        payload["includeDomains"] = include_domains

    try:
        r = requests.post(LINKUP_SEARCH_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()

        # Extract text from response
        texts = []

        # Try to get the answer field
        answer = data.get("answer", "")
        if answer:
            texts.append(str(answer))

        # Try to get sources
        sources = data.get("sources", [])
        for source in sources:
            content = source.get("content", "") or source.get("snippet", "")
            if content:
                texts.append(str(content))

        # Combine all text
        blob = "\n\n".join(texts)[:25000]

        # Extract pricing using regex patterns
        inp = _first(blob, P_IN)
        out = _first(blob, P_OUT)

        if inp is None and out is None:
            return None

        return {
            "input_per_mtok_usd": float(inp or 0.0),
            "output_per_mtok_usd": float(out or 0.0)
        }
    except Exception as e:
        print(f"Warning: Linkup price lookup failed for {provider}/{model}: {e}")
        return None

def fallback_price_lookup(provider: str, model: str) -> Optional[dict]:
    """
    Fallback pricing lookup using hardcoded common models.
    This is used when Linkup API is not available or fails.
    """
    # Normalize provider and model names
    provider_lower = provider.lower()
    model_lower = model.lower()
    
    # Common pricing (as of 2025, approximate)
    pricing_db = {
        ("openai", "gpt-4"): {"input_per_mtok_usd": 30.0, "output_per_mtok_usd": 60.0},
        ("openai", "gpt-4-turbo"): {"input_per_mtok_usd": 10.0, "output_per_mtok_usd": 30.0},
        ("openai", "gpt-4o"): {"input_per_mtok_usd": 5.0, "output_per_mtok_usd": 15.0},
        ("openai", "gpt-4o-mini"): {"input_per_mtok_usd": 0.15, "output_per_mtok_usd": 0.6},
        ("openai", "gpt-3.5-turbo"): {"input_per_mtok_usd": 0.5, "output_per_mtok_usd": 1.5},
        ("openai", "gpt-5-mini"): {"input_per_mtok_usd": 0.15, "output_per_mtok_usd": 0.6},  # Assuming similar to 4o-mini
        
        ("anthropic", "claude-3-opus"): {"input_per_mtok_usd": 15.0, "output_per_mtok_usd": 75.0},
        ("anthropic", "claude-3-sonnet"): {"input_per_mtok_usd": 3.0, "output_per_mtok_usd": 15.0},
        ("anthropic", "claude-3-haiku"): {"input_per_mtok_usd": 0.25, "output_per_mtok_usd": 1.25},
        ("anthropic", "claude-3-5-sonnet"): {"input_per_mtok_usd": 3.0, "output_per_mtok_usd": 15.0},
        
        ("google", "gemini-pro"): {"input_per_mtok_usd": 0.5, "output_per_mtok_usd": 1.5},
        ("google", "gemini-2.5-flash"): {"input_per_mtok_usd": 0.075, "output_per_mtok_usd": 0.3},
        ("google", "gemini-1.5-pro"): {"input_per_mtok_usd": 1.25, "output_per_mtok_usd": 5.0},
        ("google", "gemini-1.5-flash"): {"input_per_mtok_usd": 0.075, "output_per_mtok_usd": 0.3},
        
        ("mistral", "mistral-small"): {"input_per_mtok_usd": 1.0, "output_per_mtok_usd": 3.0},
        ("mistral", "mistral-medium"): {"input_per_mtok_usd": 2.7, "output_per_mtok_usd": 8.1},
        ("mistral", "mistral-large"): {"input_per_mtok_usd": 4.0, "output_per_mtok_usd": 12.0},
        
        ("cohere", "command"): {"input_per_mtok_usd": 1.0, "output_per_mtok_usd": 2.0},
        ("cohere", "command-light"): {"input_per_mtok_usd": 0.3, "output_per_mtok_usd": 0.6},
        
        ("groq", "llama-3.1-70b"): {"input_per_mtok_usd": 0.59, "output_per_mtok_usd": 0.79},
        ("groq", "llama-3.1-8b"): {"input_per_mtok_usd": 0.05, "output_per_mtok_usd": 0.08},
        ("groq", "mixtral-8x7b"): {"input_per_mtok_usd": 0.24, "output_per_mtok_usd": 0.24},
    }
    
    # Try exact match first
    key = (provider_lower, model_lower)
    if key in pricing_db:
        return pricing_db[key]
    
    # Try partial matches (e.g., "gpt-4o-mini-2024-07-18" matches "gpt-4o-mini")
    for (p, m), pricing in pricing_db.items():
        if provider_lower == p and m in model_lower:
            return pricing
    
    # For OpenRouter, try to extract the base model
    if "openrouter" in provider_lower or "/" in model:
        # OpenRouter format: "provider/model"
        parts = model.split("/")
        if len(parts) == 2:
            base_provider, base_model = parts
            return fallback_price_lookup(base_provider, base_model)
    
    return None

def combined_price_lookup(provider: str, model: str) -> Optional[dict]:
    """
    Combined pricing lookup: tries Linkup first, then falls back to hardcoded prices.
    """
    # Try Linkup first
    result = linkup_price_lookup(provider, model)
    if result:
        return result
    
    # Fall back to hardcoded prices
    result = fallback_price_lookup(provider, model)
    if result:
        return result
    
    # Last resort: return zeros (will be cached, so manual update possible)
    return {"input_per_mtok_usd": 0.0, "output_per_mtok_usd": 0.0}

