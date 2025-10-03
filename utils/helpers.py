"""
Miscellaneous helper functions.
Extracted from streamlit_test_v5.py to reduce main file size.

This module contains:
- Retry logic
- Prompt enhancement
- Configuration capture and display
- Report generation helpers
"""

import asyncio
import streamlit as st
from typing import Dict, Any, Optional, Callable
from datetime import datetime


async def _retry(op: Callable, attempts: int = 3, base_delay: float = 0.5):
    """Retry an async operation with exponential backoff."""
    last = None
    for i in range(attempts):
        try:
            return await op()
        except Exception as e:
            last = e
            if i < attempts - 1:
                await asyncio.sleep(base_delay * (2 ** i))
    raise last


def enhance_prompt_with_user_input(base_prompt: str, user_input: str, context: Dict[str, Any]) -> str:
    """
    Enhances a base prompt with user input and context.
    
    Args:
        base_prompt: The base prompt template
        user_input: User's custom input
        context: Additional context dictionary
    
    Returns:
        Enhanced prompt string
    """
    if not user_input or not user_input.strip():
        return base_prompt
    
    # Build context string
    context_str = ""
    if context:
        context_items = []
        for key, value in context.items():
            if value:
                context_items.append(f"{key}: {value}")
        if context_items:
            context_str = "\n\nAdditional Context:\n" + "\n".join(context_items)
    
    # Combine base prompt with user input and context
    enhanced = f"{base_prompt}\n\nUser Instructions:\n{user_input}{context_str}"
    
    return enhanced


def capture_run_config(test_name: str, overrides: Optional[Dict[str, Any]] = None):
    """
    Snapshots the current run's configuration into session state.
    
    Args:
        test_name: Name of the test (e.g., "Test 1", "Test 2")
        overrides: Optional dict of config values to override
    """
    config_key = f"{test_name}_run_config"
    
    # Base configuration from session state
    config = {
        "test_name": test_name,
        "timestamp": datetime.now().isoformat(),
        "api_routing_mode": st.session_state.get("api_routing_mode", "openrouter"),
        "models": {
            "openrouter": st.session_state.get("openrouter_model", "mistralai/mistral-large-2411"),
            "openai": st.session_state.get("openai_model", "gpt-5-mini"),
            "third_model_kind": st.session_state.get("third_model_kind", "None"),
            "third_model": st.session_state.get("third_model", ""),
        },
        "dataset": {
            "row_limit": st.session_state.get("row_limit", None),
            "dataset_size": st.session_state.get("dataset_size", 0),
        }
    }
    
    # Apply overrides
    if overrides:
        config.update(overrides)
    
    # Store in session state
    st.session_state[config_key] = config


def display_run_config(test_name: str):
    """
    Displays the captured configuration in a Streamlit expander.
    
    Args:
        test_name: Name of the test (e.g., "Test 1", "Test 2")
    """
    config_key = f"{test_name}_run_config"
    
    if config_key not in st.session_state:
        return
    
    config = st.session_state[config_key]
    
    with st.expander("📋 Run Configuration", expanded=False):
        st.json(config)


def _non_empty(s):
    """Count non-empty values in a pandas Series."""
    return s.fillna("").astype(str).str.strip().replace("nan","").replace("None","").ne("").sum()


def format_cost(cost: float) -> str:
    """Format cost as currency string."""
    if cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1.0:
        return f"${cost:.3f}"
    else:
        return f"${cost:.2f}"


def format_tokens(tokens: int) -> str:
    """Format token count with thousands separator."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.2f}M"
    elif tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    else:
        return str(tokens)


def truncate_text(text: str, max_length: int = 150) -> str:
    """Truncate text to max length with ellipsis."""
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def safe_json_loads(s: str, default=None):
    """Safely load JSON string, returning default on error."""
    import json
    try:
        return json.loads(s)
    except Exception:
        return default if default is not None else {}


def safe_get(d: dict, *keys, default=None):
    """Safely get nested dict value."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key)
        else:
            return default
    return d if d is not None else default


def calculate_percentage(part: float, total: float) -> float:
    """Calculate percentage, handling division by zero."""
    if total == 0:
        return 0.0
    return (part / total) * 100


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage with specified decimals."""
    return f"{value:.{decimals}f}%"


def get_color_for_score(score: float, thresholds: Dict[str, float] = None) -> str:
    """
    Get color based on score thresholds.
    
    Args:
        score: Score value (0-100)
        thresholds: Dict with 'excellent', 'good', 'fair' keys
    
    Returns:
        Color string (hex or name)
    """
    if thresholds is None:
        thresholds = {"excellent": 90, "good": 70, "fair": 50}
    
    if score >= thresholds.get("excellent", 90):
        return "#28a745"  # Green
    elif score >= thresholds.get("good", 70):
        return "#ffc107"  # Yellow
    elif score >= thresholds.get("fair", 50):
        return "#fd7e14"  # Orange
    else:
        return "#dc3545"  # Red


def get_status_emoji(status: str) -> str:
    """Get emoji for status string."""
    status_map = {
        "success": "✅",
        "complete": "✅",
        "verified": "✅",
        "running": "🔄",
        "in_progress": "🔄",
        "pending": "⏳",
        "waiting": "⏳",
        "failed": "❌",
        "error": "❌",
        "warning": "⚠️",
        "partial": "⚠️",
        "cancelled": "🚫",
        "skipped": "⏭️",
    }
    return status_map.get(status.lower(), "ℹ️")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_progress_bar(current: int, total: int, width: int = 20) -> str:
    """Create ASCII progress bar."""
    if total == 0:
        return "[" + " " * width + "]"
    
    filled = int((current / total) * width)
    bar = "█" * filled + "░" * (width - filled)
    percentage = (current / total) * 100
    
    return f"[{bar}] {percentage:.0f}%"


def validate_model_id(model_id: str) -> bool:
    """Validate model ID format."""
    if not model_id:
        return False
    
    # Check for provider/model format
    if "/" in model_id:
        parts = model_id.split("/")
        return len(parts) == 2 and all(p.strip() for p in parts)
    
    # Allow standalone model names
    return bool(model_id.strip())


def normalize_model_name(model_id: str) -> str:
    """Normalize model name for display."""
    if "/" in model_id:
        return model_id.split("/")[1]
    return model_id


def get_model_provider(model_id: str) -> str:
    """Extract provider from model ID."""
    if "/" in model_id:
        return model_id.split("/")[0]
    
    # Auto-detect from model name
    model_lower = model_id.lower()
    if "gpt" in model_lower:
        return "openai"
    elif "gemini" in model_lower:
        return "google"
    elif "claude" in model_lower:
        return "anthropic"
    elif "mistral" in model_lower:
        return "mistralai"
    else:
        return "unknown"

