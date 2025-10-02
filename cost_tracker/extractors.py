# cost_tracker/extractors.py
"""
Built-in usage extractors for common LLM providers.
Import and register these at app startup to handle provider-specific response formats.
"""

from .tracker import register_usage_extractor


def anthropic_extractor(provider, model, raw_obj, raw_json):
    """
    Extract usage from Anthropic Messages API response.
    Anthropic returns usage.input_tokens / usage.output_tokens on message responses.
    Docs: https://docs.anthropic.com/api/messages
    """
    u = getattr(raw_obj, "usage", None) or {}
    try:
        return {
            "prompt_tokens": int(getattr(u, "input_tokens", 0) or 0),
            "completion_tokens": int(getattr(u, "output_tokens", 0) or 0),
            "total_tokens": int((getattr(u, "input_tokens", 0) or 0) + 
                               (getattr(u, "output_tokens", 0) or 0))
        }
    except Exception:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def cohere_extractor(provider, model, raw_obj, raw_json):
    """
    Extract usage from Cohere API response.
    Cohere provides usage under meta.billed_units.{input_tokens, output_tokens} for chat/stream.
    Docs: https://docs.cohere.com/reference/chat-stream
    """
    meta = (raw_json or {}).get("meta") or {}
    b = meta.get("billed_units") or meta.get("billedUnits") or {}
    pt = int(b.get("input_tokens", 0) or 0)
    ct = int(b.get("output_tokens", 0) or 0)
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}


def groq_extractor(provider, model, raw_obj, raw_json):
    """
    Extract usage from Groq API response.
    Groq uses an OpenAI-compatible schema; responses include usage.* on chat completions.
    Docs: https://console.groq.com/docs/api-reference
    """
    u = (raw_json or {}).get("usage", {}) if isinstance(raw_json, dict) else {}
    return {
        "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(u.get("completion_tokens", 0) or 0),
        "total_tokens": int(u.get("total_tokens", 0) or 0),
    }


def mistral_extractor(provider, model, raw_obj, raw_json):
    """
    Extract usage from Mistral API response.
    Mistral's platform returns OpenAI-style usage.* on chat APIs.
    Docs: https://docs.mistral.ai/api/
    """
    u = (raw_json or {}).get("usage", {}) if isinstance(raw_json, dict) else {}
    return {
        "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(u.get("completion_tokens", 0) or 0),
        "total_tokens": int(u.get("total_tokens", 0) or 0),
    }


def openai_extractor(provider, model, raw_obj, raw_json):
    """
    Extract usage from OpenAI API response.
    OpenAI returns usage.prompt_tokens / usage.completion_tokens / usage.total_tokens.
    Docs: https://platform.openai.com/docs/api-reference/chat/object
    """
    # Try JSON first
    if isinstance(raw_json, dict) and "usage" in raw_json:
        u = raw_json.get("usage") or {}
        return {
            "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(u.get("completion_tokens", 0) or 0),
            "total_tokens": int(u.get("total_tokens", 0) or 0),
        }
    
    # Try object
    u = getattr(raw_obj, "usage", None)
    if u is not None:
        return {
            "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(u, "total_tokens", 0) or 0),
        }
    
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def gemini_extractor(provider, model, raw_obj, raw_json):
    """
    Extract usage from Google Gemini API response.
    Gemini objects: usage_metadata.promptTokenCount / candidatesTokenCount / totalTokenCount
    Docs: https://ai.google.dev/api/generate-content
    """
    um = getattr(raw_obj, "usage_metadata", None)
    if um is not None:
        pt = int(getattr(um, "promptTokenCount", 0) or getattr(um, "prompt_token_count", 0) or 0)
        ct = int(getattr(um, "candidatesTokenCount", 0) or getattr(um, "candidates_token_count", 0) or 0)
        tt = int(getattr(um, "totalTokenCount", 0) or getattr(um, "total_token_count", 0) or (pt + ct))
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}
    
    # Try JSON format
    if isinstance(raw_json, dict):
        um = raw_json.get("usage_metadata") or raw_json.get("usageMetadata") or {}
        pt = int(um.get("promptTokenCount", 0) or um.get("prompt_token_count", 0) or 0)
        ct = int(um.get("candidatesTokenCount", 0) or um.get("candidates_token_count", 0) or 0)
        tt = int(um.get("totalTokenCount", 0) or um.get("total_token_count", 0) or (pt + ct))
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}
    
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def openrouter_extractor(provider, model, raw_obj, raw_json):
    """
    Extract usage from OpenRouter API response.
    OpenRouter uses OpenAI-compatible format with usage.* fields.
    Docs: https://openrouter.ai/docs
    """
    u = (raw_json or {}).get("usage", {}) if isinstance(raw_json, dict) else {}
    return {
        "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(u.get("completion_tokens", 0) or 0),
        "total_tokens": int(u.get("total_tokens", 0) or 0),
    }


def together_extractor(provider, model, raw_obj, raw_json):
    """
    Extract usage from Together AI API response.
    Together uses OpenAI-compatible format.
    Docs: https://docs.together.ai/reference/completions-1
    """
    u = (raw_json or {}).get("usage", {}) if isinstance(raw_json, dict) else {}
    return {
        "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(u.get("completion_tokens", 0) or 0),
        "total_tokens": int(u.get("total_tokens", 0) or 0),
    }


def perplexity_extractor(provider, model, raw_obj, raw_json):
    """
    Extract usage from Perplexity API response.
    Perplexity uses OpenAI-compatible format.
    Docs: https://docs.perplexity.ai/reference/post_chat_completions
    """
    u = (raw_json or {}).get("usage", {}) if isinstance(raw_json, dict) else {}
    return {
        "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(u.get("completion_tokens", 0) or 0),
        "total_tokens": int(u.get("total_tokens", 0) or 0),
    }


def ollama_extractor(provider, model, raw_obj, raw_json):
    """
    Extract usage from Ollama API response.
    Ollama returns prompt_eval_count and eval_count in the response.
    Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
    """
    if isinstance(raw_json, dict):
        pt = int(raw_json.get("prompt_eval_count", 0) or 0)
        ct = int(raw_json.get("eval_count", 0) or 0)
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def register_all_extractors():
    """
    Register all built-in extractors.
    Call this once at app startup to enable automatic usage extraction for all supported providers.
    """
    register_usage_extractor("Anthropic", anthropic_extractor)
    register_usage_extractor("Cohere", cohere_extractor)
    register_usage_extractor("Groq", groq_extractor)
    register_usage_extractor("Mistral", mistral_extractor)
    register_usage_extractor("OpenAI", openai_extractor)
    register_usage_extractor("Google", gemini_extractor)
    register_usage_extractor("Gemini", gemini_extractor)
    register_usage_extractor("OpenRouter", openrouter_extractor)
    register_usage_extractor("Together", together_extractor)
    register_usage_extractor("Perplexity", perplexity_extractor)
    register_usage_extractor("Ollama", ollama_extractor)

