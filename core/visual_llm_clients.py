"""
Visual LLM API clients for Test 6: Visual LLM Model Comparison and Artifact Detection.

This module provides unified API clients for:
- GPT-5 Vision (OpenAI) - via OpenRouter or native API
- Gemini 2.5 Vision (Google) - via OpenRouter or native API
- Claude 4.5 Vision (Anthropic via OpenRouter)
- Llama 3.2 Vision (Meta via OpenRouter)

All clients support:
- Image analysis from file paths or URLs
- Structured output for artifact detection
- Cost tracking integration
- Error handling and fallback logic
- Automatic model discovery from OpenRouter
"""

import asyncio
import base64
import json
import re
import os
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
import httpx
import streamlit as st
from openai import AsyncOpenAI
from google import genai
from google.genai import types

from core.models import VisualLLMAnalysis
from core.dynamic_visual_analysis import (
    DynamicVisualLLMAnalysis,
    parse_dynamic_visual_response
)
from core.vision_model_discovery import (
    get_recommended_vision_models,
    get_vision_model_info
)



STRUCTURED_OUTPUT_INSTRUCTIONS = """Return your analysis as a JSON object with the following keys:
{
  \"movement_rating\": number 1-5 or null,
  \"visual_quality_rating\": number 1-5 or null,
  \"artifact_presence_rating\": number 1-5 or null,
  \"detected_artifacts\": array of strings,
  \"confidence\": number between 0 and 1,
  \"rationale\": string explanation
}
Use null for any rating you cannot provide. Confidence must be between 0 and 1. Do not include any text outside of the JSON."""
# Module-level configuration
_CONFIG = {}
_CONFIG = {}


def _format_structured_prompt(prompt: str) -> str:
    """Append structured output instructions if not already present."""
    prompt = (prompt or "").strip()
    if STRUCTURED_OUTPUT_INSTRUCTIONS.strip() in prompt:
        return prompt
    return f"{prompt}\n\n{STRUCTURED_OUTPUT_INSTRUCTIONS.strip()}"


def _parse_visual_analysis(raw_content: str, model_name: str) -> VisualLLMAnalysis:
    """Parse JSON returned by vision models into VisualLLMAnalysis."""
    from core.rating_extractor import parse_visual_llm_response

    raw = (raw_content or "").strip()

    # Try to extract JSON from markdown code blocks
    if raw.startswith("```"):
        # Remove code fence markers
        lines = raw.split("\n")
        # Remove first line (```json or ```)
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    def _try_load(candidate: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(candidate)
        except Exception:
            return None

    # Try to parse as JSON
    data = _try_load(raw)

    # If that fails, try to extract JSON object from text
    if data is None:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = _try_load(match.group(0))

    # If we successfully parsed JSON, convert to VisualLLMAnalysis
    if isinstance(data, dict):
        payload = data.copy()
        payload.setdefault("movement_rating", None)
        payload.setdefault("visual_quality_rating", None)
        payload.setdefault("artifact_presence_rating", None)
        artifacts = payload.get("detected_artifacts", [])
        if not isinstance(artifacts, list):
            artifacts = [str(artifacts)] if artifacts else []
        payload["detected_artifacts"] = [str(a) for a in artifacts]

        # Convert ratings to float or None
        for key in ("movement_rating", "visual_quality_rating", "artifact_presence_rating"):
            value = payload.get(key)
            try:
                payload[key] = None if value is None else float(value)
            except Exception:
                payload[key] = None

        # Ensure confidence is in valid range
        try:
            conf = float(payload.get("confidence", 0.0))
            # If confidence is > 1, assume it's a percentage
            if conf > 1.0:
                conf = conf / 100.0
            payload["confidence"] = max(0.0, min(1.0, conf))
        except Exception:
            payload["confidence"] = 0.0

        payload["rationale"] = str(payload.get("rationale", ""))
        payload["model_name"] = model_name
        payload["raw_response"] = raw_content

        try:
            return VisualLLMAnalysis.model_validate(payload)
        except Exception as e:
            # Log validation error for debugging
            print(f"Validation error for {model_name}: {e}")
            print(f"Payload: {payload}")

    # If JSON parsing failed, fall back to text extraction
    if st.session_state.get('debug_mode', False):
        st.warning(f"⚠️ JSON parsing failed for {model_name}, falling back to text extraction")
        with st.expander(f"Debug: Raw response from {model_name}", expanded=False):
            st.code(raw_content[:500])  # Show first 500 chars

    return parse_visual_llm_response(raw_content, model_name)
def configure(context: Dict[str, Any]) -> None:
    """Configure module with global variables from main app."""
    _CONFIG.clear()
    _CONFIG.update(context)


async def retry_with_exponential_backoff(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    model_name: str = "Model"
) -> Any:
    """
    Retry an async function with exponential backoff.

    Handles:
    - 429 Too Many Requests (rate limiting)
    - 500-599 Server errors
    - Network timeouts
    - Connection errors

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        model_name: Name of model for logging

    Returns:
        Result from successful function call

    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()

        except httpx.HTTPStatusError as e:
            last_exception = e
            status_code = e.response.status_code

            # Don't retry on client errors (except 429)
            if 400 <= status_code < 500 and status_code != 429:
                raise

            # Calculate delay with exponential backoff
            if attempt < max_retries:
                delay = min(initial_delay * (exponential_base ** attempt), max_delay)

                # Add jitter to prevent thundering herd
                import random
                delay = delay * (0.5 + random.random())

                if status_code == 429:
                    st.warning(f"⏳ {model_name}: Rate limited (429). Retrying in {delay:.1f}s... (Attempt {attempt + 1}/{max_retries})")
                else:
                    st.warning(f"⏳ {model_name}: Server error ({status_code}). Retrying in {delay:.1f}s... (Attempt {attempt + 1}/{max_retries})")

                await asyncio.sleep(delay)
            else:
                st.error(f"❌ {model_name}: Failed after {max_retries} retries (Status: {status_code})")
                raise

        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadTimeout) as e:
            last_exception = e

            if attempt < max_retries:
                delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                import random
                delay = delay * (0.5 + random.random())

                st.warning(f"⏳ {model_name}: Connection error. Retrying in {delay:.1f}s... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
            else:
                st.error(f"❌ {model_name}: Connection failed after {max_retries} retries")
                raise

        except Exception as e:
            last_exception = e
            error_str = str(e).lower()

            # Check if it's a rate limit error (various formats)
            if any(keyword in error_str for keyword in ['rate limit', 'too many requests', '429', 'quota']):
                if attempt < max_retries:
                    delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                    import random
                    delay = delay * (0.5 + random.random())

                    st.warning(f"⏳ {model_name}: Rate limited. Retrying in {delay:.1f}s... (Attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    st.error(f"❌ {model_name}: Rate limit exceeded after {max_retries} retries")
                    raise

            # Check if it's a server error
            elif any(keyword in error_str for keyword in ['server error', '500', '502', '503', '504']):
                if attempt < max_retries:
                    delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                    import random
                    delay = delay * (0.5 + random.random())

                    st.warning(f"⏳ {model_name}: Server error. Retrying in {delay:.1f}s... (Attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    st.error(f"❌ {model_name}: Server error after {max_retries} retries")
                    raise

            else:
                # Don't retry on unexpected errors
                st.error(f"❌ {model_name}: Error - {str(e)}")
                raise

    # Should never reach here, but just in case
    if last_exception:
        raise last_exception


def get_default_vision_models() -> Dict[str, str]:
    """
    Get default vision models for each provider.
    Uses cached OpenRouter model discovery.

    Returns:
        Dict mapping provider names to model IDs
    """
    return get_recommended_vision_models()


def _encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def _get_image_mime_type(image_path: str) -> str:
    """Determine MIME type from file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp'
    }
    return mime_types.get(ext, 'image/jpeg')


async def analyze_image_with_gpt5_vision(
    image_path: str,
    prompt: str,
    model: str = None,
    openai_api_key: str = None
) -> VisualLLMAnalysis:
    """
    Analyze image using GPT-5 Vision (via OpenAI API).

    Args:
        image_path: Path to image file
        prompt: Analysis prompt
        model: OpenAI vision model to use (defaults to recommended model from OpenRouter discovery)
        openai_api_key: OpenAI API key

    Returns:
        VisualLLMAnalysis object with structured results
    """
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not set")

    # Use recommended model if not specified
    if model is None:
        recommended = get_default_vision_models()
        model = recommended.get("openai", "gpt-5-nano")
        # Strip provider prefix if present (OpenAI API uses native format)
        if "/" in model:
            model = model.split("/")[1]

    client = AsyncOpenAI(api_key=openai_api_key)
    
    # Encode image
    base64_image = _encode_image_to_base64(image_path)
    mime_type = _get_image_mime_type(image_path)
    structured_prompt = _format_structured_prompt(prompt)

    # Build messages
    messages = [
        {
            "role": "system",
            "content": "You are an expert visual QA analyst. Follow the instructions precisely."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": structured_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    
    # Define API call function for retry wrapper
    async def make_api_call():
        return await client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=16000,  # Increased for detailed visual analysis
            response_format={"type": "json_object"}
        )

    # Call API with retry logic
    # Note: GPT-5 models use max_completion_tokens instead of max_tokens
    # Note: GPT-5 models only support temperature=1 (default)
    response = await retry_with_exponential_backoff(
        make_api_call,
        max_retries=3,
        initial_delay=2.0,
        model_name=f"GPT-5 Vision ({model})"
    )

    # Track cost
    if st.session_state.get('cost_tracker'):
        from core.pricing import custom_openai_price_lookup
        st.session_state.cost_tracker.update(
            provider="OpenAI",
            model=model,
            api="chat.completions",
            raw_response_obj=response,
            pricing_resolver=custom_openai_price_lookup
        )

    # Parse response
    content = response.choices[0].message.content
    return _parse_visual_analysis(content, f"GPT-5 Vision ({model})")


async def analyze_image_with_gemini_vision(
    image_path: str,
    prompt: str,
    model: str = None,
    gemini_api_key: str = None
) -> VisualLLMAnalysis:
    """
    Analyze image using Gemini 2.5 Vision.

    Args:
        image_path: Path to image file
        prompt: Analysis prompt
        model: Gemini vision model to use (defaults to recommended model from OpenRouter discovery)
        gemini_api_key: Gemini API key

    Returns:
        VisualLLMAnalysis object with structured results
    """
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not set")

    # Use recommended model if not specified
    if model is None:
        recommended = get_default_vision_models()
        model = recommended.get("google", "gemini-2.5-flash-lite")
        # Strip provider prefix if present (Gemini API uses native format)
        if "/" in model:
            model = model.split("/")[1]

    client = genai.Client(api_key=gemini_api_key)

    # Upload image file
    with open(image_path, 'rb') as f:
        image_data = f.read()

    mime_type = _get_image_mime_type(image_path)

    # Add structured output instructions to prompt
    structured_prompt = _format_structured_prompt(prompt)

    # Create content with image
    contents = [
        types.Part(text=structured_prompt),
        types.Part(inline_data=types.Blob(data=image_data, mime_type=mime_type))
    ]

    # Define API call function for retry wrapper
    async def make_api_call():
        return await asyncio.to_thread(
            lambda: client.models.generate_content(
                model=model,
                contents=contents
            )
        )

    # Call API with retry logic
    response = await retry_with_exponential_backoff(
        make_api_call,
        max_retries=3,
        initial_delay=2.0,
        model_name=f"Gemini 2.5 Vision ({model})"
    )

    # Track cost
    if st.session_state.get('cost_tracker'):
        from core.pricing import custom_gemini_price_lookup
        st.session_state.cost_tracker.update(
            provider="Google",
            model=model,
            api="generate_content",
            raw_response_obj=response,
            pricing_resolver=custom_gemini_price_lookup
        )

    content = response.text

    return _parse_visual_analysis(content, f"Gemini 2.5 Vision ({model})")


async def analyze_image_with_claude_vision(
    image_path: str,
    prompt: str,
    model: str = None,
    openrouter_api_key: str = None
) -> VisualLLMAnalysis:
    """
    Analyze image using Claude 4.5 Vision via OpenRouter.

    Args:
        image_path: Path to image file
        prompt: Analysis prompt
        model: Claude vision model to use (defaults to recommended model from OpenRouter discovery)
        openrouter_api_key: OpenRouter API key

    Returns:
        VisualLLMAnalysis object with structured results
    """
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    # Use recommended model if not specified
    if model is None:
        recommended = get_default_vision_models()
        model = recommended.get("anthropic", "anthropic/claude-3.5-sonnet")
    
    # Encode image
    base64_image = _encode_image_to_base64(image_path)
    mime_type = _get_image_mime_type(image_path)
    
    # Build messages
    structured_prompt = _format_structured_prompt(prompt)
    messages = [
        {
            "role": "system",
            "content": STRUCTURED_OUTPUT_INSTRUCTIONS
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": structured_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    
    # Define API call function for retry wrapper
    async def make_api_call():
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 16000,  # Increased for detailed visual analysis
                }
            )
            response.raise_for_status()
            return response.json()

    # Call OpenRouter API with retry logic
    data = await retry_with_exponential_backoff(
        make_api_call,
        max_retries=3,
        initial_delay=2.0,
        model_name=f"Claude 4.5 Vision ({model})"
    )

    # Validate response structure
    if 'choices' not in data or not data['choices']:
        raise ValueError(f"Invalid API response: {data}")

    if 'message' not in data['choices'][0] or 'content' not in data['choices'][0]['message']:
        raise ValueError(f"Invalid response structure: {data}")

    # Track cost
    if st.session_state.get('cost_tracker'):
        from core.pricing import custom_openrouter_price_lookup
        st.session_state.cost_tracker.update(
            provider="OpenRouter",
            model=model,
            api="chat.completions",
            raw_response_obj=data,
            pricing_resolver=custom_openrouter_price_lookup
        )

    content = data['choices'][0]['message']['content']

    return _parse_visual_analysis(content, f"Claude 4.5 Vision ({model})")


async def analyze_image_with_llama_vision(
    image_path: str,
    prompt: str,
    model: str = None,
    openrouter_api_key: str = None
) -> VisualLLMAnalysis:
    """
    Analyze image using Llama 3.2 Vision via OpenRouter.

    Args:
        image_path: Path to image file
        prompt: Analysis prompt
        model: Llama vision model to use (defaults to recommended model from OpenRouter discovery)
        openrouter_api_key: OpenRouter API key

    Returns:
        VisualLLMAnalysis object with structured results
    """
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    # Use recommended model if not specified
    if model is None:
        recommended = get_default_vision_models()
        model = recommended.get("meta-llama", "meta-llama/llama-3.2-90b-vision-instruct")

    # Encode image
    base64_image = _encode_image_to_base64(image_path)
    mime_type = _get_image_mime_type(image_path)

    # Build messages
    structured_prompt = _format_structured_prompt(prompt)
    messages = [
        {
            "role": "system",
            "content": STRUCTURED_OUTPUT_INSTRUCTIONS
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": structured_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    # Define API call function for retry wrapper
    async def make_api_call():
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 16000,  # Increased for detailed visual analysis
                }
            )
            response.raise_for_status()
            return response.json()

    # Call OpenRouter API with retry logic
    data = await retry_with_exponential_backoff(
        make_api_call,
        max_retries=3,
        initial_delay=2.0,
        model_name=f"Llama 3.2 Vision ({model})"
    )

    # Validate response structure
    if 'choices' not in data or not data['choices']:
        raise ValueError(f"Invalid API response: {data}")

    if 'message' not in data['choices'][0] or 'content' not in data['choices'][0]['message']:
        raise ValueError(f"Invalid response structure: {data}")

    # Track cost
    if st.session_state.get('cost_tracker'):
        from core.pricing import custom_openrouter_price_lookup
        st.session_state.cost_tracker.update(
            provider="OpenRouter",
            model=model,
            api="chat.completions",
            raw_response_obj=data,
            pricing_resolver=custom_openrouter_price_lookup
        )

    content = data['choices'][0]['message']['content']

    return _parse_visual_analysis(content, f"Llama 3.2 Vision ({model})")


async def analyze_image_multi_model(
    image_path: str,
    prompt: str,
    selected_models: List[str],
    openai_api_key: str = None,
    gemini_api_key: str = None,
    openrouter_api_key: str = None
) -> Dict[str, VisualLLMAnalysis]:
    """
    Analyze image with multiple visual LLM models in parallel.
    Uses recommended models from OpenRouter discovery (cached locally).

    Args:
        image_path: Path to image file
        prompt: Analysis prompt
        selected_models: List of model identifiers to use ("gpt-5-mini", "gemini", "claude", "llama")
        openai_api_key: OpenAI API key
        gemini_api_key: Gemini API key
        openrouter_api_key: OpenRouter API key

    Returns:
        Dictionary mapping model names to VisualLLMAnalysis results
    """
    tasks = []
    model_names = []

    # Get recommended models (uses cached OpenRouter discovery)
    recommended = get_default_vision_models()

    # Map model identifiers to API calls (using recommended models)
    model_mapping = {
        "gpt5": ("GPT-5 Vision", lambda: analyze_image_with_gpt5_vision(
            image_path, prompt, None, openai_api_key  # None = use recommended
        )),
        "gemini": ("Gemini 2.5 Vision", lambda: analyze_image_with_gemini_vision(
            image_path, prompt, None, gemini_api_key  # None = use recommended
        )),
        "claude": ("Claude 4.5 Vision", lambda: analyze_image_with_claude_vision(
            image_path, prompt, None, openrouter_api_key  # None = use recommended
        )),
        "llama": ("Llama 3.2 Vision", lambda: analyze_image_with_llama_vision(
            image_path, prompt, None, openrouter_api_key  # None = use recommended
        ))
    }

    # Create tasks for selected models
    for model_id in selected_models:
        if model_id in model_mapping:
            name, task_fn = model_mapping[model_id]
            model_names.append(name)
            tasks.append(task_fn())

    # Run all tasks in parallel with error handling
    results = {}
    task_results = await asyncio.gather(*tasks, return_exceptions=True)

    for name, result in zip(model_names, task_results):
        if isinstance(result, Exception):
            # Log error and create error result
            st.warning(f"⚠️ {name} failed: {str(result)}")
            results[name] = VisualLLMAnalysis(
                model_name=name,
                detected_artifacts=[],
                confidence=0.0,
                rationale=f"Error: {str(result)}",
                raw_response=None
            )
        else:
            results[name] = result

    return results


async def analyze_image_multi_model_dynamic(
    image_path: str,
    prompt: str,
    selected_models: List[str],
    openai_api_key: str = None,
    gemini_api_key: str = None,
    openrouter_api_key: str = None
) -> Dict[str, DynamicVisualLLMAnalysis]:
    """
    Analyze image with multiple visual LLM models using DYNAMIC schema.

    This version does NOT enforce any predefined schema - it accepts
    whatever fields the LLM returns and categorizes them automatically.

    Args:
        image_path: Path to image file
        prompt: Analysis prompt (should NOT specify output format)
        selected_models: List of model identifiers
        openai_api_key: OpenAI API key
        gemini_api_key: Gemini API key
        openrouter_api_key: OpenRouter API key

    Returns:
        Dictionary mapping model names to DynamicVisualLLMAnalysis results
    """
    # First get regular results
    regular_results = await analyze_image_multi_model(
        image_path=image_path,
        prompt=prompt,
        selected_models=selected_models,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        openrouter_api_key=openrouter_api_key
    )

    # Convert to dynamic analysis objects
    dynamic_results = {}

    for model_name, analysis in regular_results.items():
        # Extract all fields from the VisualLLMAnalysis object
        if hasattr(analysis, 'raw_response') and analysis.raw_response:
            # Parse the raw response dynamically
            dynamic_results[model_name] = parse_dynamic_visual_response(
                raw_response=analysis.raw_response,
                model_name=model_name
            )
        else:
            # Convert existing fields to dynamic format
            fields = {}
            if hasattr(analysis, 'movement_rating') and analysis.movement_rating is not None:
                fields['movement_rating'] = analysis.movement_rating
            if hasattr(analysis, 'visual_quality_rating') and analysis.visual_quality_rating is not None:
                fields['visual_quality_rating'] = analysis.visual_quality_rating
            if hasattr(analysis, 'artifact_presence_rating') and analysis.artifact_presence_rating is not None:
                fields['artifact_presence_rating'] = analysis.artifact_presence_rating
            if hasattr(analysis, 'detected_artifacts'):
                fields['detected_artifacts'] = analysis.detected_artifacts
            if hasattr(analysis, 'confidence'):
                fields['confidence'] = analysis.confidence
            if hasattr(analysis, 'rationale'):
                fields['rationale'] = analysis.rationale

            dynamic_results[model_name] = DynamicVisualLLMAnalysis(
                model_name=model_name,
                raw_response=getattr(analysis, 'raw_response', ''),
                **fields
            )

    return dynamic_results


def build_vr_avatar_analysis_prompt(
    artifact_types: List[str] = None
) -> str:
    """
    Build analysis prompt for VR avatar artifact detection (Mode A).

    Args:
        artifact_types: List of specific artifact types to look for

    Returns:
        Formatted prompt string
    """
    if artifact_types is None:
        artifact_types = [
            "red lines in eyes",
            "finger movement issues",
            "feet not moving",
            "avatar distortions",
            "clothing distortions during movement"
        ]

    artifact_list = "\n".join([f"- {artifact}" for artifact in artifact_types])


    base_prompt = f"""Analyze this VR avatar image/video for visual artifacts and quality issues.
    return _format_structured_prompt(base_prompt)

Please evaluate the following aspects on a scale of 1-5 (1=worst, 5=best):

1. **Movement Quality** (1-5): How natural and smooth are the avatar's movements?
2. **Visual Quality** (1-5): Overall visual fidelity and rendering quality
3. **Artifact Presence** (1-5): Absence of visual artifacts (5=no artifacts, 1=severe artifacts)

Common artifacts to look for:
{artifact_list}

Provide your analysis in the following format:
- Movement Rating: [1-5]
- Visual Quality Rating: [1-5]
- Artifact Presence Rating: [1-5]
- Detected Artifacts: [list any artifacts you observe]
- Confidence: [0.0-1.0]
- Rationale: [detailed explanation of your analysis]
"""


def build_general_visual_analysis_prompt(
    task_description: str = "general visual analysis"
) -> str:
    """
    Build analysis prompt for general visual LLM comparison (Mode B).

    Args:
        task_description: Description of the analysis task

    Returns:
        Formatted prompt string
    """

    base_prompt = f"""Analyze this image for: {task_description}
    return _format_structured_prompt(base_prompt)

Please provide:
1. **Object/Artifact Detection**: What objects or artifacts do you see?
2. **Classification**: What category does this image belong to?
3. **Confidence Score**: How confident are you in your analysis? (0.0-1.0)
4. **Detailed Description**: Provide a comprehensive description of what you observe

Format your response clearly with these sections.
"""


