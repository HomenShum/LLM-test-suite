"""
API client functions for classification and data generation.
Extracted from streamlit_test_v5.py to reduce main file size.

This module contains:
- Classification functions for OpenAI, Gemini, Ollama, OpenRouter
- Structured JSON helpers
- Text generation functions
- Judge and pruner functions
- Synthetic data generation
"""

import json
import asyncio
import time
import pandas as pd
import streamlit as st
import httpx
from openai import AsyncOpenAI
from google import genai
from google.genai import types
from typing import List, Dict, Any, Optional, Tuple
from asyncio import Semaphore

# Import models
from core.models import (
    Classification,
    ClassificationWithConf,
    SyntheticDataItem,
    ToolCallSequenceItem,
    PruningDataItem
)

# Import pricing
from core.pricing import (
    custom_openrouter_price_lookup,
    custom_gemini_price_lookup,
    _to_openrouter_model_id,
    _to_native_model_id
)

# Import helpers
from utils.helpers import _retry
from utils.data_helpers import _allowed_labels, _normalize_label
from config.scenarios import SKELETON_COLUMNS


# Rate limiter for API calls
_rate_limiter = Semaphore(10)  # Max 10 concurrent API calls

# API configuration (will be imported from main file's globals)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


_CONFIG = {}

def configure(context: Dict[str, Any]) -> None:
    """Store API client configuration values."""
    _CONFIG.clear()
    _CONFIG.update(context)


def combined_price_lookup(provider: str, model: str) -> Optional[Dict[str, float]]:
    """Combined price lookup for all providers."""
    if provider.lower() == "google":
        return custom_gemini_price_lookup(provider, model)
    else:
        return custom_openrouter_price_lookup(provider, model)


async def classify_with_openai(text: str, allowed: List[str], 
                               api_routing_mode: str = "openrouter",
                               openai_api_key: str = None,
                               openai_model: str = "gpt-5-mini") -> ClassificationWithConf:
    """
    Classify text using OpenAI with structured output.
    Routes through OpenRouter or native API based on api_routing_mode.
    """
    # Determine selected model (per-test override takes precedence)
    selected_oai_model = st.session_state.get('openai_model_override', openai_model)

    # Route through OpenRouter if configured
    if api_routing_mode == "openrouter":
        openrouter_key = st.session_state.get("OPENROUTER_API_KEY") or _CONFIG.get("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", "")
        return await classify_with_openrouter(
            text, allowed,
            model=_to_openrouter_model_id(selected_oai_model, "openai"),
            openrouter_api_key=openrouter_key
        )

    # Use native OpenAI API
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not set")

    # Convert to native model ID (strip provider prefix if present)
    native_model = _to_native_model_id(selected_oai_model)

    async with _rate_limiter:
        client = AsyncOpenAI(api_key=openai_api_key)
        allowed_hint = f"Allowed labels: {allowed if allowed else '[unconstrained]'}.\nPick exactly ONE of these values in 'classification_result'."
        try:
            resp = await client.chat.completions.parse(
                model=native_model,
                messages=[
                    {"role": "system", "content": "Return a structured classification with optional confidence 0..1."},
                    {"role": "user", "content": f"{allowed_hint}\nText: {text}\nRespond as JSON with keys: classification_result, rationale, confidence (0..1 optional)."}
                ],
                response_format=ClassificationWithConf
            )
            # Track the API call
            if hasattr(st.session_state, 'cost_tracker') and st.session_state.cost_tracker:
                st.session_state.cost_tracker.update(
                    provider="OpenAI", model=native_model, api="chat.completions.parse",
                    raw_response_obj=resp, pricing_resolver=combined_price_lookup
                )
            parsed = resp.choices[0].message.parsed
            return parsed if isinstance(parsed, ClassificationWithConf) else ClassificationWithConf.model_validate(parsed)
        except Exception:
            comp = await client.chat.completions.create(
                model=native_model,
                messages=[
                    {"role": "system", "content": "Respond ONLY with JSON having keys classification_result, rationale, confidence (0..1 optional)."},
                    {"role": "user", "content": f"{allowed_hint}\nText: {text}"}
                ],
            )
            # Track the fallback API call
            if hasattr(st.session_state, 'cost_tracker') and st.session_state.cost_tracker:
                st.session_state.cost_tracker.update(
                    provider="OpenAI", model=native_model, api="chat.completions.create",
                    raw_response_obj=comp, pricing_resolver=combined_price_lookup
                )
            content = comp.choices[0].message.content or "{}"
            if content.strip().startswith("```"):
                    if content.strip().startswith("```"):
                        content = content.strip().split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(content)
            if "confidence" in data:
                try:
                    data["confidence"] = float(data["confidence"])
                except Exception:
                    data["confidence"] = None
            return ClassificationWithConf.model_validate(data)


async def openai_structured_json(client: AsyncOpenAI, model: str, system: str, user_jsonable: Any) -> Dict[str, Any]:
    """Helper for OpenAI structured JSON output."""
    # Ensure the system prompt includes an explicit request for JSON
    enhanced_system = system + " Respond ONLY with the requested JSON object."

    # Always use the native model ID with the OpenAI SDK
    native_model = _to_native_model_id(model)

    comp = await client.chat.completions.create(
        model=native_model,
        messages=[
            {"role":"system","content":enhanced_system},
            {"role":"user","content":json.dumps(user_jsonable, indent=2)}
        ],
        response_format={"type":"json_object"}
    )
    # Track the API call
    if hasattr(st.session_state, 'cost_tracker') and st.session_state.cost_tracker:
        st.session_state.cost_tracker.update(
            provider="OpenAI", model=native_model, api="chat.completions.create",
            raw_response_obj=comp, pricing_resolver=combined_price_lookup
        )
    content = comp.choices[0].message.content or "{}"
    return json.loads(content)


async def classify_with_gemini(text: str, allowed: List[str], model: str,
                               api_routing_mode: str = "openrouter",
                               gemini_api_key: str = None) -> ClassificationWithConf:
    """
    Classify text using Google Gemini with structured output.
    Routes through OpenRouter or native API based on api_routing_mode.
    """
    # Route through OpenRouter if configured
    if api_routing_mode == "openrouter":
        return await classify_with_openrouter(text, allowed, model=_to_openrouter_model_id(model, "google"))

    # Use native Google Genai API
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not set.")

    # Convert to native model ID (strip provider prefix if present)
    native_model = _to_native_model_id(model)

    async with _rate_limiter:
        try:
            client = genai.Client(api_key=gemini_api_key)

            schema_dict = ClassificationWithConf.model_json_schema()

            # System prompt enforces the schema and allowed labels
            system_prompt = f"You are a text classifier. Return ONLY a single JSON object that strictly adheres to the provided schema. The 'classification_result' MUST be one of the following: {allowed if allowed else 'any string'}."

            def sync_api_call():
                return client.models.generate_content(
                    model=native_model,
                    contents=f"{system_prompt}\n\nText to classify: {text}",
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema_dict
                    )
                )

            # Run the synchronous API call in a thread pool
            response = await asyncio.to_thread(sync_api_call)

            # Track cost
            if hasattr(st.session_state, 'cost_tracker') and st.session_state.cost_tracker:
                st.session_state.cost_tracker.update(
                    provider="Google",
                    model=native_model,
                    api="generate_content",
                    raw_response_obj=response,
                    pricing_resolver=custom_gemini_price_lookup
                )

            # Parse the JSON response
            result_text = response.text
            result_data = json.loads(result_text)
            return ClassificationWithConf.model_validate(result_data)

        except Exception as e:
            # Return error as classification result
            return ClassificationWithConf(
                classification_result="",
                rationale=f"Gemini API error: {str(e)}",
                confidence=None
            )


async def classify_with_ollama(base_url: str, model: str, text: str, allowed: List[str]) -> ClassificationWithConf:
    """Classify text using local Ollama."""
    from utils.model_discovery import _normalize_ollama_root
    
    root = _normalize_ollama_root(base_url)
    chat_url = root + "/api/generate"
    
    schema = {
        "type": "object",
        "properties": {
            "classification_result": ({"type": "string", "enum": allowed} if allowed else {"type": "string"}),
            "rationale": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "required": ["classification_result", "rationale"],
        "additionalProperties": False
    }
    
    system_msg = f"Return ONLY compact JSON matching this schema: {json.dumps(schema)}."
    body = {"model": model, "prompt": f"{system_msg}\nText: {text}", "stream": False, "format": "json"}
    
    async with httpx.AsyncClient(timeout=120) as hc:
        r = await hc.post(chat_url, json=body)
        r.raise_for_status()
        data = r.json()
        content = data.get("response", "{}")
        parsed = json.loads(content)
        return ClassificationWithConf.model_validate(parsed)


async def ollama_json(base_url: str, model: str, system: str, user_jsonable: Any) -> Dict[str, Any]:
    """Helper for Ollama structured JSON output."""
    body = {
        "model": model,
        "messages": [
            {"role":"system","content":system},
            {"role":"user","content":json.dumps(user_jsonable, indent=2)}
        ],
        "stream": False,
        "format": "json"
    }
    async with httpx.AsyncClient(timeout=120) as hc:
        from utils.model_discovery import _normalize_ollama_root
        chat_url = _normalize_ollama_root(base_url) + "/api/chat"
        r = await hc.post(chat_url, json=body)
        r.raise_for_status()
        data = r.json()
        content = data.get("message", {}).get("content", "{}")
        return json.loads(content)


async def classify_with_openrouter(text: str, allowed: List[str], model: Optional[str] = None,
                                   openrouter_api_key: str = None,
                                   openrouter_model: str = "mistralai/mistral-large-2411") -> ClassificationWithConf:
    """Classify text using OpenRouter."""
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    if model is None:
        model = openrouter_model

    messages = [
        {"role": "system", "content": (
            "Return ONLY compact JSON with keys: classification_result, rationale, confidence (0..1 optional)." +
            (f" Allowed labels: {allowed}." if allowed else "")
        )},
        {"role": "user", "content": text}
    ]

    headers = {"Authorization": f"Bearer {openrouter_api_key}"}
    body = {"model": model, "messages": messages}

    async with httpx.AsyncClient(timeout=60) as hc:
        r = await hc.post(OPENROUTER_URL, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()

        # Track the API call
        if hasattr(st.session_state, 'cost_tracker') and st.session_state.cost_tracker:
            st.session_state.cost_tracker.update(
                provider="OpenRouter",
                model=model,
                api="chat.completions",
                raw_response_json=data,
                pricing_resolver=custom_openrouter_price_lookup
            )

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        if content.strip().startswith("```"):
            content = content.strip().split("\n", 1)[1].rsplit("```", 1)[0]

        parsed = json.loads(content)
        if "confidence" in parsed:
            try:
                parsed["confidence"] = float(parsed["confidence"])
            except Exception:
                parsed["confidence"] = None

        return ClassificationWithConf.model_validate(parsed)


async def openrouter_json(model: str, system: str, user_jsonable: Any, schema_name: str, schema: Dict[str, Any],
                         openrouter_api_key: str = None) -> Dict[str, Any]:
    """Helper for OpenRouter structured JSON output."""
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_jsonable, indent=2)}
    ]

    headers = {"Authorization": f"Bearer {openrouter_api_key}"}
    body = {
        "model": model,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema
            }
        }
    }

    async with httpx.AsyncClient(timeout=120) as hc:
        r = await hc.post(OPENROUTER_URL, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()

        # Track the API call
        if hasattr(st.session_state, 'cost_tracker') and st.session_state.cost_tracker:
            st.session_state.cost_tracker.update(
                provider="OpenRouter",
                model=model,
                api="chat.completions",
                raw_response_json=data,
                pricing_resolver=custom_openrouter_price_lookup
            )

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        return json.loads(content)


async def _classify_df_async(
    df: pd.DataFrame,
    use_openai: bool,
    use_ollama: bool,
    openrouter_model: str,
    use_ollama_local: bool = False,
    ollama_base_url: str = "",
    ollama_model: str = "",
    third_kind: str = "None",
    third_model: str = ""
) -> pd.DataFrame:
    """Batch classify a DataFrame using the configured providers."""
    labels = _allowed_labels(df)
    for col in SKELETON_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Check session state first (user input), then config, then secrets
    openai_api_key = st.session_state.get("OPENAI_API_KEY") or _CONFIG.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    openrouter_api_key = st.session_state.get("OPENROUTER_API_KEY") or _CONFIG.get("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", "")
    api_routing_mode = _CONFIG.get("API_ROUTING_MODE", "openrouter")
    openai_model = _CONFIG.get("OPENAI_MODEL", "gpt-5-mini")

    async def worker(idx: int, text: str):
        oai_res = {"result": None, "latency": 0.0}
        oll_res = {"result": None, "latency": 0.0}
        thd_res = {"result": None, "latency": 0.0}
        gmn_res = {"result": None, "latency": 0.0}

        async def run_task(task_fn, result_dict):
            start_time = time.monotonic()
            try:
                result_dict["result"] = await _retry(task_fn)
            except Exception as exc:
                result_dict["result"] = ClassificationWithConf(
                    classification_result="",
                    rationale=f"Error: {exc}",
                    confidence=None,
                )
            result_dict["latency"] = time.monotonic() - start_time

        tasks_to_run = []
        if use_openai:
            tasks_to_run.append(run_task(lambda: classify_with_openai(text, labels, api_routing_mode=api_routing_mode, openai_api_key=openai_api_key, openai_model=openai_model), oai_res))

        if use_ollama:
            async def openrouter_task():
                try:
                    return await _retry(lambda: classify_with_openrouter(text, labels, model=openrouter_model, openrouter_api_key=openrouter_api_key))
                except Exception as primary_exc:
                    if use_ollama_local:
                        try:
                            return await _retry(lambda: classify_with_ollama(ollama_base_url, ollama_model, text, labels))
                        except Exception as secondary_exc:
                            raise Exception(
                                f"OpenRouter failed: {primary_exc}; Local Ollama failed: {secondary_exc}"
                            )
                    raise primary_exc

            tasks_to_run.append(run_task(openrouter_task, oll_res))
        elif use_ollama_local:
            tasks_to_run.append(
                run_task(lambda: classify_with_ollama(ollama_base_url, ollama_model, text, labels), oll_res)
            )

        if st.session_state.get('use_gemini'):
            gemini_model = st.session_state.get('gemini_model', 'gemini-2.5-flash')
            tasks_to_run.append(run_task(lambda: classify_with_gemini(text, labels, gemini_model), gmn_res))

        if third_kind != "None":
            async def third_task_fn():
                if third_kind == "OpenAI":
                    native_model = _to_native_model_id(third_model)
                    if not openai_api_key:
                        raise ValueError("OPENAI_API_KEY not set for third model classification")
                    client = AsyncOpenAI(api_key=openai_api_key)
                    comp = await client.chat.completions.create(
                        model=native_model,
                        messages=[
                            {
                                "role": "system",
                                "content": "Respond ONLY with JSON having keys classification_result, rationale, confidence (0..1 optional).",
                            },
                            {"role": "user", "content": text},
                        ],
                    )
                    content = comp.choices[0].message.content or "{}"
                    if content.strip().startswith("```"):
                        content = content.strip().split("\\n", 1)[1].rsplit("```", 1)[0]
                    data = json.loads(content)
                    if "confidence" in data:
                        try:
                            data["confidence"] = float(data["confidence"])
                        except Exception:
                            data["confidence"] = None
                    return ClassificationWithConf.model_validate(data)
                if third_kind == "OpenRouter":
                    return await classify_with_openrouter(text, labels, model=third_model)
                return ClassificationWithConf(
                    classification_result="",
                    rationale="Unsupported third model kind",
                    confidence=None,
                )

            tasks_to_run.append(run_task(third_task_fn, thd_res))

        await asyncio.gather(*tasks_to_run)
        return idx, oai_res, oll_res, thd_res, gmn_res

    tracker = st.session_state.get('execution_tracker')
    test_name = "Classification Test"

    if tracker:
        tracker.emit(
            test_name,
            "start",
            "classification_run",
            "Classification Run",
            "orchestrator",
            total_items=len(df),
        )

    batch_size = 100
    all_coroutines = [worker(idx, q) for idx, row in df.iterrows() if (q := str(row.get("query", "")).strip())]
    if not all_coroutines:
        return df

    progress = st.progress(0.0, text="Running classification...")
    all_results = []
    total_coroutines = len(all_coroutines)

    progress_metadata = {
        'batch_timestamps': [],
        'batch_sizes': [],
        'cumulative_counts': [],
        'batch_latencies': [],
        'success_rates': [],
    }

    for i in range(0, total_coroutines, batch_size):
        batch_num = i // batch_size + 1
        batch_id = f"batch_{batch_num}"
        batch_start = time.time()

        if tracker:
            tracker.emit(
                test_name,
                "start",
                batch_id,
                f"Batch {batch_num}",
                "batch",
                parent_id="classification_run",
                batch_size=min(batch_size, total_coroutines - i),
            )

        batch_coroutines = all_coroutines[i:i + batch_size]
        batch_results = await asyncio.gather(*batch_coroutines)
        all_results.extend(batch_results)

        batch_end = time.time()
        completed_count = len(all_results)
        batch_latency = batch_end - batch_start

        successful = sum(
            1
            for _, oai, oll, thd, gmn in batch_results
            if (oai["result"] or oll["result"] or thd["result"] or gmn["result"])
        )
        success_rate = successful / len(batch_results) if batch_results else 0.0

        progress_metadata['batch_timestamps'].append(batch_end)
        progress_metadata['batch_sizes'].append(len(batch_results))
        progress_metadata['cumulative_counts'].append(completed_count)
        progress_metadata['batch_latencies'].append(batch_latency)
        progress_metadata['success_rates'].append(success_rate)

        if tracker:
            tracker.emit(
                test_name,
                "complete",
                batch_id,
                f"Batch {batch_num}",
                "batch",
                parent_id="classification_run",
                items_processed=len(batch_results),
                success_rate=success_rate,
                latency=batch_latency,
            )

        progress.progress(
            completed_count / total_coroutines,
            text=f"Running classification... ({completed_count}/{total_coroutines})",
        )

    progress.empty()

    if 'last_progress_metadata' not in st.session_state:
        st.session_state.last_progress_metadata = {}
    st.session_state.last_progress_metadata['classification'] = progress_metadata

    if tracker:
        tracker.emit(
            test_name,
            "complete",
            "classification_run",
            "Classification Run",
            "orchestrator",
            total_items=len(df),
            total_batches=len(progress_metadata['batch_timestamps']),
        )

    for idx, oai_res, oll_res, thd_res, gmn_res in all_results:
        if (oll := oll_res["result"]) is not None:
            df.loc[idx, "classification_result_openrouter_mistral"] = _normalize_label(
                str(oll.classification_result or "")
            )
            df.loc[idx, "classification_result_openrouter_mistral_rationale"] = oll.rationale or "(no rationale)"
            df.loc[idx, "classification_result_openrouter_mistral_confidence"] = oll.confidence
            df.loc[idx, "latency_openrouter_mistral"] = oll_res["latency"]
        if (oai := oai_res["result"]) is not None:
            df.loc[idx, "classification_result_openai"] = _normalize_label(str(oai.classification_result or ""))
            df.loc[idx, "classification_result_openai_rationale"] = oai.rationale or "(no rationale)"
            df.loc[idx, "classification_result_openai_confidence"] = oai.confidence
            df.loc[idx, "latency_openai"] = oai_res["latency"]
        if (gmn := gmn_res["result"]) is not None:
            df.loc[idx, "classification_result_gemini"] = _normalize_label(str(gmn.classification_result or ""))
            df.loc[idx, "classification_result_gemini_rationale"] = gmn.rationale or "(no rationale)"
            df.loc[idx, "classification_result_gemini_confidence"] = gmn.confidence
            df.loc[idx, "latency_gemini"] = gmn_res["latency"]
        if (thd := thd_res["result"]) is not None:
            df.loc[idx, "classification_result_third"] = _normalize_label(str(thd.classification_result or ""))
            df.loc[idx, "classification_result_third_rationale"] = thd.rationale or "(no rationale)"
            df.loc[idx, "classification_result_third_confidence"] = thd.confidence
            df.loc[idx, "latency_third"] = thd_res["latency"]

    return df


async def generate_synthetic_data(
    task_prompt: str,
    data_size: int,
    data_type: str,
    generation_model: str
) -> pd.DataFrame:
    """Generate synthetic dataset rows using the configured LLM provider."""

    if data_type == "Classification":
        item_schema = SyntheticDataItem.model_json_schema()
    elif data_type == "Tool/Agent Sequence":
        item_schema = ToolCallSequenceItem.model_json_schema()
    elif data_type == "Context Pruning":
        item_schema = PruningDataItem.model_json_schema()
    else:
        raise ValueError(f"Invalid data type: {data_type}")

    list_wrapper_schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": item_schema}},
        "required": ["items"],
    }

    payload = {
        "required_output_count": data_size,
        "task_description": task_prompt,
        "instructions": (
            f"Generate exactly {data_size} distinct data items, ensuring all output "
            "is wrapped in a JSON object under the key 'items'."
        ),
        "target_schema": list_wrapper_schema,
    }

    st.info(f"Generating {data_size} items using {generation_model}...")

    api_routing_mode = _CONFIG.get("API_ROUTING_MODE", "openrouter")
    # Check session state first (user input), then config, then secrets
    openrouter_api_key = st.session_state.get("OPENROUTER_API_KEY") or _CONFIG.get("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", "")
    openai_api_key = st.session_state.get("OPENAI_API_KEY") or _CONFIG.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

    provider = _get_provider_from_model_id(generation_model)
    system_prompt = (
        "You are a synthetic data generator. Respond ONLY with a single JSON object "
        "matching the requested schema, containing a list of objects under the key 'items'."
    )

    try:
        if api_routing_mode == "openrouter" or provider in {"mistralai", "anthropic", "meta-llama", "deepseek"}:
            if not openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY not set for OpenRouter mode.")
            raw_result = await openrouter_json(
                _to_openrouter_model_id(generation_model),
                system_prompt,
                payload,
                "synthetic_dataset_list",
                list_wrapper_schema,
                openrouter_api_key=openrouter_api_key
            )
        elif provider == "openai":
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not set for OpenAI model.")
            client = AsyncOpenAI(api_key=openai_api_key)
            raw_result = await openai_structured_json(
                client,
                _to_native_model_id(generation_model),
                system_prompt,
                payload,
            )
        elif provider == "google":
            if not openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY required for Gemini models in data generation.")
            raw_result = await openrouter_json(
                _to_openrouter_model_id(generation_model),
                system_prompt,
                payload,
                "synthetic_dataset_list",
                list_wrapper_schema,
                openrouter_api_key=openrouter_api_key
            )
        else:
            if not openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY not set.")
            raw_result = await openrouter_json(
                _to_openrouter_model_id(generation_model),
                system_prompt,
                payload,
                "synthetic_dataset_list",
                list_wrapper_schema,
                openrouter_api_key=openrouter_api_key
            )

        data_list = []
        if isinstance(raw_result, dict):
            data_list = raw_result.get("items", [])
            if not data_list:
                for value in raw_result.values():
                    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                        data_list = value
                        break

        if not data_list:
            raise ValueError("Could not extract a list of structured items from the LLM response.")

        return pd.DataFrame(data_list)

    except Exception as exc:  # pragma: no cover - UI feedback only
        st.error(f"Data generation failed: {exc}")
        return pd.DataFrame()


async def generate_text_async(prompt: str, use_ollama: bool = True, use_openai: bool = True,
                              openrouter_api_key: str = None, openai_api_key: str = None,
                              openrouter_model: str = "mistralai/mistral-large-2411",
                              openai_model: str = "gpt-5-mini") -> str:
    """Generates text using the first available provider (prioritizes OpenRouter)."""
    # Prioritize OpenRouter if enabled
    if use_ollama and openrouter_api_key:
        try:
            headers = {"Authorization": f"Bearer {openrouter_api_key}"}
            messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
            body = {"model": openrouter_model, "messages": messages, "max_tokens": 512}
            async with httpx.AsyncClient(timeout=60) as hc:
                r = await hc.post(OPENROUTER_URL, headers=headers, json=body)
                r.raise_for_status()
                data = r.json()
                # Track the API call
                if hasattr(st.session_state, 'cost_tracker') and st.session_state.cost_tracker:
                    st.session_state.cost_tracker.update(
                        provider="OpenRouter", model=openrouter_model, api="chat.completions",
                        raw_response_json=data, pricing_resolver=custom_openrouter_price_lookup
                    )
                return data.get("choices", [{}])[0].get("message", {}).get("content", "Error: No content.")
        except Exception as e:
            # If OpenRouter fails, fall through to OpenAI if it's enabled
            if not (use_openai and openai_api_key):
                return f"Error during OpenRouter call: {e}"

    # Fallback to OpenAI if enabled
    if use_openai and openai_api_key:
        try:
            client = AsyncOpenAI(api_key=openai_api_key)
            comp = await client.chat.completions.create(model=openai_model, messages=[{"role": "user", "content": prompt}], max_tokens=512)
            # Track the API call
            if hasattr(st.session_state, 'cost_tracker') and st.session_state.cost_tracker:
                st.session_state.cost_tracker.update(
                    provider="OpenAI", model=openai_model, api="chat.completions.create",
                    raw_response_obj=comp, pricing_resolver=combined_price_lookup
                )
            return comp.choices[0].message.content or "Error: No content."
        except Exception as e:
            return f"Error during OpenAI call: {e}"

    return "Error: No text generation provider is configured or enabled."




