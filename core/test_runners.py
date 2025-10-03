"""
Test execution functions extracted from streamlit_test_v5.py

This module contains test execution functions for running classification tests.
"""

import asyncio
import time
import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any

# Import API client functions
from core.api_clients import _classify_df_async

# Import data helpers
from utils.data_helpers import _subset_for_run

# Import execution tracker
from utils.execution_tracker import ExecutionTracker

# Module-level configuration
_CONFIG = {}

def configure(context: Dict[str, Any]) -> None:
    """Configure module with global variables from main app."""
    _CONFIG.clear()
    _CONFIG.update(context)

def run_classification_flow(
    include_third_model: bool = False,
    use_openai_override: Optional[bool] = None,
    use_ollama_override: Optional[bool] = None,
    openrouter_model_override: Optional[str] = None,
    use_ollama_local_override: Optional[bool] = None,
    ollama_base_url_override: Optional[str] = None,
    ollama_model_override: Optional[str] = None,
    third_kind_override: Optional[str] = None,
    third_model_override: Optional[str] = None,
):
    """
    Centralized function to run the async classification process.
    Reads per-test settings (overrides) and updates the session state DataFrame.

    Args:
        include_third_model (bool): If True, includes the third model in the run.
        *_override: Per-test overrides for provider toggles and model IDs.
    """
    # Effective toggles/models: per-test overrides take precedence over sidebar/global
    use_openai_eff = use_openai_override if use_openai_override is not None else _CONFIG.get('use_openai', True)
    use_ollama_eff = use_ollama_override if use_ollama_override is not None else _CONFIG.get('use_ollama', False)
    use_ollama_local_eff = use_ollama_local_override if use_ollama_local_override is not None else _CONFIG.get('use_ollama_local', False)
    openrouter_model_eff = openrouter_model_override if openrouter_model_override else _CONFIG.get('OPENROUTER_MODEL', 'mistralai/mistral-small-3.2-24b-instruct')
    ollama_base_url_eff = ollama_base_url_override if ollama_base_url_override else _CONFIG.get('OLLAMA_BASE_URL', 'http://localhost:11434/api/generate')
    ollama_model_eff = ollama_model_override if ollama_model_override else _CONFIG.get('OLLAMA_MODEL', 'mistral-small:24b-instruct-2501-q4_K_M')
    third_kind_eff = third_kind_override if third_kind_override is not None else (_CONFIG.get('THIRD_KIND', 'None') if include_third_model else "None")
    third_model_eff = third_model_override if third_model_override is not None else (_CONFIG.get('THIRD_MODEL', '') if include_third_model else "")

    if include_third_model:
        if third_kind_eff == "None" or not third_model_eff:
            st.error("Test 2 and Test 3 require a configured third model. Set it in the sidebar before running.")
            return
        openai_key = st.session_state.get('OPENAI_API_KEY') or _CONFIG.get('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY', '')
        openrouter_key = st.session_state.get('OPENROUTER_API_KEY') or _CONFIG.get('OPENROUTER_API_KEY') or st.secrets.get('OPENROUTER_API_KEY', '')
        if third_kind_eff == "OpenAI" and not openai_key:
            st.error("OPENAI_API_KEY is required for the third model. Add it in the sidebar secrets.")
            return
        if third_kind_eff == "OpenRouter" and not openrouter_key:
            st.error("OPENROUTER_API_KEY is required for the third model. Add it in the sidebar secrets.")
            return
        st.session_state['third_model_kind'] = third_kind_eff
        st.session_state['third_model'] = third_model_eff

    # Check if any provider is selected for the current run
    provider_enabled = (
        use_openai_eff or use_ollama_eff or use_ollama_local_eff or
        (include_third_model and third_kind_eff != "None")
    )
    if not provider_enabled:
        st.warning("Please enable at least one provider in the test's configuration to run classification.")
        return  # Stop execution if no models are configured to run

    # Track start time for history
    start_time = time.time()

    # Initialize test history if needed
    if 'test_history' not in st.session_state:
        st.session_state.test_history = []

    loop = asyncio.get_event_loop()
    with st.spinner("Classifying... this may take a few minutes."):
        try:
            ROW_LIMIT_N = st.session_state.get('ROW_LIMIT_N', None)
            df_run = _subset_for_run(st.session_state.df, ROW_LIMIT_N).copy()

            out_df = loop.run_until_complete(
                _classify_df_async(
                    df_run,
                    use_openai_eff,
                    use_ollama_eff,
                    openrouter_model_eff,
                    use_ollama_local_eff,
                    ollama_base_url_eff,
                    ollama_model_eff,
                    third_kind=third_kind_eff,
                    third_model=third_model_eff,
                )
            )

            # Robustly update the main DataFrame
            st.session_state.df.loc[out_df.index, out_df.columns] = out_df
            st.success(f"Classification complete on {len(df_run)} rows.")

            # Track test completion in history
            duration = time.time() - start_time
            metadata = st.session_state.last_progress_metadata.get('classification', {})
            batch_count = len(metadata.get('batch_timestamps', []))

            # Count active models (effective settings)
            model_count = sum([use_openai_eff, use_ollama_eff, use_ollama_local_eff, include_third_model and third_kind_eff != "None"])

            st.session_state.test_history.append({
                'name': 'Classification Test',
                'icon': '🔍',
                'description': f'Classified {len(df_run)} items across {model_count} model(s)',
                'status': 'complete',
                'duration': duration,
                'batch_count': batch_count,
                'cost': st.session_state.cost_tracker.totals.get('total_cost_usd', 0.0),
                'timestamp': time.time()
            })

        except Exception as e:
            st.error(f"An error occurred during classification: {e}")
            st.exception(e) # Show full traceback for better debugging

            # Track error in history
            duration = time.time() - start_time
            st.session_state.test_history.append({
                'name': 'Classification Test',
                'icon': '🔍',
                'description': f'Error: {str(e)[:50]}...',
                'status': 'error',
                'duration': duration,
                'batch_count': 0,
                'cost': 0.0,
                'timestamp': time.time()
            })


# ---------- Model Callers (OpenAI, Ollama, OpenRouter) ----------
