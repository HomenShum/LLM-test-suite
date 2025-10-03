"""Sidebar renderers extracted from the main app."""

from __future__ import annotations

from typing import Dict, Any

import streamlit as st

import pandas as pd

_CONFIGURED = False

def configure(context: Dict[str, Any]) -> None:
    global _CONFIGURED
    for key, value in context.items():
        if key.startswith("__") or key in {"configure", "render_api_sidebar", "render_primary_sidebar"}:
            continue
        globals()[key] = value
    _CONFIGURED = True

def render_api_sidebar() -> None:
    if not _CONFIGURED:
        st.error("Sidebar module not configured. Call configure() first.")
        return
    with st.sidebar:
        with st.sidebar:
            st.header("⚙️ API Configuration")

            # API Key inputs
            st.subheader("🔑 API Keys")

            # Only show user-entered keys, NOT secrets (for security)
            # Get current keys from session state only (don't expose secrets)
            current_openrouter_key = st.session_state.get('OPENROUTER_API_KEY', "")
            current_openai_key = st.session_state.get('OPENAI_API_KEY', "")
            current_gemini_key = st.session_state.get('GEMINI_API_KEY', "")
            current_linkup_key = st.session_state.get('LINKUP_API_KEY', "")

            # Show status if secrets are configured
            has_secrets = bool(st.secrets.get("OPENROUTER_API_KEY") or st.secrets.get("OPENAI_API_KEY"))
            if has_secrets:
                st.info("ℹ️ API keys are configured. Enter your own keys below to override.")

            # OpenRouter API Key
            openrouter_key = st.text_input(
                "OpenRouter API Key",
                value=current_openrouter_key,
                type="password",
                placeholder="sk-or-v1-...",
                help="Required for OpenRouter mode and most models"
            )
            if openrouter_key:
                st.session_state['OPENROUTER_API_KEY'] = openrouter_key

            # OpenAI API Key
            openai_key = st.text_input(
                "OpenAI API Key",
                value=current_openai_key,
                type="password",
                placeholder="sk-proj-...",
                help="Required for native OpenAI API calls"
            )
            if openai_key:
                st.session_state['OPENAI_API_KEY'] = openai_key

            # Gemini API Key
            gemini_key = st.text_input(
                "Gemini API Key",
                value=current_gemini_key,
                type="password",
                placeholder="AIza...",
                help="Required for native Gemini API calls"
            )
            if gemini_key:
                st.session_state['GEMINI_API_KEY'] = gemini_key

            # Linkup API Key
            linkup_key = st.text_input(
                "Linkup API Key",
                value=current_linkup_key,
                type="password",
                placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                help="Optional: For Linkup search integration and pricing lookup"
            )
            if linkup_key:
                st.session_state['LINKUP_API_KEY'] = linkup_key

            st.divider()

            # Global API routing mode selector
            api_mode = st.radio(
                "API Routing Mode",
                options=["openrouter", "native"],
                index=0 if API_ROUTING_MODE == "openrouter" else 1,
                help="""
                **OpenRouter Mode**: Routes all API calls through OpenRouter
                - ✅ Unified pricing from one source
                - ✅ Consistent model IDs
                - ✅ Accurate cost tracking
                - ✅ Access to latest models

                **Native Mode**: Uses native provider APIs (OpenAI SDK, Google Genai SDK)
                - ✅ Access to advanced features (file analysis, vision, etc.)
                - ✅ Direct provider integration
                - ⚠️ Requires separate API keys for each provider
                """
            )

            # Update global routing mode
            globals()['API_ROUTING_MODE'] = api_mode

            # Manual pricing refresh button
            st.divider()
            if st.button("🔄 Refresh Pricing Data", help="Manually fetch latest pricing from OpenRouter/Linkup APIs. Only use if absolutely needed."):
                with st.spinner("Fetching latest pricing data..."):
                    from core.pricing import fetch_openrouter_pricing, fetch_gemini_models_from_linkup, fetch_openai_models_from_linkup

                    # Force refresh pricing data
                    fetch_openrouter_pricing(force_refresh=True)
                    fetch_gemini_models_from_linkup.clear()  # Clear Streamlit cache
                    fetch_openai_models_from_linkup.clear()  # Clear Streamlit cache

                    st.success("✅ Pricing data refreshed!")
                    st.info("Note: Cached pricing is used by default to avoid API calls on Streamlit Cloud.")
        
            st.divider()
        
            # Display current model configuration
            st.subheader("📋 Current Models")
        
            if api_mode == "openrouter":
                st.info("**Using OpenRouter for all calls**")
                st.code(f"OpenAI: {OPENAI_MODEL}\nGemini: {GEMINI_MODEL}\nOpenRouter: {OPENROUTER_MODEL}", language="text")
            else:
                st.info("**Using Native APIs**")
                st.code(f"OpenAI: {_to_native_model_id(OPENAI_MODEL)}\nGemini: {_to_native_model_id(GEMINI_MODEL)}", language="text")
        
            st.divider()
        

def render_primary_sidebar() -> None:
    if not _CONFIGURED:
        st.error("Sidebar module not configured. Call configure() first.")
        return
    with st.sidebar:
        with st.sidebar:
            st.header("Providers & Models (Deprecated)")
            st.info("Model selection moved into each test section. Configure per test below. This sidebar block will be removed.")
        
            # --- PATCH 26/29: Unified Format Function & UI Updates ---
            def format_model_option(model_id: str) -> str:
                """Format model ID with metadata for display in dropdown (unified for all providers)."""
                if model_id == "Custom...":
                    return model_id
        
                # Check Gemini first due to potential model naming conflicts
                if model_id in GEMINI_MODEL_METADATA:
                    metadata = GEMINI_MODEL_METADATA.get(model_id, {})
                    context = metadata.get('context', 'N/A')
                    input_cost = metadata.get('input_cost', 'N/A')
                    output_cost = metadata.get('output_cost', 'N/A')
                    return f"{model_id} (Ctx: {context}, In: {input_cost}/M, Out: {output_cost}/M)"
        
                # Determine provider based on ID structure
                if model_id in OPENROUTER_MODEL_METADATA:
                    metadata = OPENROUTER_MODEL_METADATA.get(model_id, {})
                    context = metadata.get('context', 'N/A')
                    input_cost = metadata.get('input_cost', 'N/A')
                    output_cost = metadata.get('output_cost', 'N/A')
                    display_name = model_id.split('/')[-1]
                    return f"{display_name} (Ctx: {context}, In: {input_cost}/M, Out: {output_cost}/M)"
        
                if model_id in OPENAI_MODEL_METADATA:
                    metadata = OPENAI_MODEL_METADATA.get(model_id, {})
                    context = metadata.get('context', 'N/A')
                    input_cost = metadata.get('input_cost', 'N/A')
                    output_cost = metadata.get('output_cost', 'N/A')
                    return f"{model_id} (Ctx: {context}, In: {input_cost}/M, Out: {output_cost}/M)"
        
                if model_id in OLLAMA_MODEL_METADATA:
                    metadata = OLLAMA_MODEL_METADATA.get(model_id, {})
                    context = metadata.get('context', 'N/A')
                    info = metadata.get('local_info', 'N/A')
                    return f"{model_id} (Ctx: {context}, Info: {info})"
        
                return model_id
            # -------------------------------------------------------------
        
            # 1. OpenRouter Selection (Main Model)
            _openrouter_model_ids = list(OPENROUTER_MODEL_METADATA.keys()) + ["Custom..."]
            _default_idx_or = _openrouter_model_ids.index(OPENROUTER_MODEL) if OPENROUTER_MODEL in _openrouter_model_ids else (len(_openrouter_model_ids) - 1)
            _selected_or = st.selectbox("OpenRouter model", options=_openrouter_model_ids, index=_default_idx_or, format_func=format_model_option, key="openrouter_main_select")
            OPENROUTER_MODEL = st.text_input("Custom OpenRouter model ID", value=OPENROUTER_MODEL) if _selected_or == "Custom..." else _selected_or
            if use_ollama and not OPENROUTER_API_KEY: st.warning("OPENROUTER_API_KEY not set.")
        
            # 2. Gemini Selection (PATCH 29)
            use_gemini = st.checkbox("Use Google Gemini (structured output)", value=False, key='use_gemini')
            _gemini_model_ids = list(GEMINI_MODEL_METADATA.keys())
            GEMINI_MODEL = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
            _default_idx_gmn = _gemini_model_ids.index(GEMINI_MODEL) if GEMINI_MODEL in _gemini_model_ids else 0
            selected_gemini = st.selectbox("Gemini model", options=_gemini_model_ids, index=_default_idx_gmn, format_func=format_model_option, key="gemini_main_select")
            st.session_state['gemini_model'] = selected_gemini
            if use_gemini and not GEMINI_API_KEY: st.warning("GEMINI_API_KEY not set.")
        
            # 3. OpenAI Selection
            use_openai = st.checkbox("Use OpenAI (structured output)", value=True)
            _openai_model_ids = list(OPENAI_MODEL_METADATA.keys())
            _default_idx_oai = _openai_model_ids.index(OPENAI_MODEL) if OPENAI_MODEL in _openai_model_ids else 0
            OPENAI_MODEL = st.selectbox("OpenAI model", options=_openai_model_ids, index=_default_idx_oai, format_func=format_model_option, key="openai_main_select")
            if use_openai and not OPENAI_API_KEY: st.warning("OPENAI_API_KEY not set.")
        
            # 4. Ollama Selection (Local)
            use_ollama_local = st.checkbox("Use Ollama (local/private)", value=False)
            OLLAMA_BASE_URL = st.text_input("Ollama base URL", value=OLLAMA_BASE_URL)
            _ollama_model_ids = list(OLLAMA_MODEL_METADATA.keys())
            _default_idx_ollama = _ollama_model_ids.index(OLLAMA_MODEL) if OLLAMA_MODEL in _ollama_model_ids else (len(_ollama_model_ids) - 1)
            _selected_ollama = st.selectbox("Ollama model", options=_ollama_model_ids, index=_default_idx_ollama, format_func=format_model_option, key="ollama_main_select")
            OLLAMA_MODEL = st.text_input("Custom Ollama model ID", value=OLLAMA_MODEL) if _selected_ollama == "Custom..." else _selected_ollama
        
            st.divider()
            st.subheader("Third Model (Test 2/3)")
            third_kind_options = ["None", "OpenRouter", "OpenAI"]
            default_third_kind = THIRD_KIND if THIRD_KIND in third_kind_options else "OpenAI"
            THIRD_KIND = st.selectbox("Third model kind", third_kind_options, index=third_kind_options.index(default_third_kind))
            if THIRD_KIND == "OpenRouter":
                default_third_index = _openrouter_model_ids.index(THIRD_MODEL) if THIRD_MODEL in _openrouter_model_ids else 0
                THIRD_MODEL = st.selectbox("Third model (OpenRouter)", options=_openrouter_model_ids, format_func=format_model_option, index=default_third_index, key="openrouter_third_select")
            elif THIRD_KIND == "OpenAI":
                default_openai_index = _openai_model_ids.index(THIRD_MODEL) if THIRD_MODEL in _openai_model_ids else 0
                THIRD_MODEL = st.selectbox("Third model (OpenAI)", options=_openai_model_ids, format_func=format_model_option, index=default_openai_index, key="openai_third_select")
            else:
                THIRD_MODEL = ""
            st.session_state['third_model_kind'] = THIRD_KIND
            st.session_state['third_model'] = THIRD_MODEL
            st.divider()
            st.subheader("DataFrame Controls")
            if st.button("Clear results"):
                for c in st.session_state.df.columns:
                    if c not in ["query", "classification"]: st.session_state.df[c] = None
                st.info("Cleared model outputs")
            st.subheader("Row limit for tests")
            _limit_choice = st.selectbox("Rows to test", list(ROW_LIMIT_OPTIONS.keys()), index=3)
            ROW_LIMIT_N = ROW_LIMIT_OPTIONS[_limit_choice]
            st.session_state['ROW_LIMIT_N'] = ROW_LIMIT_N  # Store in session state for test tabs
            st.divider()
            st.subheader("Analysis Options")
            explain_cm = st.checkbox("Explain Confusion Matrices with LLM", value=True, help="Uses the selected OpenRouter/OpenAI model to provide a natural language explanation of confusion matrix results.")
            st.session_state['explain_cm'] = explain_cm  # Store in session state for test tabs

        # --- COST TRACKING UI (AT BOTTOM) ---
        with st.sidebar:
            st.divider()
            st.subheader("💰 Cost Tracking")

            ct = st.session_state.cost_tracker

            # Compact display - just the essentials
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Cost", f"${ct.totals['total_cost_usd']:.4f}")
            with col2:
                st.metric("Total Tokens", f"{ct.totals['total_tokens']:,}")

            # Expandable details
            with st.expander("📊 Detailed Breakdown"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Input Cost", f"${ct.totals['input_cost_usd']:.4f}")
                    st.metric("Prompt Tokens", f"{ct.totals['prompt_tokens']:,}")
                with col2:
                    st.metric("Output Cost", f"${ct.totals['output_cost_usd']:.4f}")
                    st.metric("Completion Tokens", f"{ct.totals['completion_tokens']:,}")

                st.divider()

                # Interactive cost visualizations
                render_cost_dashboard()

            # Display breakdown by provider/model
            with st.expander("🔍 Cost by Model"):
                summary = ct.get_summary()
                if summary:
                    for (provider, model), stats in summary.items():
                        st.markdown(f"**{provider} / {model}**")
                        st.text(f"  Calls: {stats['calls']}")
                        st.text(f"  Tokens: {stats['total_tokens']:,}")
                        st.text(f"  Cost: ${stats['total_cost_usd']:.4f}")
                        st.divider()
                else:
                    st.info("No API calls tracked yet.")

            # Display recent calls
            with st.expander("📝 Recent Calls"):
                if ct.by_call:
                    recent_calls = ct.by_call[-5:]  # Last 5 calls
                    for i, call in enumerate(reversed(recent_calls), 1):
                        st.markdown(f"**{call['provider']}** / {call['model']}")
                        st.text(f"  Tokens: {call['total_tokens']:,} | Cost: ${call['total_cost_usd']:.4f}")
                        st.divider()
                else:
                    st.info("No API calls yet.")

            # Reset button
            if st.button("🔄 Reset Tracking", use_container_width=True):
                st.session_state.cost_tracker.reset()
                st.success("Cost tracking reset!")
                st.rerun()
        
        


