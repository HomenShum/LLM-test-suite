"""Test tab renderers extracted from the main Streamlit app."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import asyncio
import os
import pandas as pd
import streamlit as st

from utils.data_helpers import _normalize_label, _style_selected_rows, _subset_for_run
from utils.plotly_config import PLOTLY_CONFIG

def _smarter_weighted_pick_row(row: pd.Series, f1_maps: Dict[str, Dict[str, float]]) -> Tuple[Optional[str], Optional[str]]:
    scores = {}
    model_preds = {
        "mistral": {"pred": row.get("classification_result_openrouter_mistral"), "conf": row.get("classification_result_openrouter_mistral_confidence")},
        "gpt5": {"pred": row.get("classification_result_openai"), "conf": row.get("classification_result_openai_confidence")},
        "third": {"pred": row.get("classification_result_third"), "conf": row.get("classification_result_third_confidence")}
    }

    for model, data in model_preds.items():
        pred = _normalize_label(data["pred"])
        if pred and model in f1_maps:
            conf = float(data["conf"] or 0.0)
            class_f1 = f1_maps[model].get(pred, 0.0) # Use F1 for the predicted class
            scores[model] = class_f1 * conf

    if not scores:
        return None, None

    model_pick = max(scores, key=scores.get)
    label_pick = model_preds[model_pick]["pred"]
    return model_pick, label_pick

# --- UPDATED: Flexible Judge and Pruner Functions ---
JUDGE_SCHEMA = {"type": "object","properties": {"final_choice_model": {"type": "string", "description": "One of: mistral, gpt5, third"}, "final_label": {"type": "string"}, "judge_rationale": {"type": "string"}},"required": ["final_choice_model", "final_label", "judge_rationale"], "additionalProperties": False}
JUDGE_INSTRUCTIONS = "You are a neutral judge... Return ONLY JSON with: final_choice_model, final_label, judge_rationale."


_CONFIGURED = False

def configure(context: Dict[str, Any]) -> None:
    global _CONFIGURED
    for key, value in context.items():
        if key.startswith("__") or key in {"configure"}:
            continue
        globals()[key] = value
    _CONFIGURED = True

def render_test1_tab(tab) -> None:
    if not _CONFIGURED:
        st.error("Test tab module not configured. Call configure() first.")
        return

    # Get variables from session state (set by sidebar)
    ROW_LIMIT_N = st.session_state.get('ROW_LIMIT_N', None)
    explain_cm = st.session_state.get('explain_cm', True)

    with tab:
        with tabs[1]:
            # Dashboard header
            render_test_flow_diagram(1, "Test 1: Two-Model Classification + F1/Latency Analysis")

            # Dataset preview for classification tests (Tests 1-3)
            st.subheader("Dataset Preview (Classification)")
            _df_preview = st.session_state.df if "df" in st.session_state else pd.DataFrame(columns=SKELETON_COLUMNS)

            # Show dataset source
            if os.path.exists(CLASSIFICATION_DATASET_PATH):
                dataset_source = f"📁 Source: `{CLASSIFICATION_DATASET_PATH}` ({len(_df_preview)} rows)"
            else:
                dataset_source = "⚠️ No dataset loaded. Generate datasets in the Preparation tab."

            st.caption(dataset_source)
            st.dataframe(_style_selected_rows(_df_preview, ROW_LIMIT_N), use_container_width=True, height=250)
            st.caption(f"Highlighted {len(_subset_for_run(_df_preview, ROW_LIMIT_N))} of {len(_df_preview)} rows will be used in Tests 1-3.")
        
            st.divider()
        
            # --- Per-Test Model Selection (Test 1) ---
            st.subheader("Model Selection (Test 1)")
            t1_col1, t1_col2 = st.columns(2)
            with t1_col1:
                t1_use_ollama = st.checkbox("Use OpenRouter (Test 1)", value=True, key="t1_use_ollama")
                _t1_or_ids = list(OPENROUTER_MODEL_METADATA.keys())
                _t1_or_default = _t1_or_ids.index(OPENROUTER_MODEL) if OPENROUTER_MODEL in _t1_or_ids else 0
                t1_openrouter_model = st.selectbox("OpenRouter model (Test 1)", options=_t1_or_ids, index=_t1_or_default, key="t1_or_model")
            with t1_col2:
                t1_use_openai = st.checkbox("Use OpenAI (Test 1)", value=True, key="t1_use_openai")
                _t1_oai_ids = list(OPENAI_MODEL_METADATA.keys())
                _t1_oai_default = _t1_oai_ids.index(OPENAI_MODEL) if OPENAI_MODEL in _t1_oai_ids else 0
                t1_openai_model = st.selectbox("OpenAI model (Test 1)", options=_t1_oai_ids, index=_t1_oai_default, key="t1_oai_model")
        
            # Collapsible configuration section
            with st.expander("⚙️ Test Configuration", expanded=False):
                row_limit_display = [k for k, v in ROW_LIMIT_OPTIONS.items() if v == ROW_LIMIT_N]
                row_limit_str = row_limit_display[0] if row_limit_display else f"{ROW_LIMIT_N} rows"
                st.markdown(f"""
                Models: OpenRouter={t1_openrouter_model if t1_use_ollama else '—'}; OpenAI={t1_openai_model if t1_use_openai else '—'}
                Row Limit: {row_limit_str}
                Providers Enabled: OpenRouter={t1_use_ollama}, OpenAI={t1_use_openai}
                Explain Confusion Matrix: {explain_cm}
                """)
        
            st.divider()
        
            up = st.file_uploader("Upload CSV (query, classification)", type=["csv"], key="t1_up")
            if st.button("Load uploaded (replace DF)", key="t1_load") and up:
                # ... (loading logic unchanged) ...
                st.rerun()
        
            if st.button("▶️ Run Test 1", type="primary", use_container_width=True):
                # Persist per-test OpenAI model override for this run
                st.session_state['openai_model_override'] = t1_openai_model
        
                overrides = {
                    'use_openai': t1_use_openai,
                    'openai_model': t1_openai_model,
                    'use_ollama': t1_use_ollama,
                    'openrouter_model': t1_openrouter_model,
                }
                capture_run_config("Test 1", overrides)  # CAPTURE
        
                run_classification_flow(
                    include_third_model=False,
                    use_openai_override=t1_use_openai,
                    use_ollama_override=t1_use_ollama,
                    openrouter_model_override=t1_openrouter_model,
                )
        
                # The analysis code below remains the same, but now runs on fresh data
                df = _subset_for_run(st.session_state.df, ROW_LIMIT_N)
        
                if len(df) and "classification" in df.columns:
                    # 1. FIRST: Progress replay (what just happened)
                    st.subheader("📈 Processing Timeline")
                    render_progress_replay("classification")
        
                    st.divider()
        
                    # 2. THEN: Organized results with subtabs
                    st.subheader("📊 Test Results")
                    render_organized_results(
                        df,
                        test_type="classification",
                        model_cols=["openrouter_mistral", "openai"],
                        model_names=["Mistral (OpenRouter)", "OpenAI"]
                    )
        
                    st.divider()
        
                # --- Save results at the end of the test run ---
                save_results_df(st.session_state.df, "Test 1", ROW_LIMIT_N)
                # --- PATCH 7: Structured Summary Call for Test 1 ---
                if len(df) and "classification" in df.columns:
                    report_text = f"Test 1 Results (N={len(df)}): Mistral Avg Latency: {df['latency_openrouter_mistral'].mean():.2f}s, OpenAI Avg Latency: {df['latency_openai'].mean():.2f}s. Performance reports calculated."
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(display_final_summary_for_test("Test 1 Classification", report_text))
                # ----------------------------------------------------
        

def render_test2_tab(tab) -> None:
    if not _CONFIGURED:
        st.error("Test tab module not configured. Call configure() first.")
        return

    # Get variables from session state (set by sidebar)
    ROW_LIMIT_N = st.session_state.get('ROW_LIMIT_N', None)
    explain_cm = st.session_state.get('explain_cm', True)

    with tab:
        with tabs[2]:
            # Dashboard header
            render_test_flow_diagram(2, "Test 2: Advanced Ensembling with Per-Class F1 Weighting")
        
        
            # --- Per-Test Model Selection (Test 2) ---
            st.subheader("Model Selection (Test 2)")
            t2_c1, t2_c2 = st.columns(2)
            with t2_c1:
                t2_use_ollama = st.checkbox("Use OpenRouter (Test 2)", value=True, key="t2_use_ollama")
                _t2_or_ids = list(OPENROUTER_MODEL_METADATA.keys())
                _t2_or_default = _t2_or_ids.index(OPENROUTER_MODEL) if OPENROUTER_MODEL in _t2_or_ids else 0
                t2_openrouter_model = st.selectbox("OpenRouter model (Test 2)", options=_t2_or_ids, index=_t2_or_default, key="t2_or_model")
            with t2_c2:
                t2_use_openai = st.checkbox("Use OpenAI (Test 2)", value=True, key="t2_use_openai")
                _t2_oai_ids = list(OPENAI_MODEL_METADATA.keys())
                _t2_oai_default = _t2_oai_ids.index(OPENAI_MODEL) if OPENAI_MODEL in _t2_oai_ids else 0
                t2_openai_model = st.selectbox("OpenAI model (Test 2)", options=_t2_oai_ids, index=_t2_oai_default, key="t2_oai_model")
        
            t2_third_kind = st.selectbox("Third model provider (Test 2)", ["None", "OpenRouter", "OpenAI"], index=["None","OpenRouter","OpenAI"].index(THIRD_KIND if THIRD_KIND in ["None","OpenRouter","OpenAI"] else "None"), key="t2_third_kind")
            t2_third_model = ""
            if t2_third_kind == "OpenRouter":
                t2_third_model = st.selectbox("Third model (OpenRouter)", options=_t2_or_ids, key="t2_third_model_or")
            elif t2_third_kind == "OpenAI":
                t2_third_model = st.selectbox("Third model (OpenAI)", options=_t2_oai_ids, key="t2_third_model_oai")
            # Collapsible configuration section
            with st.expander("⚙️ Test Configuration", expanded=False):
                row_limit_display = [k for k, v in ROW_LIMIT_OPTIONS.items() if v == ROW_LIMIT_N]
                row_limit_str = row_limit_display[0] if row_limit_display else f"{ROW_LIMIT_N} rows"
                st.markdown(f"""
                Models: OpenRouter={t2_openrouter_model if t2_use_ollama else '—'}; OpenAI={t2_openai_model if t2_use_openai else '—'}; Third={t2_third_model if t2_third_kind != 'None' else '—'}
                Row Limit: {row_limit_str}
                Weighting Strategy: Confidence × Per-Class F1 Score
                Explain Confusion Matrix: {explain_cm}
                """)
        
            st.divider()
            st.info("This test uses a smarter weighting: score = confidence * F1-score-of-the-predicted-class.")
        
        
            if st.button("▶️ Run Test 2", type="primary", use_container_width=True):
                # Persist per-test OpenAI model override for this run
                st.session_state['openai_model_override'] = t2_openai_model
        
                overrides = {
                    'use_openai': t2_use_openai,
                    'openai_model': t2_openai_model,
                    'use_ollama': t2_use_ollama,
                    'openrouter_model': t2_openrouter_model,
                    'third_kind': t2_third_kind,
                    'third_model': t2_third_model,
                }
                capture_run_config("Test 2", overrides)  # CAPTURE
                # First, run classification for configured models
                run_classification_flow(
                    include_third_model=(t2_third_kind != "None"),
                    use_openai_override=t2_use_openai,
                    use_ollama_override=t2_use_ollama,
                    openrouter_model_override=t2_openrouter_model,
                    third_kind_override=t2_third_kind,
                    third_model_override=t2_third_model,
                )
        
        
        
                # First, ensure classifications exist by running Test 1 logic
                # ... (classification run logic as in Test 1) ...
                df = _subset_for_run(st.session_state.df, ROW_LIMIT_N)
                if len(df):
                    # --- PATCH 13: Dynamic name for reporting ---
                    third_model_name = get_third_model_display_name()
        
                    y_true = df["classification"].tolist()
                    # Generate reports including the third model (for ensemble calculation)
                    report_m = generate_classification_report(y_true, df["classification_result_openrouter_mistral"].tolist(), "Mistral", explain=False)
                    report_g = generate_classification_report(y_true, df["classification_result_openai"].tolist(), "OpenAI", explain=False)
                    report_t = generate_classification_report(y_true, df["classification_result_third"].tolist(), third_model_name, explain=False) if "classification_result_third" in df else None
                    # ---------------------------------------------
        
                    # Create a map from class to its F1 score for each model
                    f1_maps = {
                        "mistral": {label: data['f1-score'] for label, data in report_m.items() if isinstance(data, dict)} if report_m else {},
                        "gpt5": {label: data['f1-score'] for label, data in report_g.items() if isinstance(data, dict)} if report_g else {},
                        "third": {label: data['f1-score'] for label, data in report_t.items() if isinstance(data, dict)} if report_t else {},
                    }
        
                    picks = df.apply(lambda row: _smarter_weighted_pick_row(row, f1_maps), axis=1)
                    df['weighted_pick_model'], df['weighted_pick_label'] = zip(*picks)
                    st.session_state.df.loc[df.index, ['weighted_pick_model', 'weighted_pick_label']] = df[['weighted_pick_model', 'weighted_pick_label']]
        
                    # 1. FIRST: Progress replay
                    st.subheader("📈 Processing Timeline")
                    render_progress_replay("classification")
        
                    st.divider()
        
                    # 2. THEN: Organized results with subtabs
                    st.subheader("📊 Test Results")
                    render_organized_results(
                        df,
                        test_type="classification",
                        model_cols=["openrouter_mistral", "openai", "third"],
                        model_names=["Mistral (OpenRouter)", "OpenAI", third_model_name]
                    )
        
                    st.divider()
        
                    # 2.5. Confidence Distribution Analysis
                    st.subheader("📊 Confidence Distribution Analysis")
        
                    conf_cols = [
                        ('classification_result_openrouter_mistral_confidence', 'Mistral'),
                        ('classification_result_openai_confidence', 'OpenAI'),
                        ('classification_result_third_confidence', third_model_name)
                    ]
        
                    fig_conf = go.Figure()
        
                    for col, name in conf_cols:
                        if col in df.columns:
                            confidences = df[col].dropna()
                            fig_conf.add_trace(go.Violin(
                                y=confidences,
                                name=name,
                                box_visible=True,
                                meanline_visible=True,
                                hovertemplate=f'<b>{name}</b><br>Confidence: %{{y:.3f}}<extra></extra>'
                            ))
        
                    fig_conf.update_layout(
                        title="Confidence Score Distribution by Model",
                        yaxis_title="Confidence Score",
                        height=400,
                        showlegend=True
                    )
        
                    st.plotly_chart(fig_conf, use_container_width=True, config=PLOTLY_CONFIG)
        
                    st.divider()
        
                    # 3. Ensemble results
                    with st.expander("🎯 Ensemble Performance (Weighted Pick)", expanded=True):
                        st.info("This ensemble uses: score = confidence × F1-score-of-the-predicted-class")
                        generate_classification_report(y_true, df["weighted_pick_label"].tolist(), "Smarter Weighted Pick", explain=explain_cm)
        
                    # --- Save results at the end of the test run ---
                    save_results_df(st.session_state.df, "Test 2", ROW_LIMIT_N)
                    # --- PATCH 7: Structured Summary Call for Test 2 ---
                    report_text = f"Test 2 Ensemble Results (N={len(df)}): Weighted Pick Macro F1: {report_m.get('weighted avg', {}).get('f1-score', 0.0):.4f}. Focus analysis on model disagreement."
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(display_final_summary_for_test("Test 2 Advanced Ensembling", report_text))
                    # ----------------------------------------------------
        

def render_test3_tab(tab) -> None:
    if not _CONFIGURED:
        st.error("Test tab module not configured. Call configure() first.")
        return

    # Get variables from session state (set by sidebar)
    ROW_LIMIT_N = st.session_state.get('ROW_LIMIT_N', None)
    explain_cm = st.session_state.get('explain_cm', True)

    with tab:
        with tabs[3]:
            # Dashboard header
            render_test_flow_diagram(3, "Test 3: LLM as Judge")
        
            # --- Classification Models (Test 3) ---
            st.subheader("Classification Models (Test 3)")
            t3_c1, t3_c2 = st.columns(2)
            with t3_c1:
                t3_use_ollama = st.checkbox("Use OpenRouter (Test 3)", value=True, key="t3_use_ollama")
                _t3_or_ids = list(OPENROUTER_MODEL_METADATA.keys())
                _t3_or_default = _t3_or_ids.index(OPENROUTER_MODEL) if OPENROUTER_MODEL in _t3_or_ids else 0
                t3_openrouter_model = st.selectbox("OpenRouter model (Test 3)", options=_t3_or_ids, index=_t3_or_default, key="t3_or_model")
            with t3_c2:
                t3_use_openai = st.checkbox("Use OpenAI (Test 3)", value=True, key="t3_use_openai")
                _t3_oai_ids = list(OPENAI_MODEL_METADATA.keys())
                _t3_oai_default = _t3_oai_ids.index(OPENAI_MODEL) if OPENAI_MODEL in _t3_oai_ids else 0
                t3_openai_model = st.selectbox("OpenAI model (Test 3)", options=_t3_oai_ids, index=_t3_oai_default, key="t3_oai_model")
        
            t3_third_kind = st.selectbox("Third model provider (Test 3)", ["None", "OpenRouter", "OpenAI"], index=["None","OpenRouter","OpenAI"].index(THIRD_KIND if THIRD_KIND in ["None","OpenRouter","OpenAI"] else "None"), key="t3_third_kind")
            t3_third_model = ""
            if t3_third_kind == "OpenRouter":
                t3_third_model = st.selectbox("Third model (OpenRouter)", options=_t3_or_ids, key="t3_third_model_or")
            elif t3_third_kind == "OpenAI":
                t3_third_model = st.selectbox("Third model (OpenAI)", options=_t3_oai_ids, key="t3_third_model_oai")
        
            # Collapsible configuration section
            with st.expander("⚙️ Test Configuration", expanded=False):
                row_limit_display = [k for k, v in ROW_LIMIT_OPTIONS.items() if v == ROW_LIMIT_N]
                row_limit_str = row_limit_display[0] if row_limit_display else f"{ROW_LIMIT_N} rows"
                st.markdown(f"""
                Models: OpenRouter={t3_openrouter_model if t3_use_ollama else '—'}; OpenAI={t3_openai_model if t3_use_openai else '—'}; Third={t3_third_model if t3_third_kind != 'None' else '—'}
                Judge Model: {st.session_state.get('judge_model', 'openai/gpt-5-mini')}
                Row Limit: {row_limit_str}
                Judging Strategy: Weighted F1 scores + confidence
                Explain Confusion Matrix: {explain_cm}
                """)
        
            st.divider()
        
            # --- NEW: Judge Model Selector ---
            col1, col2 = st.columns([3, 1])
            with col1:
                default_judge_model = "openai/gpt-5-mini"
                if default_judge_model not in AVAILABLE_MODELS:
                    default_judge_model = AVAILABLE_MODELS[0] if AVAILABLE_MODELS else OPENAI_MODEL
        
                default_judge_index = AVAILABLE_MODELS.index(default_judge_model) if default_judge_model in AVAILABLE_MODELS else 0
        
                judge_model = st.selectbox(
                    "Judge Model",
                    options=AVAILABLE_MODELS,
                    index=default_judge_index,
                    key='judge_model',
                    help="Select the model to act as the judge. Defaults to gpt-5-mini for cost-effectiveness."
                )
        
            with col2:
                st.metric("Judge Model", _to_native_model_id(judge_model))
        
            if st.button("▶️ Run Test 3 (Judge)", type="primary", use_container_width=True):
                # Persist per-test OpenAI model override for this run
                st.session_state['openai_model_override'] = t3_openai_model
        
                overrides = {
                    'use_openai': t3_use_openai,
                    'openai_model': t3_openai_model,
                    'use_ollama': t3_use_ollama,
                    'openrouter_model': t3_openrouter_model,
                    'third_kind': t3_third_kind,
                    'third_model': t3_third_model,
                }
                capture_run_config("Test 3", overrides)  # CAPTURE
                # Always ensure fresh classification data is available for all models
                run_classification_flow(
                    include_third_model=(t3_third_kind != "None"),
                    use_openai_override=t3_use_openai,
                    use_ollama_override=t3_use_ollama,
                    openrouter_model_override=t3_openrouter_model,
                    third_kind_override=t3_third_kind,
                    third_model_override=t3_third_model,
                )
        
                df = _subset_for_run(st.session_state.df, ROW_LIMIT_N)
        
                if not len(df):
                    st.info("No rows to judge.")
                else:
                    with st.spinner("Running Judge on classified rows..."):
                        # --- 1. Compute global F1s for weights (null-safe + filtered) ---
                        y_true_all = df["classification"].fillna("").map(_normalize_label).tolist()
        
                        # Helper to get F1 scores
                        def get_f1_report_dict(pred_series):
                            pred_list = pred_series.fillna("").map(_normalize_label).tolist()
                            valid_indices = [i for i, (t, p) in enumerate(zip(y_true_all, pred_list)) if t and p]
                            if not valid_indices: return None
                            y_true_f = [y_true_all[i] for i in valid_indices]
                            y_pred_f = [pred_list[i] for i in valid_indices]
                            return classification_report(y_true_f, y_pred_f, output_dict=True, zero_division=0)
        
                        report_m = get_f1_report_dict(df.get("classification_result_openrouter_mistral", pd.Series(dtype='str')))
                        report_g = get_f1_report_dict(df.get("classification_result_openai", pd.Series(dtype='str')))
                        report_t = get_f1_report_dict(df.get("classification_result_third", pd.Series(dtype='str')))
        
                        global_f1s = {
                            "mistral": report_m['macro avg']['f1-score'] if report_m else 0.0,
                            "gpt5": report_g['macro avg']['f1-score'] if report_g else 0.0,
                            "third": report_t['macro avg']['f1-score'] if report_t else 0.0,
                        }
        
                        # --- 2. Define and run the async judge worker ---
                        async def _judge_all_rows_async(df_to_judge, f1_scores):
        
                            async def judge_one_row(idx, row):
                                payload = {
                                    "query": row.get("query"),
                                    "candidates": {
                                        "mistral": {"label": row.get("classification_result_openrouter_mistral"), "confidence": row.get("classification_result_openrouter_mistral_confidence"), "rationale": row.get("classification_result_openrouter_mistral_rationale")},
                                        "gpt5": {"label": row.get("classification_result_openai"), "confidence": row.get("classification_result_openai_confidence"), "rationale": row.get("classification_result_openai_rationale")},
                                        "third": {"label": row.get("classification_result_third"), "confidence": row.get("classification_result_third_confidence"), "rationale": row.get("classification_result_third_rationale")} if pd.notna(row.get("classification_result_third")) else None
                                    },
                                    "weighted_scores": {k: v for k, v in [
                                        ("mistral", (f1_scores.get("mistral", 0) * float(row.get("classification_result_openrouter_mistral_confidence") or 0))),
                                        ("gpt5", (f1_scores.get("gpt5", 0) * float(row.get("classification_result_openai_confidence") or 0))),
                                        ("third", (f1_scores.get("third", 0) * float(row.get("classification_result_third_confidence") or 0)))
                                    ] if v > 0}
                                }
                                try:
                                    # Using run_judge_ollama as the primary (which calls OpenRouter)
                                    result = await run_judge_ollama(payload)
                                except Exception as e:
                                    result = {"final_choice_model": None, "final_label": None, "judge_rationale": f"Judge Error: {e}"}
        
                                # Return index with the result for safe mapping
                                return idx, result
        
                            tasks = [judge_one_row(idx, row) for idx, row in df_to_judge.iterrows() if str(row.get("query", "")).strip()]
                            return await asyncio.gather(*tasks)
        
                        # --- 3. Execute the async runner and write back results safely ---
                        loop = asyncio.get_event_loop()
                        judge_results = loop.run_until_complete(_judge_all_rows_async(df, global_f1s))
        
                        for idx, res in judge_results:
                            st.session_state.df.at[idx, "judge_choice_model"] = res.get("final_choice_model")
                            st.session_state.df.at[idx, "judge_choice_label"] = _normalize_label(res.get("final_label"))
                            st.session_state.df.at[idx, "judge_rationale"] = res.get("judge_rationale")
        
                        st.success(f"Judging complete for {len(judge_results)} rows.")
        
                        # Get fresh data with judge results
                        final_df = _subset_for_run(st.session_state.df, ROW_LIMIT_N)
                        third_model_name = get_third_model_display_name()
        
                        # 1. FIRST: Progress replay
                        st.subheader("📈 Processing Timeline")
                        render_progress_replay("classification")
        
                        st.divider()
        
                        # 2. THEN: Organized results with subtabs
                        st.subheader("📊 Test Results (Individual Models)")
                        render_organized_results(
                            final_df,
                            test_type="classification",
                            model_cols=["openrouter_mistral", "openai", "third"],
                            model_names=["Mistral (OpenRouter)", "OpenAI", third_model_name]
                        )
        
                        st.divider()
        
                        # 2.5. Judge Decision Flow Visualization
                        st.subheader("👨‍⚖️ Judge Decision Flow")
        
                        # Create Sankey diagram showing which models were chosen
                        judge_choices = final_df['judge_choice_model'].value_counts()
        
                        # Build flow data
                        labels = ['Mistral', 'OpenAI', third_model_name, 'Judge Decision']
                        source = [0, 1, 2]  # Models
                        target = [3, 3, 3]  # All flow to judge
                        values = [
                            judge_choices.get('mistral', 0),
                            judge_choices.get('gpt5', 0),
                            judge_choices.get('third', 0)
                        ]
        
                        fig_sankey = go.Figure(data=[go.Sankey(
                            node=dict(
                                pad=15,
                                thickness=20,
                                label=labels,
                                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                            ),
                            link=dict(
                                source=source,
                                target=target,
                                value=values,
                                color=['rgba(31,119,180,0.4)', 'rgba(255,127,14,0.4)', 'rgba(44,160,44,0.4)']
                            )
                        )])
        
                        fig_sankey.update_layout(
                            title="Model Selection by Judge",
                            height=300
                        )
        
                        st.plotly_chart(fig_sankey, use_container_width=True, config=PLOTLY_CONFIG)
        
                        st.divider()
        
                        # 3. Judge-specific performance
                        with st.expander("👨‍⚖️ Judge Performance (Detailed)", expanded=True):
                            st.info("The judge selects the best answer from the 3 models based on weighted F1 scores and confidence.")
                            generate_classification_report(
                                final_df["classification"].tolist(),
                                final_df["judge_choice_label"].tolist(),
                                "LLM Judge",
                                explain=explain_cm
                            )
        
                            # Show judge decisions
                            st.subheader("Judge Decisions")
                            display_cols = [
                                "query", "classification",
                                "classification_result_openrouter_mistral", "classification_result_openai", "classification_result_third",
                                "judge_choice_model", "judge_choice_label", "judge_rationale"
                            ]
                            existing_cols = [col for col in display_cols if col in final_df.columns]
        
                            # Rename third column for clarity
                            display_df = final_df.copy()
                            if "classification_result_third" in existing_cols:
                                display_df = display_df.rename(columns={"classification_result_third": f"classification_result_{third_model_name}"})
                                existing_cols = [col if col != "classification_result_third" else f"classification_result_{third_model_name}" for col in existing_cols]
        
                            # Truncate judge rationale for display to prevent MemoryError
                            judge_display_df = display_df[existing_cols].copy()
                            if "judge_rationale" in judge_display_df.columns:
                                judge_display_df["judge_rationale"] = judge_display_df["judge_rationale"].astype(str).str.slice(0, 150) + "..."
                            st.dataframe(judge_display_df, use_container_width=True)
        
                            # Download option
                            csv = display_df[existing_cols].to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "⬇️ Download Judge Results",
                                data=csv,
                                file_name="test3_judge_results.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
        
                        # --- Save results at the end of the test run ---
                        save_results_df(st.session_state.df, "Test 3", ROW_LIMIT_N)
                        # --- PATCH 7: Structured Summary Call for Test 3 ---
                        report_text = f"Test 3 LLM Judge Results (N={len(final_df)}): Judge selection finalized. Focus analysis on optimizing judge prompt or candidate presentation."
                        loop = asyncio.get_event_loop()
                        loop.run_until_complete(display_final_summary_for_test("Test 3 LLM as Judge", report_text, JUDGE_INSTRUCTIONS))
                        # ----------------------------------------------------
        
        

def render_test4_tab(tab) -> None:
    if not _CONFIGURED:
        st.error("Test tab module not configured. Call configure() first.")
        return

    # Get variables from session state (set by sidebar)
    ROW_LIMIT_N = st.session_state.get('ROW_LIMIT_N', None)
    explain_cm = st.session_state.get('explain_cm', True)

    with tab:
        with tabs[4]:
            show_main_df_previews = False
        
            # Dashboard header
            render_test_flow_diagram(4, "Test 4: Quantitative Context Pruning & Action")
        
            # Collapsible configuration section
            with st.expander("⚙️ Test Configuration", expanded=False):
                st.markdown(f"""
                **Pruner Model:** Configurable (default: gpt-5-mini)
                **Test Data Source:** Generated data or {CONTEXT_PRUNING_DATASET_PATH}
                **Metrics:** Action Accuracy + Jaccard Similarity for kept keys
                **Expected Columns:** instruction, summary, user_msgs, agent_resps, tool_logs, new_question, expected_action, expected_kept_keys
                """)
        
            st.divider()
        
            # --- NEW: Pruner Model Selector ---
            col1, col2 = st.columns([3, 1])
            with col1:
                default_pruner_model = "openai/gpt-5-mini"
                if default_pruner_model not in AVAILABLE_MODELS:
                    default_pruner_model = AVAILABLE_MODELS[0] if AVAILABLE_MODELS else OPENAI_MODEL
        
                default_pruner_index = AVAILABLE_MODELS.index(default_pruner_model) if default_pruner_model in AVAILABLE_MODELS else 0
        
                pruner_model = st.selectbox(
                    "Pruner Model",
                    options=AVAILABLE_MODELS,
                    index=default_pruner_index,
                    key='pruner_model',
                    help="Select the model to perform context pruning. Defaults to gpt-5-mini for cost-effectiveness."
                )
        
            with col2:
                st.metric("Pruner Model", _to_native_model_id(pruner_model))
        
            st.info(f"This test runs the pruner against either generated data or `{CONTEXT_PRUNING_DATASET_PATH}`.")
            st.code("Expected columns: instruction, summary, user_msgs, agent_resps, tool_logs, new_question, expected_action, expected_kept_keys (comma-separated)")
        
            # --- PATCH 18: Check for session state data first ---
            if not st.session_state.pruning_df.empty:
                pruning_df = st.session_state.pruning_df
                st.subheader(f"Pruning Testset Preview (Generated Data, N={len(pruning_df)})")
                st.dataframe(pruning_df, use_container_width=True)
                loaded_from_file = False
            else:
                # Fallback to file if session state is empty
                try:
                    pruning_df = pd.read_csv(CONTEXT_PRUNING_DATASET_PATH).fillna("")
                    st.subheader(f"Pruning Testset Preview (Loaded from {CONTEXT_PRUNING_DATASET_PATH})")
                    st.dataframe(pruning_df, use_container_width=True)
                    loaded_from_file = True
                except FileNotFoundError:
                    st.error(f"Create `{CONTEXT_PRUNING_DATASET_PATH}` or generate data in the Preparation tab to run this test.")
                    pruning_df = None # Ensure df is None if not found
                    loaded_from_file = True # Treat as file error for messaging
            # -------------------------------------------------------
        
            if st.button("▶️ Run Test 4 (Pruning)", type="primary", use_container_width=True) and pruning_df is not None:
                capture_run_config("Test 4") # CAPTURE
                async def run_all_pruners():
                    results = []
                    # Note: The logic below assumes the column names match the PruningDataItem schema,
                    # which is guaranteed if loaded from session state (PATCH 17) or if the CSV is compliant.
                    for _, row in pruning_df.iterrows():
                        # The columns in the dataframe are already assumed to be strings/pipe-separated.
                        context_items = {
                            "instruction": row.get("instruction", ""),
                            "summary": row.get("summary", ""),
                            "user_messages": str(row.get("user_msgs", "")).split("||"),
                            "agent_responses": str(row.get("agent_resps", "")).split("||"),
                            "tool_logs": str(row.get("tool_logs", "")).split("||"),
                        }
                        payload = { "context": context_items, "new_question": row.get("new_question", "") }
                        results.append(run_pruner(payload))
                    return await asyncio.gather(*results)
        
                with st.spinner(f"Running pruner on {len(pruning_df)} test cases..."):
                    pruner_outputs = asyncio.run(run_all_pruners())
        
                correct_actions, key_scores = 0, []
                results_data = [] # For the detailed results table
        
                for i, row in pruning_df.iterrows():
                    output = pruner_outputs[i]
        
                    # --- Action Comparison ---
                    model_action = output.get('action')
                    expected_action = row['expected_action']
                    action_correct = (model_action == expected_action)
                    if action_correct:
                        correct_actions += 1
        
                    # --- Kept Keys Comparison (Jaccard) ---
                    # Ensure expected_kept_keys is treated as a string before splitting
                    expected_keys = set(str(row.get('expected_kept_keys', '')).split(','))
                    expected_keys.discard('') # Remove empty string if split results in one
                    model_keys_raw = output.get('kept_context_keys', [])
                    model_keys = set(model_keys_raw) if isinstance(model_keys_raw, list) else set()
        
                    intersection = len(expected_keys.intersection(model_keys))
                    union = len(expected_keys.union(model_keys))
                    jaccard_score = intersection / union if union > 0 else 0
                    key_scores.append(jaccard_score)
        
                    results_data.append({
                        "Question": row['new_question'],
                        "Expected Action": expected_action,
                        "Model Action": model_action,
                        "Action Correct": "✅" if action_correct else "❌",
                        "Expected Keys": ", ".join(sorted(expected_keys)),
                        "Model Keys": ", ".join(sorted(model_keys)),
                        "Key Score (Jaccard)": jaccard_score
                    })
        
                action_accuracy = correct_actions / len(pruning_df) if len(pruning_df) > 0 else 0
                avg_key_score = sum(key_scores) / len(key_scores) if key_scores else 0
        
                # Create results DataFrame
                results_df = pd.DataFrame(results_data)
        
                # Use organized rendering
                st.subheader("📊 Test Results")
                render_kpi_metrics(results_df, test_type="pruning")
        
                st.divider()
                st.subheader("📊 Pruning Analysis Visualizations")
        
                viz_col1, viz_col2 = st.columns(2)
        
                with viz_col1:
                    # Heatmap: Action vs. Count of Kept Keys
                    st.markdown("**Heatmap: Action vs. Number of Kept Keys**")
        
                    # Prepare data
                    heatmap_data = []
                    for _, row in results_df.iterrows():
                        num_keys = len(row['Model Keys'].split(', ')) if row['Model Keys'] else 0
                        heatmap_data.append({
                            'Action': row['Model Action'],
                            'Num_Keys': num_keys
                        })
        
                    heatmap_df = pd.DataFrame(heatmap_data)
                    pivot = heatmap_df.groupby(['Action', 'Num_Keys']).size().reset_index(name='Count')
                    pivot_matrix = pivot.pivot_table(index='Action', columns='Num_Keys', values='Count', fill_value=0)
        
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=pivot_matrix.values,
                        x=pivot_matrix.columns,
                        y=pivot_matrix.index,
                        colorscale='Blues',
                        text=pivot_matrix.values,
                        texttemplate='%{text}',
                        textfont={"size": 14},
                        hovertemplate='Action: %{y}<br>Keys: %{x}<br>Count: %{z}<extra></extra>',
                        colorbar=dict(title="Frequency")
                    ))
        
                    fig_heatmap.update_layout(
                        xaxis_title="Number of Keys Kept",
                        yaxis_title="Action Type",
                        height=300
                    )
        
                    st.plotly_chart(fig_heatmap, use_container_width=True, config=PLOTLY_CONFIG)
        
                with viz_col2:
                    # Stacked bar: Specific kept keys by action
                    st.markdown("**Key Distribution by Action (Stacked)**")
        
                    # Parse all kept keys by action
                    key_action_data = []
                    for _, row in results_df.iterrows():
                        action = row['Model Action']
                        keys = [k.strip() for k in row['Model Keys'].split(',') if k.strip()]
                        for key in keys:
                            key_action_data.append({'Action': action, 'Key': key})
        
                    if key_action_data:
                        ka_df = pd.DataFrame(key_action_data)
                        key_counts = ka_df.groupby(['Action', 'Key']).size().reset_index(name='Count')
        
                        fig_stacked = go.Figure()
        
                        actions = key_counts['Action'].unique()
                        keys = key_counts['Key'].unique()
        
                        for key in keys:
                            key_data = key_counts[key_counts['Key'] == key]
                            fig_stacked.add_trace(go.Bar(
                                name=key,
                                x=key_data['Action'],
                                y=key_data['Count'],
                                text=key_data['Count'],
                                textposition='inside',
                                hovertemplate='<b>%{fullData.name}</b><br>Action: %{x}<br>Count: %{y}<extra></extra>'
                            ))
        
                        fig_stacked.update_layout(
                            barmode='stack',
                            xaxis_title="Action Type",
                            yaxis_title="Key Count",
                            height=300,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
        
                        st.plotly_chart(fig_stacked, use_container_width=True, config=PLOTLY_CONFIG)
                    else:
                        st.info("No key data available for visualization.")
        
                st.divider()
        
                # Show detailed results in organized format
                with st.expander("📋 Detailed Row-by-Row Results", expanded=True):
                    # Truncate long rationale columns for display
                    display_df = results_df.copy()
                    rationale_cols = [c for c in display_df.columns if c.endswith("_rationale") or c == "judge_rationale"]
                    for col in rationale_cols:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].astype(str).str.slice(0, 150) + "..."
                    st.dataframe(display_df, use_container_width=True)
        
                    # Download option
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Pruning Results",
                        data=csv,
                        file_name="test4_pruning_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
                # --- Save results at the end of the test run ---
                save_results_df(results_df, "Test 4", len(pruning_df), is_pruning_test=True)
                # --- PATCH 7: Structured Summary Call for Test 4 ---
                report_text = f"Test 4 Pruning Results (N={len(pruning_df)}): Action Accuracy: {action_accuracy:.2%}, Avg Key Similarity: {avg_key_score:.3f}. Focus analysis on context key pruning efficiency."
                loop = asyncio.get_event_loop()
                loop.run_until_complete(display_final_summary_for_test("Test 4 Quantitative Context Pruning", report_text, PRUNER_INSTRUCTIONS))
                # ----------------------------------------------------
        
        

def render_test5_tab(tab) -> None:
    if not _CONFIGURED:
        st.error("Test tab module not configured. Call configure() first.")
        return

    # Get variables from session state (set by sidebar)
    ROW_LIMIT_N = st.session_state.get('ROW_LIMIT_N', None)
    explain_cm = st.session_state.get('explain_cm', True)

    with tab:
        with tabs[5]:
            # Dashboard header
            render_test_flow_diagram(5, "Test 5: Unified Orchestrator (Three Modes)")
        
            # Collapsible configuration section
            with st.expander("⚙️ Test Configuration", expanded=False):
                st.markdown("""
                **Architecture:** Unified orchestrator with three execution modes
        
                **Execution Modes:**
                1. **🎯 Direct Inference:** Pattern matching (classification, prediction)
                   - Each turn: Generate → Evaluate → Analyze Failures → Refine
                   - Best for: Prediction tasks, classification, tool sequence prediction
                   - Example: "Predict tool sequences for user queries"
        
                2. **📊 Computational Analysis:** Statistics, simulations, optimization
                   - Uses code execution for computational tasks
                   - Best for: Data analysis, statistical computations, optimization
                   - Example: "Analyze performance metrics and compute statistics"
        
                3. **🔍 Research Tasks:** Multi-source information gathering
                   - Each turn: Decompose → Execute in Parallel → Synthesize
                   - Best for: Open-ended research, multi-source information gathering
                   - Example: "Research George Morgan, Symbolica AI, and their fundraising"
        
                **Features:**
                - ✅ Auto-mode detection based on goal
                - ✅ Parallel task execution (asyncio.gather)
                - ✅ Smart caching with deduplication
                - ✅ Convergence detection (mode-specific)
        
                **Budget Modes:**
                - **Fixed Turns:** Run exactly N iterations (predictable, simpler)
                - **Cost/Token Limit:** Run until budget exhausted or converged (efficient, dynamic)
        
                **Stopping Conditions:**
                - Turn mode: Max turns reached OR no improvement in last 3 turns
                - Cost mode: Budget exhausted OR marginal value < 1%
        
                **Execution Model:** Gemini 2.5 Flash with Code Execution
                **Test Data:** Tool/Agent Sequence dataset (for inference mode)
                **Evaluation Metric:** Exact sequence match accuracy (inference mode)
                **Turn Tracking:** Per-turn metrics with improvement trajectory visualization
                """)
        
            st.divider()
        
            # ============================================================
            # DEMO LAUNCHER UI COMPONENT
            # ============================================================
            st.markdown("### 🎬 Quick Demonstration Scenarios")
            st.caption("Pre-configured scenarios showcasing Leaf Agent Scaffold versatility across domains")
        
            col1, col2 = st.columns(2)
        
            with col1:
                if st.button("🤖 Demo 1: PI Agent (Laundry Folding)", use_container_width=True, help="Autonomous robotics with vision-to-motor control and adaptive learning"):
                    st.session_state['demo_scenario'] = 'pi_agent'
                    st.session_state['demo_goal'] = PI_AGENT_GOAL_PROMPT
                    st.session_state['demo_agents'] = ["web_researcher", "code_executor", "validator", "content_generator"]
                    st.session_state['demo_memory_policy'] = FOLDING_POLICY_BLOCK
                    st.session_state['demo_mode'] = 'research'
                    st.session_state['demo_coordination'] = 'leaf_scaffold'
                    st.session_state['auto_run_demo'] = True  # Flag to auto-run
                    st.success("✅ PI Agent demo starting...")
                    st.rerun()
        
            with col2:
                if st.button("🔒 Demo 2: Cybersecurity (Phishing Analysis)", use_container_width=True, help="Threat detection with reasoning, risk scoring, and policy enforcement"):
                    st.session_state['demo_scenario'] = 'cybersecurity'
                    st.session_state['demo_goal'] = CYBERSECURITY_GOAL_PROMPT
                    st.session_state['demo_agents'] = ["web_researcher", "validator", "code_executor", "content_generator"]
                    st.session_state['demo_memory_policy'] = THREAT_POLICY_BLOCK
                    st.session_state['demo_mode'] = 'research'
                    st.session_state['demo_coordination'] = 'leaf_scaffold'
                    st.session_state['auto_run_demo'] = True  # Flag to auto-run
                    st.success("✅ Cybersecurity demo starting...")
                    st.rerun()
        
            # Display active scenario indicator
            if 'demo_scenario' in st.session_state:
                scenario_name = "PI Agent (Laundry Folding)" if st.session_state['demo_scenario'] == 'pi_agent' else "Cybersecurity (Phishing Analysis)"
                st.info(f"📌 Active Demo: **{scenario_name}**")
        
                # Add clear demo button
                if st.button("🔄 Clear Demo Configuration", help="Reset to manual configuration"):
                    for key in ['demo_scenario', 'demo_goal', 'demo_agents', 'demo_memory_policy', 'demo_mode', 'demo_coordination']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.success("Demo configuration cleared!")
                    st.rerun()
        
            st.divider()
        
            # Mode selection
            st.markdown("### Task Type")
            default_mode = st.session_state.get('demo_mode', 'auto')
            mode_index = ["auto", "inference", "research", "analysis"].index(default_mode) if default_mode in ["auto", "inference", "research", "analysis"] else 0
            mode_option = st.radio(
                "Select Mode",
                options=["auto", "inference", "research", "analysis"],
                index=mode_index,
                format_func=lambda x: {
                    "auto": "🤖 Auto-detect",
                    "inference": "🎯 Direct Inference",
                    "research": "🔍 Research Tasks",
                    "analysis": "📊 Computational Analysis"
                }.get(x, x),
                horizontal=True,
                help="""
                - **Auto-detect:** Automatically choose mode based on goal
                - **Direct Inference:** Prediction/classification (e.g., tool sequences)
                - **Research Tasks:** Open-ended multi-source questions
                - **Analysis:** Computational statistics and optimization
                """
            )
        
            # Mode-specific info (hide for auto mode)
            if mode_option == "inference":
                st.info("💡 **Inference Mode**: Optimizes prompts through self-evaluation and failure analysis")
                st.caption("Best for: Classification, prediction, pattern matching")
        
            elif mode_option == "analysis":
                st.info("💡 **Analysis Mode**: Generates Python code for computational tasks")
                st.caption("Best for: Statistics, data analysis, optimization, simulations")
        
            elif mode_option == "research":
                st.info("💡 **Research Mode**: Hybrid strategy - decomposes into subtasks, uses code generation for computational parts")
                st.caption("Best for: Multi-source research with computational analysis requirements")
        
            # Auto mode: no info box (mode not yet determined)
        
            st.divider()
        
            # Coordination Pattern selection
            st.markdown("### Coordination Pattern")
            default_coordination = st.session_state.get('demo_coordination', 'auto')
            coordination_index = ["auto", "solo", "subagent", "multi_agent", "leaf_scaffold"].index(default_coordination) if default_coordination in ["auto", "solo", "subagent", "multi_agent", "leaf_scaffold"] else 0
            coordination_option = st.radio(
                "Select Coordination Pattern",
                options=["auto", "solo", "subagent", "multi_agent", "leaf_scaffold"],
                index=coordination_index,
                format_func=lambda x: {
                    "auto": "🤖 Auto-detect",
                    "solo": "👤 Solo Agent",
                    "subagent": "🏗️ Subagent Orchestration",
                    "multi_agent": "👥 Multi-Agent Collaboration",
                    "leaf_scaffold": "🌳 Leaf Agent Scaffold"
                }.get(x, x),
                horizontal=True,
                help="""
                - **Auto-detect:** Automatically choose pattern based on task complexity
                - **Solo Agent:** Single agent executes independently (fastest)
                - **Subagent Orchestration:** Hierarchical delegation with specialized subagents
                - **Multi-Agent Collaboration:** Peer agents propose, review, and build consensus
                - **Leaf Agent Scaffold:** Hierarchical multi-agent with supervisor and specialized leaf agents
                """
            )
        
            # Pattern-specific info (hide simple info for multi-agent and auto)
            if coordination_option == "solo":
                st.info("👤 **Solo Agent**: Single agent executes the task independently")
                st.caption("Best for: Simple, straightforward tasks")
        
            elif coordination_option == "subagent":
                st.info("🏗️ **Subagent Orchestration**: Hierarchical delegation (Decomposer → Generator → Evaluator → Analyzer → Synthesizer)")
                st.caption("Best for: Complex tasks requiring specialized expertise at each stage")
        
                # Optional: Show subagent workflow diagram
                with st.expander("🏗️ Subagent Workflow Details", expanded=False):
                    st.markdown("""
                    **Hierarchical Pipeline**:
                    1. **Decomposer** → Breaks goal into subtasks
                    2. **Generator** → Creates solutions for each subtask
                    3. **Evaluator** → Tests and scores solutions
                    4. **Analyzer** → Identifies failures and suggests improvements
                    5. **Synthesizer** → Combines results into final solution
        
                    Each subagent specializes in one stage of the pipeline.
                    """)
        
            elif coordination_option == "multi_agent":
                st.divider()
                st.markdown("## 👥 Multi-Agent Collaboration")
                st.info("**How it works**: Peer agents independently propose solutions, cross-review each other's work, and build consensus through collaborative synthesis")
                st.caption("✨ Best for: Tasks requiring diverse perspectives, validation, or creative problem-solving")
        
                # Multi-agent configuration
                st.markdown("### 🎭 Configure Your Agent Team")
        
                # Get default roles based on mode
                default_roles = {
                    "inference": ["Pattern Analyst", "Rule Designer", "Evaluator"],
                    "analysis": ["Data Scientist", "Algorithm Designer", "Code Reviewer"],
                    "research": ["Market Analyst", "Technical Expert", "Domain Specialist"]
                }
        
                current_mode = mode_option if mode_option != "auto" else "inference"
                default_role_list = default_roles.get(current_mode, ["Agent 1", "Agent 2", "Agent 3"])
        
                peer_roles_input = st.text_input(
                    "Peer Agent Roles (comma-separated)",
                    value=", ".join(default_role_list),
                    help="Define the roles for peer agents. Each role brings a different perspective."
                )
        
                peer_agent_roles = [role.strip() for role in peer_roles_input.split(",") if role.strip()]
        
                if len(peer_agent_roles) < 2:
                    st.warning("⚠️ Multi-agent mode requires at least 2 peer agents")
                else:
                    st.success(f"✅ {len(peer_agent_roles)} peer agents configured: {', '.join(peer_agent_roles)}")
        
                # Show workflow preview
                with st.expander("📋 Multi-Agent Workflow Preview", expanded=True):
                    st.markdown(f"""
                    **Round 1: Independent Proposals** 💡
                    - Each of the {len(peer_agent_roles)} agents proposes their solution independently
                    - Agents: {', '.join(f'**{role}**' for role in peer_agent_roles)}
        
                    **Round 2: Cross-Review** 🔍
                    - Each agent reviews proposals from other agents
                    - Provides constructive feedback and identifies concerns
        
                    **Round 3: Consensus Building** 🤝
                    - Synthesizer combines all proposals and reviews
                    - Creates unified solution incorporating best ideas
        
                    **Round 4: Joint Evaluation** ✅
                    - All agents jointly evaluate the consensus
                    - Assess quality, agreement, and completeness
        
                    **View Results**: Check the **Agent Dashboard** (Tab 6) to see the full interaction timeline!
                    """)
        
                # Quick demo button
                st.markdown("#### 🎯 Quick Demo")
                demo_col1, demo_col2 = st.columns([3, 1])
                with demo_col1:
                    st.caption("Try a pre-configured multi-agent collaboration to see how it works")
                with demo_col2:
                    if st.button("🎬 Load Demo", type="secondary", use_container_width=True):
                        # Set demo values
                        st.session_state['test5_prompt'] = "Design a robust tool sequence predictor with 90%+ accuracy"
                        st.rerun()
        
                # Show message if demo was just loaded
                if st.session_state.get('test5_prompt') == "Design a robust tool sequence predictor with 90%+ accuracy":
                    st.success("✅ Demo prompt loaded! Scroll down and click the '👥 Run Multi-Agent Collaboration' button to execute.")
        
            elif coordination_option == "leaf_scaffold":
                st.divider()
                st.markdown("## 🌳 Leaf Agent Scaffold")
                st.info("**Hierarchical multi-agent system** with supervisor orchestration and specialized leaf agents")
                st.caption("✨ Best for: Complex tasks requiring multiple specialized capabilities (research, computation, writing, validation)")
        
                # Show architecture diagram
                with st.expander("🏗️ Architecture Overview", expanded=True):
                    st.markdown("""
                    **Hierarchical Structure**:
        
                    ```
                    🧠 Supervisor Agent
                        ├── 📋 Task Planning (decomposes complex task)
                        ├── 🎯 Delegation (routes to specialists)
                        └── 🔄 Synthesis (combines results)
        
                    👥 Specialized Leaf Agents
                        ├── 🔍 Web Researcher (information gathering)
                        ├── 💻 Code Executor (computation & analysis)
                        ├── ✍️ Content Generator (writing & formatting)
                        └── ✅ Validator (quality assurance)
                    ```
        
                    **How it works**:
                    1. **Supervisor** analyzes your goal and breaks it into specialized sub-tasks
                    2. **Leaf Agents** execute their assigned sub-tasks independently
                    3. **Supervisor** synthesizes all results into final answer
        
                    **Check Agent Dashboard (Tab 6)** to see the full execution hierarchy!
                    """)
        
                # Agent selection
                st.markdown("### 🎭 Select Leaf Agents")
        
                available_agents = {
                    "web_researcher": "🔍 Web Researcher (information gathering from web sources)",
                    "code_executor": "💻 Code Executor (Python computation & data analysis)",
                    "content_generator": "✍️ Content Generator (writing & formatting)",
                    "validator": "✅ Validator (quality assurance & validation)"
                }
        
                # Use demo agents if available, otherwise default
                default_agents = st.session_state.get('demo_agents', ["web_researcher", "validator", "content_generator"])
                selected_agent_types = st.multiselect(
                    "Choose specialized agents for this task",
                    options=list(available_agents.keys()),
                    default=default_agents,
                    format_func=lambda x: available_agents[x],
                    help="Select agents based on task requirements. The supervisor will automatically delegate sub-tasks to appropriate agents. For research tasks, include 'validator' to prevent hallucinations."
                )
        
                # Store in session state for orchestrator
                st.session_state['selected_agent_types'] = selected_agent_types
        
                if len(selected_agent_types) < 1:
                    st.warning("⚠️ Select at least one leaf agent")
                else:
                    st.success(f"✅ {len(selected_agent_types)} specialized agents configured")
        
                    # Show selected agents
                    st.markdown("**Your Agent Team**:")
                    for agent_type in selected_agent_types:
                        st.write(f"  • {available_agents[agent_type]}")
        
                # Quick demo button
                st.markdown("#### 🎯 Quick Demo")
                demo_col1, demo_col2 = st.columns([3, 1])
                with demo_col1:
                    st.caption("Try a pre-configured leaf agent scaffold to see hierarchical orchestration in action")
                with demo_col2:
                    if st.button("🎬 Load Demo", key="leaf_demo", type="secondary", use_container_width=True):
                        # Set demo values
                        st.session_state['test5_prompt'] = "Research and analyze the latest trends in multi-agent AI systems, then write a comprehensive report"
                        st.rerun()
        
                # Show message if demo was just loaded
                if st.session_state.get('test5_prompt') == "Research and analyze the latest trends in multi-agent AI systems, then write a comprehensive report":
                    st.success("✅ Demo prompt loaded! Scroll down and click the '🌳 Run Leaf Agent Scaffold' button to execute.")
        
                peer_agent_roles = None
        
            else:
                peer_agent_roles = None
                selected_agent_types = None
        
            # Dataset-aware suggested prompts
            TEST5_SUGGESTED_PROMPTS = {
                "inference": [
                    "Design an agent that predicts tool sequences from user queries with 90%+ accuracy through iterative refinement.",
                    "Build a pattern-matching agent using keyword extraction and rule-based mapping for tool sequence prediction.",
                    "Create a rule-based tool sequence predictor with learning capability that analyzes failures and adds new rules."
                ],
                "research": [
                    "Research [Company/Person Name] for due diligence: background, financial health, market position, recent news, and key stakeholders.",
                    "Competitive analysis for [Industry/Market]: identify top players, compare features/pricing, analyze strengths/weaknesses.",
                    "Investment thesis research: fundamentals, market opportunity, competitive moat, team execution, and risk assessment."
                ],
                "analysis": [
                    "Analyze test dataset performance metrics: sequence length distribution, common tool combinations, coverage analysis.",
                    "Optimize prediction accuracy through data analysis: baseline accuracy, hard vs easy examples, feature importance, error analysis.",
                    "Compute statistical summary of dataset: complexity scores, optimal training split, correlation analysis."
                ]
            }
        
            # Initialize prompt in session state
            if 'test5_prompt' not in st.session_state:
                st.session_state['test5_prompt'] = ""
        
            # Prompt generator function
            def generate_test5_prompt():
                """Generate a dataset-aware prompt based on current mode AND user input."""
                mode = mode_option if mode_option != "auto" else "inference"
        
                # Get current user input
                current_input = st.session_state.get('test5_prompt', '')
        
                # Select base prompt
                import random
                prompts = TEST5_SUGGESTED_PROMPTS.get(mode, TEST5_SUGGESTED_PROMPTS["inference"])
                base_prompt = random.choice(prompts) if isinstance(prompts, list) else prompts
        
                # Build context from dataset
                context = {
                    'dataset_size': len(st.session_state.agent_df) if not st.session_state.agent_df.empty else 0,
                    'columns': list(st.session_state.agent_df.columns) if not st.session_state.agent_df.empty else [],
                    'mode': mode
                }
        
                if mode == "inference" and not st.session_state.agent_df.empty:
                    # Extract actual tools from dataset
                    all_tools = set()
                    for seq in st.session_state.agent_df['expected_sequence']:
                        if isinstance(seq, list):
                            all_tools.update(seq)
                        elif isinstance(seq, str):
                            try:
                                tools = json.loads(seq.replace("'", '"'))
                                all_tools.update(tools)
                            except:
                                all_tools.update([t.strip() for t in seq.split(',') if t.strip()])
        
                    # Sample queries
                    sample_queries = st.session_state.agent_df['query'].head(3).tolist()
        
                    context['tools'] = sorted(all_tools)[:15]
                    context['sample_queries'] = sample_queries
                    context['success_criteria'] = '85%+ exact sequence match accuracy'
        
                    # Add dataset-specific approach
                    context['suggested_approach'] = f"""Pattern matching + entity extraction:
        1. Keyword dictionaries for common patterns
        2. Regex for structured data (IDs, IPs)
        3. Sequence logic for multi-step operations
        4. Confidence scoring for ambiguous cases"""
        
                    # Create dataset-aware base prompt
                    base_prompt = f"""Design a tool sequence predictor for IT operations queries.
        
        DATASET: {len(st.session_state.agent_df)} examples
        TOOLS: {', '.join(sorted(all_tools)[:12])}...
        
        SAMPLE QUERIES:
        {chr(10).join([f'- "{q}"' for q in sample_queries])}
        
        APPROACH:
        1. Pattern matching (keywords → tools)
        2. Entity extraction (IDs, IPs, device names)
        3. Sequence logic (auth before query, check before order)
        4. Error handling (unknown patterns → default)
        
        TARGET: 85%+ exact match accuracy"""
        
                # Enhance with user input
                enhanced = enhance_prompt_with_user_input(base_prompt, current_input, context)
        
                return enhanced
        
            # Different inputs based on mode
            if mode_option in ["auto", "research"]:
                use_test_data = False
        
                # Prompt management section (NO test data preview for research/auto)
                st.markdown("### 🎯 Research Goal")
        
                # Use demo goal if available, otherwise default
                if 'demo_goal' in st.session_state:
                    st.session_state['test5_prompt'] = st.session_state['demo_goal']
                    # Clear demo_goal after using it to prevent re-applying on every rerun
                    del st.session_state['demo_goal']
                elif not st.session_state['test5_prompt']:
                    st.session_state['test5_prompt'] = "Help me research George Morgan, Symbolica AI, their position on AI Engineering, and their next fundraising round"
        
                col1, col2 = st.columns([3, 1])
                with col1:
                    pass  # Placeholder for layout
                with col2:
                    if st.button("🎲 Generate Prompt", width='content'):
                        st.session_state['test5_prompt'] = generate_test5_prompt()
                        st.rerun()
        
                goal = st.text_area(
                    "Research Goal / Task Description",
                    value=st.session_state['test5_prompt'],
                    height=200,  # Increased height for demo prompts
                    help="Describe what you want to research or accomplish",
                    key="test5_goal_input"
                )
        
                # Update session state
                st.session_state['test5_prompt'] = goal
        
                if mode_option == "research":
                    st.info("💡 **Research Mode:** Decomposes goal into parallel subtasks for multi-source information gathering.")
            else:
                # For inference and analysis modes: SHOW test data preview
                if st.session_state.agent_df.empty:
                    st.warning("⚠️ Please generate a Tool/Agent Sequence dataset in the 'Preparation' tab first.")
                    use_test_data = False
                    goal = "Predict tool sequences"
                else:
                    st.subheader("📊 Agent Test Data Preview")
                    st.dataframe(st.session_state.agent_df.head(5), use_container_width=True)
                    st.caption(f"Total: {len(st.session_state.agent_df)} examples")
        
                    use_test_data = True
        
                    # Prompt management section
                    st.markdown("### 🎯 Task Description")
        
                    # Use demo goal if available, otherwise generate
                    if 'demo_goal' in st.session_state:
                        st.session_state['test5_prompt'] = st.session_state['demo_goal']
                        # Clear demo_goal after using it
                        del st.session_state['demo_goal']
                    elif not st.session_state['test5_prompt']:
                        st.session_state['test5_prompt'] = generate_test5_prompt()
        
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        pass  # Placeholder for layout
                    with col2:
                        if st.button("🎲 Generate Prompt", width='content'):
                            st.session_state['test5_prompt'] = generate_test5_prompt()
                            st.rerun()
        
                    goal = st.text_area(
                        "Agent Goal / Task Description",
                        value=st.session_state['test5_prompt'],
                        height=200,
                        help="Describe the task in detail. Be specific about inputs, outputs, and success criteria.",
                        key="test5_goal_input"
                    )
        
                    # Update session state
                    st.session_state['test5_prompt'] = goal
        
                    # Quick Start Templates (ONLY for inference/analysis, NOT research)
                    st.markdown("##### 💡 Quick Start Templates")
                    mode_key = mode_option if mode_option != "auto" else "inference"
                    prompts = TEST5_SUGGESTED_PROMPTS.get(mode_key, TEST5_SUGGESTED_PROMPTS["inference"])
        
                    cols = st.columns(len(prompts))
                    for i, prompt in enumerate(prompts):
                        with cols[i]:
                            label = f"Template {i+1}"
                            if st.button(label, key=f"test5_pill_{i}", help=prompt, use_container_width=True):
                                st.session_state['test5_prompt'] = prompt
                                st.rerun()
        
                    # Mode-specific tips (only for inference/analysis in this section)
                    if mode_option == "inference":
                        st.info("💡 **Inference Mode:** Each turn generates code, evaluates it, and refines based on failures. Best for prediction tasks.")
                    elif mode_option == "analysis":
                        st.info("💡 **Analysis Mode:** Uses code execution for computational analysis and optimization tasks.")
        
            st.divider()
        
            # Budget mode selection
            budget_mode = st.radio(
                "Budget Mode",
                options=["turns", "cost"],
                format_func=lambda x: "Fixed Turns" if x == "turns" else "Cost/Token Limit",
                horizontal=True,
                help="Fixed Turns: Run exactly N iterations. Cost/Token: Run until budget exhausted or converged."
            )
        
            col1, col2 = st.columns(2)
        
            if budget_mode == "turns":
                with col1:
                    max_turns = st.number_input("Max Turns", min_value=1, max_value=20, value=3)
                with col2:
                    st.info("Runs for exactly N turns (tracking cost for reporting)")
            else:  # cost mode
                with col1:
                    max_cost = st.number_input("Budget (USD)", min_value=0.5, max_value=20.0, value=2.0, step=0.5)
                with col2:
                    max_tokens = st.number_input("Token Limit", min_value=100_000, max_value=5_000_000, value=500_000, step=100_000)
        
            # Run button with pattern-specific messaging
            st.divider()
        
            if coordination_option == "multi_agent":
                st.markdown("### 🎬 Run Multi-Agent Collaboration")
                st.info("💡 **Tip**: After running, switch to **Tab 6 (Agent Dashboard)** to see the full agent interaction timeline!")
                button_label = "👥 Run Multi-Agent Collaboration"
            elif coordination_option == "subagent":
                st.markdown("### 🎬 Run Subagent Orchestration")
                button_label = "🏗️ Run Subagent Orchestration"
            elif coordination_option == "leaf_scaffold":
                st.markdown("### 🎬 Run Leaf Agent Scaffold")
                st.info("💡 **Tip**: After running, switch to **Tab 6 (Agent Dashboard)** to see the hierarchical execution flow!")
                button_label = "🌳 Run Leaf Agent Scaffold"
            else:
                st.markdown("### 🎬 Run Orchestrator")
                button_label = "🚀 Run Unified Orchestrator"
        
            # Check if demo should auto-run
            auto_run = st.session_state.get('auto_run_demo', False)
            if auto_run:
                # Clear the flag to prevent infinite loop
                st.session_state['auto_run_demo'] = False
                # Show demo is running
                demo_name = "PI Agent (Laundry Folding)" if st.session_state.get('demo_scenario') == 'pi_agent' else "Cybersecurity (Phishing Analysis)"
                st.info(f"🎬 **Auto-running demo:** {demo_name}")
        
            if st.button(button_label, type="primary", use_container_width=True) or auto_run:
                if not GEMINI_API_KEY:
                    st.error("GEMINI_API_KEY required")
                else:
                    # Reset tracker for fresh run
                    tracker = st.session_state.get('execution_tracker')
                    if tracker:
                        tracker.reset()
        
                    # Create budget based on mode
                    if budget_mode == "turns":
                        budget = Budget(mode="turns", max_turns=max_turns)
                    else:
                        budget = Budget(mode="cost", max_cost_usd=max_cost, max_tokens=max_tokens)
        
                    # Prepare test data if needed
                    test_data = None
                    if use_test_data and not st.session_state.agent_df.empty:
                        # CRITICAL: Parse expected_sequence from string to list with robust error handling
                        test_data = st.session_state.agent_df.to_dict('records')
        
                        parse_errors = []
                        for idx, item in enumerate(test_data):
                            seq = item.get('expected_sequence', '')
                            original_seq = seq  # Keep for error reporting
        
                            try:
                                if isinstance(seq, str):
                                    # Handle various formats: "tool1,tool2" or "['tool1','tool2']" or "tool1|tool2"
                                    if seq.startswith('['):
                                        # JSON-like format
                                        try:
                                            parsed = json.loads(seq.replace("'", '"'))
                                            # Validate it's actually a list
                                            if isinstance(parsed, list):
                                                item['expected_sequence'] = parsed
                                            else:
                                                # Single value wrapped in brackets
                                                item['expected_sequence'] = [parsed]
                                        except json.JSONDecodeError as e:
                                            # Fallback to split
                                            item['expected_sequence'] = [s.strip() for s in re.split(r'[,|;]', seq.strip('[]')) if s.strip()]
                                            parse_errors.append(f"Row {idx}: JSON parse failed, used split fallback: {str(e)[:50]}")
                                    else:
                                        # Comma or pipe separated
                                        item['expected_sequence'] = [s.strip() for s in re.split(r'[,|;]', seq) if s.strip()]
                                elif isinstance(seq, list):
                                    # Already a list - validate contents
                                    item['expected_sequence'] = [str(x).strip() for x in seq]
                                else:
                                    # Unexpected type
                                    item['expected_sequence'] = []
                                    parse_errors.append(f"Row {idx}: Unexpected type {type(seq).__name__}, defaulted to empty list")
                            except Exception as e:
                                # Catch-all for any parsing errors
                                item['expected_sequence'] = []
                                parse_errors.append(f"Row {idx}: Parse error '{str(e)[:50]}', defaulted to empty list")
        
                        # Show parsing warnings if any
                        if parse_errors:
                            st.warning(f"⚠️ {len(parse_errors)} data parsing issue(s):")
                            for err in parse_errors[:5]:  # Show first 5
                                st.text(f"  • {err}")
                            if len(parse_errors) > 5:
                                st.text(f"  ... and {len(parse_errors) - 5} more")
        
                        st.write(f"📋 Loaded {len(test_data)} test cases")
                        if test_data:
                            st.write(f"📝 Sample: {test_data[0]}")  # Verify format
        
                    # Create orchestrator with auto-detection or explicit mode and coordination pattern
                    orchestrator_kwargs = {
                        "goal": goal,
                        "test_data": test_data,
                        "budget": budget
                    }
        
                    # Add mode if not auto
                    if mode_option != "auto":
                        orchestrator_kwargs["mode"] = mode_option
        
                    # Add coordination pattern if not auto
                    if coordination_option != "auto":
                        orchestrator_kwargs["coordination_pattern"] = coordination_option
        
                    # Add peer agent roles for multi-agent mode
                    if coordination_option == "multi_agent" and peer_agent_roles:
                        orchestrator_kwargs["peer_agent_roles"] = peer_agent_roles
        
                    orchestrator = UnifiedOrchestrator(**orchestrator_kwargs)
        
                    try:
                        results = asyncio.run(orchestrator.run())
        
                        # Display coordination pattern used
                        st.info(f"🎯 Mode: **{orchestrator.mode.upper()}** | 🤝 Pattern: **{orchestrator.coordination_pattern.upper()}**")
        
                        # Handle different result types based on coordination pattern and mode
                        if orchestrator.coordination_pattern == "multi_agent":
                            # Multi-agent returns a dict with consensus
                            st.success("✅ Multi-agent collaboration completed!")
        
                            # Prominent dashboard link
                            st.info("📊 **View the full agent interaction timeline in Tab 6 (Agent Dashboard)!**")
        
                            # Show summary metrics
                            col1, col2, col3 = st.columns(3)
        
                            if isinstance(results, dict):
                                with col1:
                                    st.metric("Final Score", f"{results.get('final_score', 0.0):.3f}")
                                with col2:
                                    st.metric("Peer Agents", len(results.get('peer_roles', [])))
                                with col3:
                                    st.metric("Consensus Rounds", len(results.get('consensus_history', [])))
        
                                # Show final consensus in expandable section
                                with st.expander("📋 View Final Consensus", expanded=False):
                                    st.json(results.get('final_consensus', {}))
        
                                # Show consensus history
                                if results.get('consensus_history'):
                                    with st.expander("📚 View Consensus History", expanded=False):
                                        for idx, consensus_round in enumerate(results['consensus_history'], 1):
                                            st.markdown(f"### Turn {idx}")
        
                                            # Show proposals
                                            st.markdown("**Proposals:**")
                                            for proposal in consensus_round.get('proposals', []):
                                                role = proposal.get('role', 'Unknown')
                                                prop = proposal.get('proposal', {})
                                                st.markdown(f"- **{role}**: {prop.get('approach', 'N/A')}")
        
                                            # Show evaluation
                                            eval_data = consensus_round.get('evaluation', {})
                                            st.markdown(f"**Score**: {eval_data.get('score', 0.0):.3f} | **Agreement**: {eval_data.get('agreement', 0.0):.3f}")
                                            st.divider()
        
                        elif orchestrator.coordination_pattern == "leaf_scaffold":
                            # Leaf scaffold returns a dict with hierarchical results
                            st.success("✅ Leaf agent scaffold execution completed!")
        
                            # Prominent dashboard link
                            st.info("📊 **View the hierarchical execution flow in Tab 6 (Agent Dashboard)!**")
        
                            # Show summary metrics
                            col1, col2, col3 = st.columns(3)
        
                            if isinstance(results, dict):
                                with col1:
                                    st.metric("Leaf Agents", len(results.get('leaf_agents', [])))
                                with col2:
                                    st.metric("Sub-Tasks", results.get('sub_tasks', 0))
                                with col3:
                                    successful = results.get('metadata', {}).get('successful_tasks', 0)
                                    st.metric("Successful Tasks", successful)
        
                                # Show final result
                                if results.get('final_result'):
                                    st.markdown("### 📊 Final Synthesized Result")
                                    st.markdown(results['final_result'])
        
                                # Show metadata
                                if results.get('metadata'):
                                    with st.expander("📈 Execution Metadata", expanded=False):
                                        st.json(results['metadata'])
        
                                # Show contributing agents
                                if results.get('leaf_agents'):
                                    with st.expander("👥 Contributing Agents", expanded=False):
                                        for agent_name in results['leaf_agents']:
                                            st.write(f"  • {agent_name}")
        
                        elif orchestrator.coordination_pattern == "subagent":
                            # Subagent returns a dict with synthesized results
                            st.success("Subagent orchestration completed!")
        
                            if isinstance(results, dict):
                                st.metric("Final Score", f"{results.get('score', 0.0):.3f}")
                                st.metric("Best Turn", results.get('best_turn', 0))
                                st.metric("Total Turns", results.get('total_turns', 0))
        
                                if results.get('solution'):
                                    st.subheader("Final Solution")
                                    st.json(results['solution'])
        
                        elif orchestrator.mode == "research":
                            # Research mode returns a dict with findings
                            st.success("Research completed!")
                            st.json(results)
        
                        elif orchestrator.mode in ["inference", "analysis"]:
                            # Inference/analysis modes in solo pattern return (code, perf, history)
                            if isinstance(results, dict):
                                # New format from prompt optimization
                                best_code = results.get('best_prompt', '')
                                best_perf = results.get('best_accuracy', 0.0)
                                history = results.get('history', [])
                            else:
                                # Legacy tuple format
                                best_code, best_perf, history = results
        
                            st.success(f"Final Performance: {best_perf:.4f} (Turn {orchestrator.best_turn})")
        
                            if best_code:
                                st.subheader("Best Code Generated")
                                st.code(best_code, language='python')
        
                            # Turn-by-turn breakdown
                            st.subheader("Turn-by-Turn Progress")
        
                            turns_df = pd.DataFrame([
                                {
                                    "Turn": m.turn,
                                    "Tasks Attempted": m.tasks_attempted,
                                    "Tasks Verified": m.tasks_verified,
                                    "Best Accuracy": f"{m.best_accuracy:.4f}",
                                    "Improvement": f"{m.improvement:+.4f}",
                                    "Cost": f"${m.cost_spent:.3f}",
                                    "Tokens": f"{m.tokens_used:,}"
                                }
                                for m in history
                            ])
        
                            st.dataframe(turns_df, width='content')
        
                            # Plot improvement trajectory
                            if history:
                                fig = go.Figure()
        
                                fig.add_trace(go.Scatter(
                                    x=[m.turn for m in history],
                                    y=[m.best_accuracy for m in history],
                                    mode='lines+markers',
                                    name='Accuracy',
                                    line=dict(color='#1f77b4', width=3),
                                    marker=dict(size=8)
                                ))
        
                                fig.update_layout(
                                    title="Performance Improvement Trajectory",
                                    xaxis_title="Turn",
                                    yaxis_title="Accuracy",
                                    height=400
                                )
        
                                st.plotly_chart(fig, width='content', config=PLOTLY_CONFIG)
        
                            # NEW: Agent Memory State Expander
                            if orchestrator.memory_manager:
                                with st.expander("🧠 Agent Memory State", expanded=False):
                                    st.markdown("### Core Memory Blocks")
        
                                    for block_name, block in orchestrator.memory_manager.core_blocks.items():
                                        st.markdown(f"#### {block_name}")
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Version", block.version)
                                        with col2:
                                            st.metric("Last Modified Turn", block.last_modified_turn)
                                        with col3:
                                            st.metric("Modifications", block.modification_count)
        
                                        st.json({"content": block.content})
                                        st.divider()
        
                                    # Archival Memory Summary
                                    st.markdown("### Archival Memory")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Total Entries", len(orchestrator.memory_manager.archival_entries))
                                    with col2:
                                        st.metric("Total Retrievals", orchestrator.memory_manager.stats["total_retrievals"])
                                    with col3:
                                        avg_latency = orchestrator.memory_manager.stats["avg_retrieval_latency_ms"]
                                        st.metric("Avg Retrieval Latency", f"{avg_latency:.2f}ms")
        
                                    # Show recent archival entries
                                    if orchestrator.memory_manager.archival_entries:
                                        st.markdown("#### Recent Entries")
                                        recent_entries = orchestrator.memory_manager.archival_entries[-5:]
                                        for entry in reversed(recent_entries):
                                            st.text(f"[{entry.source_agent}] {entry.content[:100]}...")
                                            st.caption(f"Tags: {', '.join(entry.tags)} | {entry.timestamp}")
        
                            # NEW: Rethink Events on Improvement Trajectory
                            if orchestrator.self_correction_manager and orchestrator.self_correction_manager.rethink_history:
                                st.markdown("### 🔄 Self-Correction Events")
        
                                # Show rethink events as markers on the chart
                                rethink_turns = [r["turn"] for r in orchestrator.self_correction_manager.rethink_history]
        
                                if rethink_turns:
                                    st.info(f"🔄 {len(rethink_turns)} rethink event(s) occurred at turn(s): {', '.join(map(str, rethink_turns))}")
        
                                    # Table of rethink events
                                    rethink_df = pd.DataFrame([
                                        {
                                            "Turn": r["turn"],
                                            "Block Modified": r["block_name"],
                                            "Trigger": r["trigger"][:50] + "..." if len(r["trigger"]) > 50 else r["trigger"],
                                            "Timestamp": r["timestamp"]
                                        }
                                        for r in orchestrator.self_correction_manager.rethink_history
                                    ])
        
                                    st.dataframe(rethink_df, use_container_width=True)
        
                                    # Analyze effectiveness
                                    if history:
                                        performance_scores = [m.best_accuracy for m in history]
                                        effectiveness = orchestrator.self_correction_manager.analyze_correction_effectiveness(performance_scores)
        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Total Rethinks", effectiveness["total_rethinks"])
                                        with col2:
                                            st.metric("Successful Rethinks", effectiveness["successful_rethinks"])
                                        with col3:
                                            success_rate = effectiveness["success_rate"] * 100
                                            st.metric("Success Rate", f"{success_rate:.1f}%")
        
                            # NEW: Turn-by-Turn Metrics with Memory & Security
                            if orchestrator.dashboard_logger:
                                st.markdown("### 📊 Detailed Turn Metrics")
        
                                # Load execution log
                                try:
                                    run_data = DashboardLogger.load_run(orchestrator.dashboard_logger.run_id)
                                    execution_log = run_data.get("execution_log", [])
                                    security_audits = run_data.get("security_audits", [])
        
                                    # Group by turn
                                    turn_metrics = {}
                                    for entry in execution_log:
                                        turn = entry.get("turn", 0)
                                        if turn not in turn_metrics:
                                            turn_metrics[turn] = {
                                                "memory_ops": 0,
                                                "security_checks": 0,
                                                "events": []
                                            }
        
                                        if entry.get("event_type") == "MEMORY_WRITE":
                                            turn_metrics[turn]["memory_ops"] += 1
        
                                        turn_metrics[turn]["events"].append(entry)
        
                                    # Add security audits
                                    for audit in security_audits:
                                        turn = audit.get("turn", 0)
                                        if turn in turn_metrics:
                                            turn_metrics[turn]["security_checks"] += 1
        
                                    # Create enhanced metrics table
                                    enhanced_metrics = []
                                    for m in history:
                                        turn_data = turn_metrics.get(m.turn, {})
                                        enhanced_metrics.append({
                                            "Turn": m.turn,
                                            "Accuracy": f"{m.best_accuracy:.4f}",
                                            "Memory Ops": turn_data.get("memory_ops", 0),
                                            "Security Checks": turn_data.get("security_checks", 0),
                                            "Duration": f"{m.cost_spent:.3f}s",
                                            "Improvement": f"{m.improvement:+.4f}"
                                        })
        
                                    st.dataframe(pd.DataFrame(enhanced_metrics), use_container_width=True)
        
                                except Exception as e:
                                    st.warning(f"Could not load detailed metrics: {e}")
        
                            # Summary
                            st.subheader("Execution Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Turns", len(history))
                            with col2:
                                st.metric("Verified Tasks", len(orchestrator.cache.verified_tasks))
                            with col3:
                                st.metric("Total Cost", f"${budget.spent_cost:.2f}")
                            with col4:
                                st.metric("Best Turn", orchestrator.best_turn)
        
                            # Knowledge Index Summary
                            if orchestrator.index.entries:
                                st.subheader("Knowledge Index")
                                verified_count = sum(1 for e in orchestrator.index.entries if e["verdict"] == "verified")
                                partial_count = sum(1 for e in orchestrator.index.entries if e["verdict"] == "partial")
                                failed_count = sum(1 for e in orchestrator.index.entries if e["verdict"] == "failed")
        
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Verified", verified_count, delta=None)
                                with col2:
                                    st.metric("Partial", partial_count, delta=None)
                                with col3:
                                    st.metric("Failed", failed_count, delta=None)
        
                            # LLM Analysis
                            mode_desc = f"Turn-based ({len(history)} turns)" if budget_mode == "turns" else f"Cost-based (${budget.spent_cost:.2f})"
                            report = f"""Orchestrator Test Report:
        Task Type: {orchestrator.mode.upper()}
        Mode: {mode_desc}
        Final Best Accuracy: {best_perf:.4f} (achieved at turn {orchestrator.best_turn})
        Total Turns: {len(history)}
        Verified Tasks: {len(orchestrator.cache.verified_tasks)}
        Total Cost: ${budget.spent_cost:.2f}
        Total Tokens: {budget.spent_tokens:,}
        
        Turn-by-Turn Summary:
        {chr(10).join([f"Turn {m.turn}: {m.tasks_verified}/{m.tasks_attempted} verified, accuracy={m.best_accuracy:.4f}, improvement={m.improvement:+.4f}" for m in history])}
        """
        
                            summary_result = asyncio.run(get_structured_summary_and_refinement(report, best_code))
        
                            st.subheader("🎯 LLM Analysis & Suggested Improvements")
                            st.markdown(f"**Summary:** {summary_result.findings_summary}")
                            st.markdown("**Key Suggestions:**")
                            st.json(summary_result.key_suggestions)
        
                            if summary_result.suggested_improvement_code:
                                st.subheader("✨ Refined Code/Prompt for Next Iteration")
                                st.code(summary_result.suggested_improvement_code, language='python')
                            if summary_result.suggested_improvement_prompt_reasoning:
                                st.subheader("💡 Reasoning for Refinement")
                                st.markdown(summary_result.suggested_improvement_prompt_reasoning)
        
                    except Exception as e:
                        st.error(f"Orchestrator failed: {e}")
                        st.exception(e)
        
        

