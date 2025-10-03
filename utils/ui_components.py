"""
Reusable UI components for Streamlit dashboard.
Extracted from streamlit_test_v5.py to reduce main file size and eliminate repetition.
"""

import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Optional
from sklearn.metrics import classification_report

# Import label normalization from config
from config.scenarios import CANON_MAP


def _normalize_label(s):
    """Normalize label for comparison."""
    if not s or not isinstance(s, str):
        return ""
    s_clean = s.strip().lower()
    return CANON_MAP.get(s_clean, s_clean)


class ModelSelector:
    """Unified model selection component for all tests."""
    
    @staticmethod
    def render(test_name: str, defaults: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Renders unified model selection UI.
        
        Args:
            test_name: Name of the test (for unique keys)
            defaults: Default values for model selection
            
        Returns:
            Dictionary with selected model configuration
        """
        defaults = defaults or {}
        
        st.subheader(f"Model Selection ({test_name})")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_ollama = st.checkbox(
                "Use Ollama (Mistral)",
                value=defaults.get('use_ollama', False),
                key=f"{test_name}_use_ollama"
            )
            
            use_openai = st.checkbox(
                "Use OpenAI (GPT-5)",
                value=defaults.get('use_openai', True),
                key=f"{test_name}_use_openai"
            )
        
        with col2:
            third_kind = st.selectbox(
                "Third Model Provider",
                ["None", "OpenRouter", "Gemini"],
                index=defaults.get('third_kind_index', 0),
                key=f"{test_name}_third_kind"
            )
            
            third_model = None
            if third_kind != "None":
                # Get available models based on provider
                if third_kind == "OpenRouter":
                    from streamlit_test_v5 import get_openrouter_models
                    models = get_openrouter_models()
                elif third_kind == "Gemini":
                    from streamlit_test_v5 import get_gemini_models
                    models = get_gemini_models()
                else:
                    models = []
                
                third_model = st.selectbox(
                    f"{third_kind} Model",
                    models,
                    index=0,
                    key=f"{test_name}_third_model"
                )
        
        return {
            'use_ollama': use_ollama,
            'use_openai': use_openai,
            'third_kind': third_kind,
            'third_model': third_model
        }


class ConfigDisplay:
    """Unified configuration display component."""
    
    @staticmethod
    def render_collapsible(config_dict: Dict[str, Any], title: str = "Configuration"):
        """
        Renders configuration in a collapsible expander.
        
        Args:
            config_dict: Configuration dictionary to display
            title: Title for the expander
        """
        with st.expander(f"📋 {title}", expanded=False):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    st.markdown(f"**{key}:**")
                    for sub_key, sub_value in value.items():
                        st.text(f"  {sub_key}: {sub_value}")
                else:
                    st.text(f"{key}: {value}")


class TestResultTabs:
    """Unified result display with subtabs for all tests."""
    
    @staticmethod
    def render(df: pd.DataFrame, test_type: str = "classification", model_cols: List[str] = None, model_names: List[str] = None):
        """
        Renders results in organized subtabs: Summary, Performance, Errors, Raw Data.

        Args:
            df: Results dataframe
            test_type: Type of test ("classification", "pruning", "agent")
            model_cols: List of model column prefixes for classification tests
            model_names: Optional list of display names for models
        """
        if df.empty:
            st.warning("No results to display.")
            return

        # Create subtabs
        result_tabs = st.tabs(["📊 Summary", "🎯 Performance", "❌ Errors", "💾 Raw Data"])

        # Tab 1: Summary
        with result_tabs[0]:
            st.subheader("Summary Dashboard")

            if test_type == "classification":
                # Import visualization functions
                from utils.visualizations import render_kpi_metrics, render_model_comparison_chart
                
                # KPI metrics
                render_kpi_metrics(df, test_type="classification", model_cols=model_cols)

                st.divider()

                # Model comparison chart
                if model_cols:
                    render_model_comparison_chart(df, model_cols, model_names)

            elif test_type == "pruning":
                from utils.visualizations import render_kpi_metrics
                render_kpi_metrics(df, test_type="pruning")

        # Tab 2: Performance
        with result_tabs[1]:
            st.subheader("Detailed Performance Reports")

            if test_type == "classification" and "classification" in df.columns:
                y_true = df["classification"].tolist()

                # Generate reports for each model side-by-side
                if model_cols:
                    cols = st.columns(len(model_cols))

                    for i, (col, model_prefix) in enumerate(zip(cols, model_cols)):
                        result_col = f"classification_result_{model_prefix}"

                        if result_col in df.columns:
                            with col:
                                model_name = model_names[i] if model_names and i < len(model_names) else model_prefix
                                y_pred = df[result_col].tolist()

                                # Generate classification report
                                y_true_norm = [_normalize_label(s) for s in y_true]
                                y_pred_norm = [_normalize_label(s) for s in y_pred]

                                valid_indices = [idx for idx, (t, p) in enumerate(zip(y_true_norm, y_pred_norm)) if t and p]

                                if valid_indices:
                                    y_true_f = [y_true_norm[idx] for idx in valid_indices]
                                    y_pred_f = [y_pred_norm[idx] for idx in valid_indices]

                                    report_dict = classification_report(y_true_f, y_pred_f, output_dict=True, zero_division=0)

                                    st.markdown(f"**{model_name}**")
                                    st.dataframe(pd.DataFrame(report_dict).transpose().style.format("{:.2f}"), use_container_width=True)

        # Tab 3: Errors
        with result_tabs[2]:
            st.subheader("Error Analysis")

            if test_type == "classification" and "classification" in df.columns and model_cols:
                # Create error analysis dataframe
                df_analysis = df.copy()
                df_analysis['ground_truth'] = df_analysis['classification'].map(_normalize_label)

                # Add correctness columns for each model
                for model_prefix in model_cols:
                    result_col = f"classification_result_{model_prefix}"
                    if result_col in df_analysis.columns:
                        df_analysis[f'{model_prefix}_norm'] = df_analysis[result_col].map(_normalize_label)
                        df_analysis[f'{model_prefix}_correct'] = df_analysis[f'{model_prefix}_norm'] == df_analysis['ground_truth']

                # Filter options
                filter_options = ["All Errors"]

                # Add model-specific filters
                for model_prefix in model_cols:
                    if f'{model_prefix}_correct' in df_analysis.columns:
                        filter_options.append(f"{model_prefix.replace('_', ' ').title()} Errors Only")

                # Add comparison filters if multiple models
                if len(model_cols) >= 2:
                    filter_options.extend([
                        "All Models Incorrect",
                        "All Models Correct",
                        "Model Disagreements"
                    ])

                filter_choice = st.selectbox("Filter errors:", filter_options)

                # Apply filter
                if filter_choice == "All Errors":
                    # Show all rows where at least one model is incorrect
                    error_mask = False
                    for model_prefix in model_cols:
                        if f'{model_prefix}_correct' in df_analysis.columns:
                            error_mask = error_mask | ~df_analysis[f'{model_prefix}_correct']
                    view_df = df_analysis[error_mask] if isinstance(error_mask, pd.Series) else df_analysis

                elif "Errors Only" in filter_choice:
                    # Show errors for specific model
                    model_prefix = filter_choice.split(" Errors Only")[0].lower().replace(" ", "_")
                    if f'{model_prefix}_correct' in df_analysis.columns:
                        view_df = df_analysis[~df_analysis[f'{model_prefix}_correct']]
                    else:
                        view_df = df_analysis

                elif filter_choice == "All Models Incorrect":
                    # Show rows where all models are incorrect
                    all_incorrect = True
                    for model_prefix in model_cols:
                        if f'{model_prefix}_correct' in df_analysis.columns:
                            all_incorrect = all_incorrect & ~df_analysis[f'{model_prefix}_correct']
                    view_df = df_analysis[all_incorrect] if isinstance(all_incorrect, pd.Series) else df_analysis

                elif filter_choice == "All Models Correct":
                    # Show rows where all models are correct
                    all_correct = True
                    for model_prefix in model_cols:
                        if f'{model_prefix}_correct' in df_analysis.columns:
                            all_correct = all_correct & df_analysis[f'{model_prefix}_correct']
                    view_df = df_analysis[all_correct] if isinstance(all_correct, pd.Series) else df_analysis

                elif filter_choice == "Model Disagreements":
                    # Show rows where models disagree
                    if len(model_cols) >= 2:
                        # At least one correct and at least one incorrect
                        any_correct = False
                        any_incorrect = False
                        for model_prefix in model_cols:
                            if f'{model_prefix}_correct' in df_analysis.columns:
                                any_correct = any_correct | df_analysis[f'{model_prefix}_correct']
                                any_incorrect = any_incorrect | ~df_analysis[f'{model_prefix}_correct']
                        view_df = df_analysis[any_correct & any_incorrect]
                    else:
                        view_df = df_analysis
                else:
                    view_df = df_analysis

                # Display filtered results
                display_cols = ["query", "ground_truth"]
                for model_prefix in model_cols:
                    if f'{model_prefix}_norm' in view_df.columns:
                        display_cols.append(f'{model_prefix}_norm')
                    if f'{model_prefix}_correct' in view_df.columns:
                        display_cols.append(f'{model_prefix}_correct')

                existing_cols = [col for col in display_cols if col in view_df.columns]

                st.markdown(f"**Showing {len(view_df)} of {len(df_analysis)} rows**")
                st.dataframe(view_df[existing_cols], use_container_width=True)

        # Tab 4: Raw Data
        with result_tabs[3]:
            st.subheader("Complete Results Data")

            # Display truncated dataframe to avoid MemoryError with long rationales
            display_df = df.copy()
            rationale_cols = [c for c in display_df.columns if c.endswith("_rationale") or c == "judge_rationale"]
            for col in rationale_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].astype(str).str.slice(0, 150) + "..."
            st.dataframe(display_df, use_container_width=True)

            # Download options
            st.markdown("### Download Options")
            col1, col2 = st.columns(2)

            with col1:
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download as CSV",
                    data=csv_bytes,
                    file_name=f"{test_type}_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:
                json_str = df.to_json(orient='records', indent=2)
                st.download_button(
                    "⬇️ Download as JSON",
                    data=json_str,
                    file_name=f"{test_type}_results.json",
                    mime="application/json",
                    use_container_width=True
                )

