"""
Advanced Results Display for Test 6 Mode B

Provides 6-tab interface for comprehensive analysis results.
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from itertools import combinations
import re


def _get_analysis_value(analysis: Any, key: str, default: Any = None) -> Any:
    """Safely extract a value from VisualLLMAnalysis or dict."""
    if analysis is None:
        return default
    if hasattr(analysis, key):
        return getattr(analysis, key, default)
    if isinstance(analysis, dict):
        return analysis.get(key, default)
    return default


def _normalize_analysis_text(text: Any) -> Any:
    """Replace NaN markers in analysis text for readability."""
    if not isinstance(text, str):
        return text
    return re.sub(r"\bNaN\b", "0.0", text, flags=re.IGNORECASE)


def _prepare_artifact_list(raw_value: Any) -> List[str]:
    """Normalize artifact collections into a lowercase list."""
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if item]
    if isinstance(raw_value, str):
        return [raw_value.strip()]
    return []


def _calculate_image_pair_agreement(model_a: Dict[str, Any], model_b: Dict[str, Any]) -> Optional[float]:
    """Compute agreement score (0-1) between two models for a single image."""
    metrics: List[float] = []

    for rating_key in ("movement_rating", "visual_quality_rating", "artifact_presence_rating"):
        r1 = model_a.get(rating_key)
        r2 = model_b.get(rating_key)
        if r1 is not None and r2 is not None:
            try:
                diff = abs(float(r1) - float(r2))
            except (TypeError, ValueError):
                continue
            normalized = max(0.0, 1.0 - min(diff, 4.0) / 4.0)
            metrics.append(normalized)

    artifacts1 = {artifact.lower() for artifact in model_a.get("artifacts", []) if artifact}
    artifacts2 = {artifact.lower() for artifact in model_b.get("artifacts", []) if artifact}
    if artifacts1 or artifacts2:
        union = artifacts1 | artifacts2
        intersection = artifacts1 & artifacts2
        metrics.append(len(intersection) / len(union) if union else 1.0)

    conf1 = model_a.get("confidence")
    conf2 = model_b.get("confidence")
    if conf1 is not None and conf2 is not None:
        try:
            metrics.append(max(0.0, 1.0 - abs(float(conf1) - float(conf2))))
        except (TypeError, ValueError):
            pass

    if metrics:
        return float(np.mean(metrics))
    return None


def _compute_pairwise_agreement_matrix(image_data: List[Dict[str, Any]], models: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    """Aggregate pairwise agreement scores across all images."""
    pairwise_scores: Dict[str, Dict[str, List[float]]] = {
        m1: {m2: [] for m2 in models if m2 != m1}
        for m1 in models
    }

    for image in image_data:
        for m1, m2 in combinations(models, 2):
            model1 = image["models"].get(m1, {})
            model2 = image["models"].get(m2, {})
            if not (model1.get("success") and model2.get("success")):
                continue
            score = _calculate_image_pair_agreement(model1, model2)
            if score is not None:
                pairwise_scores[m1][m2].append(score)
                pairwise_scores[m2][m1].append(score)

    aggregated: Dict[str, Dict[str, Optional[float]]] = {m1: {} for m1 in models}
    for m1 in models:
        for m2 in models:
            if m1 == m2:
                aggregated[m1][m2] = 1.0
            else:
                samples = pairwise_scores.get(m1, {}).get(m2, [])
                aggregated[m1][m2] = float(np.mean(samples)) if samples else None
    return aggregated


def _compute_image_level_agreement(image_data: List[Dict[str, Any]], models: List[str]) -> List[Dict[str, float]]:
    """Calculate agreement and confidence stats per image."""
    image_scores: List[Dict[str, float]] = []

    for image in image_data:
        confidences = [
            model_info.get("confidence")
            for model_info in image["models"].values()
            if model_info.get("success") and model_info.get("confidence") is not None
        ]
        if not confidences:
            continue

        pair_scores: List[float] = []
        for m1, m2 in combinations(models, 2):
            model1 = image["models"].get(m1, {})
            model2 = image["models"].get(m2, {})
            if not (model1.get("success") and model2.get("success")):
                continue
            score = _calculate_image_pair_agreement(model1, model2)
            if score is not None:
                pair_scores.append(score)

        if pair_scores:
            image_scores.append({
                "image_name": image.get("image_name", "Unknown"),
                "mean_confidence": float(np.mean(confidences)),
                "agreement_score": float(np.mean(pair_scores))
            })

    return image_scores

from core.visual_meta_analysis import (
    plan_computational_analysis,
    execute_analysis_code,
    evaluate_visual_llm_performance
)
from core.visual_qa_interface import answer_followup_question, refine_and_reanalyze
from core.vision_visualizations import (
    create_rating_comparison_scatter,
    create_model_agreement_heatmap,
    create_confidence_distribution,
    create_performance_dashboard
)


def display_advanced_results(
    results: List[Dict[str, Any]],
    selected_models: List[str],
    preset_name: str,
    task_description: str,
    _CONFIG: Dict[str, Any]
) -> None:
    """
    Display advanced analysis results with 6-tab interface.

    Args:
        results: Visual LLM analysis results
        selected_models: Models used
        preset_name: Name of preset
        task_description: Analysis task description
        _CONFIG: Configuration dict with API keys
    """
    st.markdown("### 📊 Analysis Results")

    # Initialize session state for caching
    if 'test6_computational_results' not in st.session_state:
        st.session_state.test6_computational_results = None
    if 'test6_evaluation_results' not in st.session_state:
        st.session_state.test6_evaluation_results = None
    if 'test6_qa_history' not in st.session_state:
        st.session_state.test6_qa_history = []

    # Store analysis data for visualizations
    # Persist config and task description for downstream panels
    st.session_state.test6_config = _CONFIG
    st.session_state.test6_task_description = task_description

    st.session_state.test6_analysis_data = results

    # Add documentation popover for advanced results
    with st.popover("ℹ️ Advanced Results Guide", help="Click to understand the analysis tabs"):
        st.markdown("**Advanced Results Interface**")
        st.markdown("This interface provides 8 comprehensive analysis tabs:")

        st.markdown("**Tab Overview:**")
        st.code("""
📋 Summary & Performance
   - Quick metrics and model comparison
   - Success rates and average confidence
   - Performance dashboard

📊 Detailed Results
   - Full analysis for each image
   - Model-by-model breakdown
   - Downloadable CSV/JSON

📈 Visualizations
   - Rating comparison scatter plots
   - Model agreement heatmaps
   - Confidence distributions
   - Interactive charts

🎯 Synthesis & Insights
   - Cross-model synthesis
   - Key findings and patterns
   - Actionable recommendations

🧠 Computational Analysis
   - LLM-generated Python code analysis
   - Statistical computations
   - Custom metrics and visualizations

🏆 Model Evaluation
   - LLM judge evaluation
   - Model rankings and scores
   - Performance comparison

💬 Interactive Q&A
   - Ask questions about results
   - Automatic image context selection
   - Follow-up analysis suggestions

💾 Export
   - Download results in multiple formats
   - CSV, JSON, and full reports
        """, language="text")

        st.markdown("**Key Features:**")
        st.markdown("- **Result Caching**: Computational and evaluation results are cached")
        st.markdown("- **Image Display**: Referenced images automatically shown in Q&A")
        st.markdown("- **Downloadable**: All results exportable as CSV/JSON")
        st.markdown("- **Interactive**: Click through tabs to explore different aspects")

    # Create tabs - merged basic and advanced
    result_tabs = st.tabs([
        "📋 Summary & Performance",
        "📊 Detailed Results",
        "📈 Visualizations",
        "🎯 Synthesis & Insights",  # NEW: Comprehensive synthesis
        "🧠 Computational Analysis",
        "🏆 Model Evaluation",
        "💬 Interactive Q&A",
        "💾 Export"
    ])

    # Tab 1: Summary & Performance (merged basic summary)
    with result_tabs[0]:
        _display_summary_and_performance_tab(results, selected_models)

    # Tab 2: Detailed Results (merged basic per-image results)
    with result_tabs[1]:
        _display_detailed_results_tab(results, selected_models)

    # Tab 3: Visualizations (basic charts)
    with result_tabs[2]:
        _display_visualizations_tab(results, selected_models)

    # Tab 4: Synthesis & Insights (NEW)
    with result_tabs[3]:
        _display_synthesis_tab(results, task_description, selected_models)

    # Tab 5: Computational Analysis (advanced)
    with result_tabs[4]:
        _display_computational_analysis_tab(
            results,
            task_description,
            _CONFIG
        )

    # Tab 6: Model Evaluation (advanced)
    with result_tabs[5]:
        _display_model_evaluation_tab(
            results,
            task_description,
            _CONFIG
        )

    # Tab 7: Interactive Q&A (advanced)
    with result_tabs[6]:
        _display_qa_tab(
            results,
            task_description,
            selected_models,
            _CONFIG
        )

    # Tab 8: Export
    with result_tabs[7]:
        _display_export_tab(results, selected_models, preset_name)


def _display_summary_and_performance_tab(results: List[Dict[str, Any]], selected_models: List[str]) -> None:
    """Display summary metrics and model performance (merged basic + advanced)."""
    st.markdown("#### 📊 Analysis Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Images Analyzed", len(results))

    with col2:
        st.metric("Models Used", len(selected_models))

    with col3:
        total_analyses = len(results) * len(selected_models)
        st.metric("Total Analyses", total_analyses)

    with col4:
        # Show total cost from cost tracker
        if 'cost_tracker' in st.session_state:
            ct = st.session_state.cost_tracker
            st.metric("Total Cost", f"${ct.totals['total_cost_usd']:.4f}")
        else:
            st.metric("Total Cost", "N/A")

    # Model performance table
    st.markdown("#### 🤖 Model Performance")

    model_data = []

    # Extract actual model names from results
    actual_model_names = {}
    if results:
        first_result = results[0]
        model_results = first_result.get("model_results", {})
        for model_name in model_results.keys():
            # Map model_id to actual model name
            for model_id in selected_models:
                if model_id in model_name.lower() or model_id.replace("gpt5", "gpt") in model_name.lower():
                    actual_model_names[model_id] = model_name
                    break

    for model_id in selected_models:
        success_count = 0
        actual_name = actual_model_names.get(model_id, model_id.upper())

        for result in results:
            model_results = result.get("model_results", {})
            for model_name, analysis in model_results.items():
                if model_id in model_name.lower() or model_id.replace("gpt5", "gpt") in model_name.lower():
                    success_count += 1

        # Get cost data from session state if available
        total_cost = 0.0
        total_tokens = 0
        if 'cost_tracker' in st.session_state:
            cost_tracker = st.session_state.cost_tracker
            for call in cost_tracker.by_call:
                # Match against actual model name (e.g., "gpt-5-nano", "gemini-2.5-flash-lite")
                call_model = call.get("model", "")
                if model_id in call_model.lower() or model_id.replace("gpt5", "gpt") in call_model.lower():
                    total_cost += call.get("total_cost_usd", 0)  # ✅ Fixed: was "cost_usd"
                    total_tokens += call.get("total_tokens", 0)

        model_data.append({
            "Model": actual_name,  # Use actual model name instead of generic ID
            "Successful": success_count,
            "Total Tokens": f"{total_tokens:,}" if total_tokens > 0 else "N/A",
            "Total Cost": f"${total_cost:.4f}" if total_cost > 0 else "N/A"
        })

    st.dataframe(pd.DataFrame(model_data), use_container_width=True)


def _display_detailed_results_tab(results: List[Dict[str, Any]], selected_models: List[str]) -> None:
    """Display detailed per-image results with ground truth comparison (merged basic + advanced)."""

    # Check if ground truth is available
    has_ground_truth = 'test6_ground_truths' in st.session_state and st.session_state.test6_ground_truths

    if has_ground_truth:
        st.info("🧠 **Master LLM Ground Truth Available** - Results are compared against master expectations")

    st.markdown("#### 📊 Results Dataframe")

    # Create comprehensive dataframe
    df_rows = []
    for result in results:
        image_name = result.get("image_name", "Unknown")

        for model_name, analysis in result.get("model_results", {}).items():
            # Extract data from both Pydantic models and dictionaries
            response_text = ""
            confidence = 0.0

            # Try Pydantic model attributes
            if hasattr(analysis, 'rationale'):
                response_text = analysis.rationale
            elif hasattr(analysis, 'raw_response'):
                response_text = analysis.raw_response
            # Try dictionary keys
            elif isinstance(analysis, dict):
                response_text = analysis.get('rationale', '') or analysis.get('raw_response', '')
                confidence = analysis.get('confidence', 0.0)

            # Get confidence from Pydantic model if not already set
            if confidence == 0.0 and hasattr(analysis, 'confidence'):
                confidence = analysis.confidence

            df_rows.append({
                "Image": image_name,
                "Model": model_name,
                "Response": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                "Confidence": f"{confidence:.2%}" if confidence else "N/A",
                "Full Response": response_text
            })

    df = pd.DataFrame(df_rows)

    # Display with filtering
    st.dataframe(
        df[["Image", "Model", "Response", "Confidence"]],
        use_container_width=True,
        height=400
    )

    # Download full dataframe
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Dataframe (CSV)",
        data=csv,
        file_name=f"visual_llm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    st.divider()

    # Per-image detailed results
    st.markdown("#### 🖼️ Per-Image Detailed Results")

    for idx, result in enumerate(results):
        with st.expander(f"📸 {result['image_name']}", expanded=(idx == 0)):
            col1, col2 = st.columns([1, 2])

            with col1:
                try:
                    st.image(result['image_path'], caption=result['image_name'], use_container_width=True)
                except:
                    st.info("Image preview not available")

                # Show ground truth if available
                if has_ground_truth:
                    ground_truths = st.session_state.test6_ground_truths
                    image_path = result.get('image_path')

                    if image_path in ground_truths:
                        gt = ground_truths[image_path]

                        st.markdown("**🧠 Master LLM Ground Truth:**")
                        st.success(gt.get('expected_analysis', 'N/A')[:200] + "...")

                        with st.expander("📋 Full Ground Truth"):
                            st.markdown(f"**Expected Analysis:** {gt.get('expected_analysis')}")
                            st.markdown(f"**Key Findings:** {', '.join(gt.get('key_findings', []))}")
                            st.markdown(f"**Difficulty:** {gt.get('difficulty_level', 'N/A')}")
                            if gt.get('expected_rating'):
                                st.markdown(f"**Expected Rating:** {gt.get('expected_rating')}/5")

            with col2:
                for model_name, analysis in result.get("model_results", {}).items():
                    st.markdown(f"**{model_name}:**")

                    # Handle both Pydantic models and dictionaries
                    response_text = None
                    confidence = None

                    # Try Pydantic model attributes
                    if hasattr(analysis, 'rationale'):
                        response_text = analysis.rationale
                    elif hasattr(analysis, 'raw_response'):
                        response_text = analysis.raw_response
                    # Try dictionary keys
                    elif isinstance(analysis, dict):
                        response_text = analysis.get('rationale') or analysis.get('raw_response')
                        confidence = analysis.get('confidence')

                    # Get confidence
                    if confidence is None and hasattr(analysis, 'confidence'):
                        confidence = analysis.confidence

                    # Display response
                    if response_text:
                        st.info(response_text)
                    else:
                        # Debug: show what we actually have
                        st.warning("No response available")
                        with st.expander("🔍 Debug Info"):
                            st.write("Analysis type:", type(analysis))
                            if isinstance(analysis, dict):
                                st.write("Keys:", list(analysis.keys()))
                                st.json(analysis)
                            else:
                                st.write("Attributes:", [a for a in dir(analysis) if not a.startswith('_')])

                    # Display confidence
                    if confidence is not None:
                        st.caption(f"Confidence: {confidence:.2%}")


def _display_visualizations_tab(results: List[Dict[str, Any]], selected_models: List[str]) -> None:
    """Display basic visualizations from analysis results."""
    st.markdown("#### Analysis Visualizations")

    viz_data = _extract_visualization_data(results)

    if not viz_data:
        st.warning("No data available for visualizations")
        return

    _render_visualization_tabs(viz_data, key_prefix="test6_basic")


def _display_synthesis_tab(
    results: List[Dict[str, Any]],
    task_description: str,
    selected_models: List[str]
) -> None:
    """Display comprehensive synthesis with insights and recommendations."""
    from ui.test6_synthesis_display import display_synthesis_results

    st.markdown("#### 🎯 Comprehensive Analysis Synthesis")
    st.markdown("""
    This tab provides a comprehensive synthesis of all analysis results including:
    - **Model Rankings**: Performance-based ranking with complementary strengths
    - **Agreement Analysis**: Normalized inter-model agreement metrics
    - **Actionable Insights**: Prioritized recommendations based on results
    - **Prompt Optimization**: Model-specific prompt improvements
    - **Advanced Visualizations**: Correlation and agreement heatmaps
    """)

    # Display synthesis
    display_synthesis_results(results, task_description, selected_models)



def _render_visualization_tabs(viz_data: Dict[str, Any], key_prefix: str) -> None:
    """Render shared visualization tabs for both basic and computational views."""
    viz_tabs = st.tabs([
        "Model Agreement",
        "Confidence Distribution",
        "Confidence vs Agreement",
        "Performance Metrics",
        "Per-Image Analysis",
        "Overall Leaderboard"
    ])

    with viz_tabs[0]:
        st.markdown("#### Model Agreement Matrix")
        st.caption("Shows how often models agree on their analyses (missing pairs default to 0.0%)")
        agreement_fig = _create_model_agreement_heatmap(viz_data)
        if agreement_fig:
            st.plotly_chart(agreement_fig, use_container_width=True, key=f"{key_prefix}_agreement_chart")
        else:
            st.info("Need at least two models with overlapping results")

    with viz_tabs[1]:
        st.markdown("#### Confidence Score Distribution")
        st.caption("Distribution of confidence scores across all models")
        confidence_fig = _create_confidence_distribution_chart(viz_data)
        if confidence_fig:
            st.plotly_chart(confidence_fig, use_container_width=True, key=f"{key_prefix}_confidence_chart")
        else:
            st.info("No confidence data available")

    with viz_tabs[2]:
        st.markdown("#### Confidence vs Agreement")
        st.caption("Correlation between model confidence and agreement levels")
        correlation_fig = _create_confidence_agreement_chart(viz_data)
        if correlation_fig:
            st.plotly_chart(correlation_fig, use_container_width=True, key=f"{key_prefix}_confidence_agreement_chart")
        else:
            st.info("Not enough overlapping data to compute correlation")

    with viz_tabs[3]:
        st.markdown("#### Model Performance Comparison")
        st.caption("Comparative performance metrics across models")
        performance_fig = _create_performance_comparison_chart(viz_data)
        if performance_fig:
            st.plotly_chart(performance_fig, use_container_width=True, key=f"{key_prefix}_performance_chart")
        else:
            st.info("No performance data available")

    with viz_tabs[5]:
        st.markdown("#### Overall Model Leaderboard")
        st.caption("Uses ground truth if available; otherwise falls back to a consensus-based proxy.")
        ground_truths = st.session_state.get('test6_ground_truths')
        gt_lb = _compute_ground_truth_leaderboard(viz_data, ground_truths)
        import pandas as pd
        if gt_lb and gt_lb.get("leaderboard"):
            best = gt_lb.get("best_model")
            if best:
                st.success(f"Best overall (ground truth): {best}")
            df_gt = pd.DataFrame(gt_lb["leaderboard"])
            st.dataframe(df_gt, use_container_width=True, hide_index=True)
            st.caption("Metric: lowest average absolute error to ground truth across available ratings.")
        else:
            st.info("No usable ground truth found; using consensus proxy.")
            leaderboard = _compute_consensus_leaderboard(viz_data)
            best = leaderboard.get("best_model")
            if best:
                st.success(f"Best overall (consensus proxy): {best}")
            table = leaderboard.get("leaderboard", [])
            if table:
                df_lb = pd.DataFrame(table)
                st.dataframe(df_lb, use_container_width=True, hide_index=True)
        st.markdown("---")
        st.markdown("#### 🧑‍⚖️ LLM Judge (Compact)")
        st.caption("Quick judge-based ranking alongside the leaderboard.")
        eval_cached = st.session_state.get('test6_evaluation_results')
        if eval_cached:
            best = eval_cached.get('best_model')
            if best:
                st.success(f"Judge says best: {best}")
            ranks = eval_cached.get('model_rankings') or []
            if ranks:
                import pandas as pd
                df = pd.DataFrame([
                    {"Model": r.get("model"), "Score": r.get("score"), "Rationale": r.get("rationale", "")}
                    for r in ranks[:5]
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
            if st.button("🔄 Re-run Judge (Compact)", key=f"{key_prefix}_leaderboard_rerun_judge"):
                st.session_state.test6_evaluation_results = None
                st.rerun()
        else:
            if st.button("🚀 Run LLM Judge (Compact)", key=f"{key_prefix}_leaderboard_run_judge"):
                with st.spinner("Running judge..."):
                    results = st.session_state.get('test6_analysis_data') or []
                    task_desc = st.session_state.get('test6_task_description') or ""
                    comp_results = st.session_state.get('test6_computational_results')
                    cfg = st.session_state.get('test6_config') or {}
                    import asyncio
                    from core.visual_meta_analysis import evaluate_visual_llm_performance
                    evaluation = asyncio.run(evaluate_visual_llm_performance(
                        visual_llm_outputs=results,
                        task_description=task_desc,
                        computational_results=(comp_results.get('execution') if comp_results else None),
                        judge_model="gpt-5-nano",
                        openai_api_key=cfg.get('OPENAI_API_KEY')
                    ))
                    st.session_state.test6_evaluation_results = evaluation
                    st.success("✅ Judge complete.")
                    best = evaluation.get('best_model')
                    if best:
                        st.success(f"Judge says best: {best}")
                    ranks = evaluation.get('model_rankings') or []
                    if ranks:
                        import pandas as pd
                        df = pd.DataFrame([
                            {"Model": r.get("model"), "Score": r.get("score"), "Rationale": r.get("rationale", "")}
                            for r in ranks[:5]
                        ])
                        st.dataframe(df, use_container_width=True, hide_index=True)



    with viz_tabs[4]:
        st.markdown("#### Per-Image Analysis")
        st.caption("Detailed breakdown by image")
        detailed_fig = _create_per_image_analysis_chart(viz_data)
        if detailed_fig:
            st.plotly_chart(detailed_fig, use_container_width=True, key=f"{key_prefix}_detailed_chart")
        else:
            st.info("No detailed data available")


def _display_computational_analysis_tab(
    results: List[Dict[str, Any]],
    task_description: str,
    _CONFIG: Dict[str, Any]
) -> None:
    """Display computational analysis with code execution."""
    st.markdown("#### 📈 Computational Analysis")

    st.info("💡 This tab uses LLM-generated code to perform statistical and computational analysis on the visual LLM outputs.")

    # Add documentation popover for Computational Analysis
    with st.popover("📖 How Computational Analysis Works", help="Click for analysis details"):
        st.markdown("**Computational Analysis with Code Execution**")
        st.markdown("This tab uses LLMs to generate and execute Python code for analyzing visual LLM outputs.")

        st.markdown("**Two-Stage Orchestration:**")
        st.code("""
# Stage 1: Planning (GPT-5-mini)
plan = await plan_computational_analysis(
    visual_llm_outputs=results,
    task_description=task_description,
    planner_model="gpt-5-mini"
)

# Planner generates:
# - analysis_plan: What to analyze and why
# - expected_outputs: What results to expect
# - python_code: Executable Python code

# Example generated code:
'''
import pandas as pd
import numpy as np
from collections import Counter

# Extract all detected objects
all_objects = []
for result in visual_llm_outputs:
    for model_result in result['model_results'].values():
        objects = model_result.get('detected_objects', [])
        all_objects.extend(objects)

# Frequency analysis
object_freq = Counter(all_objects)
top_10 = object_freq.most_common(10)

# Model agreement analysis
# ... more analysis code ...
'''

# Stage 2: Execution (Gemini Code Execution)
execution_results = await execute_analysis_code(
    python_code=plan['python_code'],
    visual_llm_outputs=results,
    use_gemini_execution=True
)

# Gemini Code Execution:
# - Runs code in sandboxed environment
# - Has access to visual_llm_outputs data
# - Returns execution results and any outputs
        """, language="python")

        st.markdown("**Key Functions:**")
        st.code("""
# Planning function
async def plan_computational_analysis(
    visual_llm_outputs, task_description, planner_model
):
    # LLM analyzes the data structure
    # Generates appropriate analysis code
    # Returns plan with executable Python code

# Execution function
async def execute_analysis_code(
    python_code, visual_llm_outputs, use_gemini_execution
):
    # Gemini Code Execution API runs the code
    # Returns results, stdout, and any errors
    # Handles visualization generation
        """, language="python")

        st.markdown("**Example Analyses:**")
        st.markdown("- **Object Frequency**: Count detected objects across all images")
        st.markdown("- **Color Analysis**: Extract and analyze dominant colors")
        st.markdown("- **Model Agreement**: Calculate inter-model agreement scores")
        st.markdown("- **Confidence Calibration**: Analyze confidence vs. accuracy")
        st.markdown("- **Custom Metrics**: Any statistical analysis you need")

        st.markdown("**Note:** Results are cached - re-run if you want fresh analysis.")

    # Check if analysis already run
    if st.session_state.test6_computational_results is not None:
        st.success("✅ Analysis already completed. Showing cached results.")
        _display_computational_results(st.session_state.test6_computational_results)

        if st.button("🔄 Re-run Analysis", key="rerun_computational"):
            st.session_state.test6_computational_results = None
            st.rerun()

        return

    # Run analysis button
    if st.button("🚀 Run Computational Analysis", type="primary", key="run_computational"):
        with st.spinner("🧠 Planning analysis with GPT-5..."):
            # Plan analysis
            plan = asyncio.run(plan_computational_analysis(
                visual_llm_outputs=results,
                task_description=task_description,
                planner_model="gpt-5-mini",
                openai_api_key=_CONFIG.get('OPENAI_API_KEY')
            ))

            st.success("✅ Analysis plan generated!")

            with st.expander("📋 Analysis Plan", expanded=True):
                st.markdown(f"**Plan:** {plan.get('analysis_plan', 'N/A')}")
                st.markdown(f"**Expected Outputs:** {plan.get('expected_outputs', 'N/A')}")
                st.code(plan.get('python_code', '# No code generated'), language="python")

        with st.spinner("⚙️ Executing code with Gemini Code Execution..."):
            # Execute code using Gemini's code execution framework
            execution_results = asyncio.run(execute_analysis_code(
                python_code=plan.get('python_code', ''),
                visual_llm_outputs=results,
                use_gemini_execution=True,
                gemini_api_key=_CONFIG.get('GEMINI_API_KEY')
            ))

            if execution_results.get('success'):
                st.success("✅ Analysis executed successfully with Gemini Code Execution!")

                # Show generated code if available
                if execution_results.get('generated_code'):
                    with st.expander("🔍 View Generated Code"):
                        st.code(execution_results.get('generated_code'), language="python")

                # Cache results
                st.session_state.test6_computational_results = {
                    "plan": plan,
                    "execution": execution_results
                }

                _display_computational_results(st.session_state.test6_computational_results)
            else:
                st.error(f"❌ Execution failed: {execution_results.get('error', 'Unknown error')}")

                # Show generated code even if execution failed
                if execution_results.get('generated_code'):
                    with st.expander("🔍 View Generated Code (Failed)"):
                        st.code(execution_results.get('generated_code'), language="python")


def _display_computational_results(comp_results: Dict[str, Any]) -> None:
    """Display computational analysis results."""
    st.markdown("### 📊 Analysis Results")

    execution = comp_results.get("execution", {})
    results_text = execution.get("results", "No results")
    execution_method = execution.get("execution_method", "unknown")

    # Show execution method badge
    if execution_method == "gemini_code_execution":
        st.success("✅ Executed with Gemini Code Execution Framework")
    elif execution_method == "local_sandboxed":
        st.info("ℹ️ Executed locally (sandboxed)")

    normalized_results_text = _normalize_analysis_text(results_text)
    st.markdown(normalized_results_text)

    st.divider()

    # Create visualizations
    st.markdown("### 📈 Visualizations")

    # Get the original data that was analyzed
    if 'test6_analysis_data' in st.session_state:
        results = st.session_state.test6_analysis_data
        _create_computational_visualizations(results, comp_results)
    else:
        st.info("💡 Run analysis first to generate visualizations")


def _create_computational_visualizations(
    results: List[Dict[str, Any]],
    comp_results: Dict[str, Any]
) -> None:
    """Create Plotly visualizations from computational analysis results."""

    viz_data = _extract_visualization_data(results)

    if not viz_data:
        st.warning("No data available for visualizations")
        return

    _render_visualization_tabs(viz_data, key_prefix="test6_comp")


def _extract_visualization_data(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract data from results for visualizations."""

    all_models = sorted({
        model_name
        for result in results
        for model_name in result.get("model_results", {}).keys()
    })

    if not all_models:
        return {}

    image_data: List[Dict[str, Any]] = []

    for result in results:
        image_name = result.get("image_name", "Unknown")
        model_results = result.get("model_results", {})

        image_entry = {
            "image_name": image_name,
            "models": {}
        }

        for model_name in all_models:
            analysis = model_results.get(model_name)

            if analysis is not None:
                confidence = _get_analysis_value(analysis, "confidence", 0.0) or 0.0
                response = _get_analysis_value(analysis, "rationale") or _get_analysis_value(analysis, "raw_response", "")
                movement = _get_analysis_value(analysis, "movement_rating")
                visual_rating = _get_analysis_value(analysis, "visual_quality_rating")
                artifact_rating = _get_analysis_value(analysis, "artifact_presence_rating")
                artifacts = _prepare_artifact_list(_get_analysis_value(analysis, "detected_artifacts", []))

                image_entry["models"][model_name] = {
                    "confidence": float(confidence) if confidence is not None else 0.0,
                    "response": response or "",
                    "movement_rating": movement,
                    "visual_quality_rating": visual_rating,
                    "artifact_presence_rating": artifact_rating,
                    "artifacts": artifacts,
                    "success": True
                }
            else:
                image_entry["models"][model_name] = {
                    "confidence": 0.0,
                    "response": "",
                    "movement_rating": None,
                    "visual_quality_rating": None,
                    "artifact_presence_rating": None,
                    "artifacts": [],
                    "success": False
                }

        image_data.append(image_entry)

    pairwise_agreement = _compute_pairwise_agreement_matrix(image_data, all_models)
    image_agreement_scores = _compute_image_level_agreement(image_data, all_models)

    return {
        "all_models": all_models,
        "image_data": image_data,
        "num_images": len(results),
        "num_models": len(all_models),
        "pairwise_agreement": pairwise_agreement,
        "image_agreement_scores": image_agreement_scores
    }


def _create_model_agreement_heatmap(viz_data: Dict[str, Any]) -> Optional[go.Figure]:
    """Create heatmap showing model agreement."""

    all_models = viz_data.get("all_models", [])
    pairwise = viz_data.get("pairwise_agreement", {})

    if len(all_models) < 2 or not pairwise:
        return None

    agreement_matrix: List[List[float]] = []
    text_matrix: List[List[str]] = []

    for model1 in all_models:
        row: List[float] = []
        text_row: List[str] = []
        for model2 in all_models:
            if model1 == model2:
                row.append(100.0)
                text_row.append("100.0%")
                continue

            value = pairwise.get(model1, {}).get(model2)
            if value is None:
                row.append(0.0)
                text_row.append("0.0% (no overlap)")
            else:
                pct = max(0.0, min(100.0, value * 100))
                row.append(pct)
                text_row.append(f"{pct:.1f}%")

        agreement_matrix.append(row)
        text_matrix.append(text_row)

    fig = go.Figure(data=go.Heatmap(
        z=agreement_matrix,
        x=all_models,
        y=all_models,
        colorscale='RdYlGn',
        zmin=0,
        zmax=100,
        text=text_matrix,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='Model 1: %{y}<br>Model 2: %{x}<br>Agreement: %{z:.1f}%<extra></extra>',
        colorbar=dict(title="Agreement %")
    ))

    fig.update_layout(
        title="Normalized Model Agreement Matrix",
        xaxis_title="Model",
        yaxis_title="Model",
        height=500
    )

    return fig


def _create_confidence_distribution_chart(viz_data: Dict[str, Any]) -> Optional[go.Figure]:
    """Create distribution chart for confidence scores."""

    all_models = viz_data.get("all_models", [])
    image_data = viz_data.get("image_data", [])

    # Collect confidence scores by model
    confidence_by_model = {model: [] for model in all_models}

    for image in image_data:
        for model_name, model_data in image["models"].items():
            if model_data.get("success"):
                confidence = model_data.get("confidence", 0)
                confidence_by_model[model_name].append(confidence)

    # Check if we have data
    if not any(confidence_by_model.values()):
        return None

    # Create box plot
    fig = go.Figure()

    for model_name in all_models:
        confidences = confidence_by_model[model_name]
        if confidences:
            fig.add_trace(go.Box(
                y=confidences,
                name=model_name,
                boxmean='sd'  # Show mean and standard deviation
            ))

    fig.update_layout(
        title="Confidence Score Distribution by Model",
        yaxis_title="Confidence Score",
        xaxis_title="Model",
        height=500,
        showlegend=False
    )

    return fig


def _create_confidence_agreement_chart(viz_data: Dict[str, Any]) -> Optional[go.Figure]:
    """Create scatter plot showing correlation between confidence and agreement."""

    all_models = viz_data.get("all_models", [])
    image_data = viz_data.get("image_data", [])

    if len(all_models) < 2 or not image_data:
        return None

    # Collect data points
    data_points = []

    for image in image_data:
        # Calculate average confidence for this image
        confidences = []
        for model_name in all_models:
            model_data = image["models"].get(model_name, {})
            if model_data.get("success"):
                conf = model_data.get("confidence", 0)
                confidences.append(conf)

        if len(confidences) < 2:
            continue

        avg_confidence = np.mean(confidences)

        # Calculate agreement (inverse of std dev)
        agreement = 1.0 - min(np.std(confidences), 1.0)

        data_points.append({
            'image': image.get('image_name', 'unknown'),
            'avg_confidence': avg_confidence,
            'agreement': agreement,
            'num_models': len(confidences)
        })

    if len(data_points) < 3:
        return None

    # Create scatter plot
    df = pd.DataFrame(data_points)

    fig = px.scatter(
        df,
        x='avg_confidence',
        y='agreement',
        size='num_models',
        hover_data=['image'],
        title="Confidence vs Agreement Correlation",
        labels={
            'avg_confidence': 'Average Confidence',
            'agreement': 'Model Agreement',
            'num_models': 'Number of Models'
        }
    )

    # Add trend line if we have enough data
    if len(df) > 2:
        try:
            from scipy.stats import pearsonr

            # Check if we have enough variance in the data
            if df['avg_confidence'].std() > 0.01 and df['agreement'].std() > 0.01:
                corr, p_value = pearsonr(df['avg_confidence'], df['agreement'])

                # Add correlation annotation
                fig.add_annotation(
                    text=f"Pearson r = {corr:.3f} (p = {p_value:.3f})",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )

                # Add trend line
                z = np.polyfit(df['avg_confidence'], df['agreement'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(df['avg_confidence'].min(), df['avg_confidence'].max(), 100)
                fig.add_trace(go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name='Trend',
                    line=dict(dash='dash', color='red')
                ))
        except (np.linalg.LinAlgError, ValueError, RuntimeError, ImportError):
            # Skip trend line if scipy not available or calculation fails
            pass

    fig.update_layout(height=500)

    return fig


def _create_performance_comparison_chart(viz_data: Dict[str, Any]) -> Optional[go.Figure]:
    """Create bar chart comparing model performance metrics."""

    all_models = viz_data.get("all_models", [])
    image_data = viz_data.get("image_data", [])

    # Calculate metrics for each model
    metrics = []

    for model_name in all_models:
        successes = 0
        total_confidence = 0
        count = 0

        for image in image_data:
            model_data = image["models"].get(model_name, {})
            if model_data.get("success"):
                successes += 1
                total_confidence += model_data.get("confidence", 0)
                count += 1

        avg_confidence = (total_confidence / count) if count > 0 else 0
        success_rate = (successes / len(image_data) * 100) if image_data else 0

        metrics.append({
            "model": model_name,
            "success_rate": success_rate,
            "avg_confidence": avg_confidence,
            "total_analyses": successes
        })

    if not metrics:
        return None

    # Create grouped bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Success Rate (%)',
        x=[m["model"] for m in metrics],
        y=[m["success_rate"] for m in metrics],
        text=[f"{m['success_rate']:.1f}%" for m in metrics],
        textposition='auto',
        marker_color='lightblue'
    ))

    fig.add_trace(go.Bar(
        name='Avg Confidence (×100)',
        x=[m["model"] for m in metrics],
        y=[m["avg_confidence"] * 100 for m in metrics],
        text=[f"{m['avg_confidence']:.2f}" for m in metrics],
        textposition='auto',
        marker_color='lightgreen'
    ))

    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Percentage",
        barmode='group',
        height=500
    )

    return fig


def _create_per_image_analysis_chart(viz_data: Dict[str, Any]) -> Optional[go.Figure]:
    """Create stacked bar chart showing per-image model performance."""

    all_models = viz_data.get("all_models", [])
    image_data = viz_data.get("image_data", [])

    if not image_data:
        return None

    # Limit to first 20 images for readability
    display_images = image_data[:20]

    # Create traces for each model
    fig = go.Figure()

    for model_name in all_models:
        confidences = []
        image_names = []

        for image in display_images:
            model_data = image["models"].get(model_name, {})
            confidence = model_data.get("confidence", 0) if model_data.get("success") else 0

            confidences.append(confidence)
            image_names.append(image["image_name"])

        fig.add_trace(go.Bar(
            name=model_name,
            x=image_names,
            y=confidences,
            text=[f"{c:.2f}" for c in confidences],
            textposition='auto'
        ))

    fig.update_layout(
        title=f"Confidence Scores by Image (First {len(display_images)} images)",
        xaxis_title="Image",
        yaxis_title="Confidence Score",
        barmode='group',
        height=500,
        xaxis_tickangle=-45
    )

    return fig



def _compute_consensus_leaderboard(viz_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute an overall model leaderboard using consensus as a proxy for correctness.
    For each rating type (movement, visual_quality, artifact_presence), we compute a
    per-image consensus using the median across models. A model is counted as "agree"
    if its rating is within a tolerance of the consensus. We aggregate across images
    and rating types to get a consensus agreement percentage per model.
    """
    from statistics import median

    all_models = viz_data.get("all_models", [])
    image_data = viz_data.get("image_data", [])
    if not all_models or not image_data:
        return {"best_model": None, "leaderboard": []}

    rating_keys = ["movement_rating", "visual_quality_rating", "artifact_presence_rating"]
    tol_by_key = {  # tolerance per rating dimension
        "movement_rating": 0.5,
        "visual_quality_rating": 0.5,
        "artifact_presence_rating": 0.5,
    }

    stats = {m: {"agree": 0, "total": 0, "conf_sum": 0.0, "conf_count": 0} for m in all_models}

    for image in image_data:
        models_dict = image.get("models", {})
        # Precompute consensus per rating type for this image
        consensus: Dict[str, Optional[float]] = {}
        for key in rating_keys:
            vals = [md.get(key) for md in models_dict.values() if md.get(key) is not None]
            consensus[key] = median(vals) if len(vals) >= 2 else None

        for model_name in all_models:
            md = models_dict.get(model_name, {})
            # Confidence accumulation
            if md.get("success"):
                conf = md.get("confidence")
                if isinstance(conf, (int, float)):
                    stats[model_name]["conf_sum"] += float(conf)
                    stats[model_name]["conf_count"] += 1
            # Agreement accumulation
            for key in rating_keys:
                c = consensus.get(key)
                v = md.get(key)
                if c is None or v is None:
                    continue
                stats[model_name]["total"] += 1
                if abs(float(v) - float(c)) <= tol_by_key.get(key, 0.5):
                    stats[model_name]["agree"] += 1

    leaderboard: List[Dict[str, Any]] = []
    for model_name in all_models:
        total = stats[model_name]["total"]
        agree = stats[model_name]["agree"]
        conf_count = stats[model_name]["conf_count"]
        avg_conf = (stats[model_name]["conf_sum"] / conf_count) if conf_count else 0.0
        pct = (agree / total * 100.0) if total else 0.0
        leaderboard.append({
            "Model": model_name,
            "Consensus Agreement %": round(pct, 1),
            "Avg Confidence": round(avg_conf, 3),
            "Samples": int(total),
        })

    # Sort by agreement desc, then avg_conf desc, then samples desc
    leaderboard.sort(key=lambda r: (r["Consensus Agreement %"], r["Avg Confidence"], r["Samples"]), reverse=True)
    best_model = leaderboard[0]["Model"] if leaderboard else None
    return {"best_model": best_model, "leaderboard": leaderboard}

def _compute_ground_truth_leaderboard(viz_data: Dict[str, Any], ground_truths: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """If ground truths are available per image (e.g., rated targets), compute a
    per-model average absolute error against ground truth across rating types.
    Returns None if no usable ground truth match is found.
    Expected ground_truths structure: { image_name: { movement_rating?, visual_quality_rating?, artifact_presence_rating? } }
    """
    if not ground_truths:
        return None
    all_models = viz_data.get("all_models", [])
    image_data = viz_data.get("image_data", [])
    if not all_models or not image_data:
        return None

    rating_keys = ["movement_rating", "visual_quality_rating", "artifact_presence_rating"]
    sums = {m: 0.0 for m in all_models}
    counts = {m: 0 for m in all_models}

    for image in image_data:
        img = image.get("image_name")
        gt = ground_truths.get(img) if isinstance(ground_truths, dict) else None
        if not isinstance(gt, dict):
            continue
        for key in rating_keys:
            gt_val = gt.get(key)
            if gt_val is None:
                continue
            for m in all_models:
                v = image.get("models", {}).get(m, {}).get(key)
                if v is None:
                    continue
                try:
                    diff = abs(float(v) - float(gt_val))
                except Exception:
                    continue
                sums[m] += diff
                counts[m] += 1

    usable = any(counts[m] > 0 for m in all_models)
    if not usable:
        return None

    leaderboard: List[Dict[str, Any]] = []
    for m in all_models:
        c = counts[m]
        avg_abs_err = (sums[m] / c) if c else None
        leaderboard.append({
            "Model": m,
            "Avg Abs Error": round(avg_abs_err, 3) if avg_abs_err is not None else None,
            "Samples": c,
        })
    # Rank by smallest error, then more samples
    leaderboard = [row for row in leaderboard if row["Avg Abs Error"] is not None]
    leaderboard.sort(key=lambda r: (r["Avg Abs Error"], -r["Samples"]))
    best_model = leaderboard[0]["Model"] if leaderboard else None
    return {"best_model": best_model, "leaderboard": leaderboard}



def _display_model_evaluation_tab(
    results: List[Dict[str, Any]],
    task_description: str,
    _CONFIG: Dict[str, Any]
) -> None:
    """Display model evaluation and recommendations."""
    st.markdown("#### 🏆 Model Evaluation & Recommendations")

    st.info("💡 This tab uses an LLM judge to evaluate model performance and provide recommendations.")

    # Add documentation popover for Model Evaluation
    with st.popover("📖 How Model Evaluation Works", help="Click for evaluation details"):
        st.markdown("**Model Evaluation with LLM Judge**")
        st.markdown("This tab uses an LLM judge to evaluate and rank visual LLM performance.")

        st.markdown("**Orchestration Flow:**")
        st.code("""
# 1. Prepare evaluation context
evaluation_context = {
    "task_description": task_description,
    "visual_llm_outputs": results,
    "computational_results": computational_analysis
}

# 2. LLM Judge analyzes all model outputs
judge_prompt = f'''
Task: {task_description}

Analyze the performance of these visual LLMs:
{format_model_outputs(results)}

Computational Analysis:
{computational_results}

Evaluate each model on:
1. Accuracy and relevance
2. Consistency across images
3. Confidence calibration
4. Artifact detection capability
5. Overall usefulness

Provide:
- Best model and rationale
- Model rankings with scores (0-100)
- Strengths and weaknesses for each
- Recommendations for improvement
'''

evaluation = await evaluate_visual_llm_performance(
    visual_llm_outputs=results,
    task_description=task_description,
    computational_results=comp_results,
    judge_model="gpt-5-nano"
)

# 3. Display results
- Best model selection
- Model rankings with scores
- Detailed analysis per model
- Recommendations
        """, language="python")

        st.markdown("**Key Functions:**")
        st.code("""
async def evaluate_visual_llm_performance(
    visual_llm_outputs,
    task_description,
    computational_results,
    judge_model
):
    # Build comprehensive evaluation prompt
    # Call judge LLM
    # Parse and structure results
    return {
        "best_model": str,
        "model_rankings": List[Dict],
        "analysis": str,
        "recommendations": List[str]
    }
        """, language="python")

        st.markdown("**Expected Outputs:**")
        st.markdown("- **Best Model**: Top-performing model with rationale")
        st.markdown("- **Rankings**: All models scored 0-100")
        st.markdown("- **Analysis**: Detailed strengths/weaknesses")
        st.markdown("- **Recommendations**: Actionable improvement suggestions")

        st.markdown("**Note:** Results are cached - re-run if you want fresh evaluation.")

    # Check if evaluation already run
    if st.session_state.test6_evaluation_results is not None:
        st.success("✅ Evaluation already completed. Showing cached results.")
        _display_evaluation_results(st.session_state.test6_evaluation_results)

        if st.button("🔄 Re-run Evaluation", key="rerun_evaluation"):
            st.session_state.test6_evaluation_results = None
            st.rerun()

        return

    # Run evaluation button
    if st.button("🚀 Run Model Evaluation", type="primary", key="run_evaluation"):
        with st.spinner("Evaluating models..."):
            # Get computational results if available
            comp_results = st.session_state.test6_computational_results

            # Run evaluation
            evaluation = asyncio.run(evaluate_visual_llm_performance(
                visual_llm_outputs=results,
                task_description=task_description,
                computational_results=comp_results.get("execution") if comp_results else None,
                judge_model="gpt-5-nano",
                openai_api_key=_CONFIG.get('OPENAI_API_KEY')
            ))

            st.success("✅ Evaluation complete!")

            # Cache results
            st.session_state.test6_evaluation_results = evaluation

            _display_evaluation_results(evaluation)


def _display_evaluation_results(evaluation: Dict[str, Any]) -> None:
    """Display evaluation results."""
    st.markdown("### 🏆 Best Model")

    best_model = evaluation.get("best_model", "N/A")
    st.success(f"**{best_model}** performed best for this task")

    # Model rankings
    st.markdown("### 📊 Model Rankings")

    rankings = evaluation.get("model_rankings", [])
    if rankings:
        ranking_data = []
        for rank in rankings:
            ranking_data.append({
                "Model": rank.get("model", "Unknown"),
                "Score": f"{rank.get('score', 0)}/100",
                "Rationale": rank.get("rationale", "N/A")
            })

        st.dataframe(pd.DataFrame(ranking_data), use_container_width=True)

    # Strengths
    st.markdown("### 💪 Model Strengths")

    strengths = evaluation.get("strengths", {})
    for model, strength_list in strengths.items():
        with st.expander(f"**{model}**"):
            for strength in strength_list:
                st.markdown(f"- {strength}")

    # Recommendations
    st.markdown("### 💡 Recommendations")

    recommendations = evaluation.get("recommendations", {})

    if recommendations.get("general"):
        st.info(f"**General:** {recommendations['general']}")

    if recommendations.get("task_specific"):
        st.success(f"**Task-Specific:** {recommendations['task_specific']}")

    # Enhanced prompts
    st.markdown("### ✨ Enhanced Prompts")

    enhanced_prompts = evaluation.get("enhanced_prompts", {})
    for model, prompt in enhanced_prompts.items():
        with st.expander(f"**{model}** - Enhanced Prompt"):
            st.code(prompt, language="text")


def _display_qa_tab(
    results: List[Dict[str, Any]],
    task_description: str,
    selected_models: List[str],
    _CONFIG: Dict[str, Any]
) -> None:
    """Display interactive Q&A interface with image-specific question support."""
    st.markdown("#### 💬 Interactive Q&A")

    # Ensure Q&A history exists in session state
    if 'test6_qa_history' not in st.session_state:
        st.session_state.test6_qa_history = []

    st.info("💡 Ask questions about the analysis results **OR** specific images. The AI will answer based on all available data and show relevant images.")

    # Add documentation popover for Interactive Q&A
    with st.popover("📖 How Interactive Q&A Works", help="Click for Q&A details"):
        st.markdown("**Interactive Q&A with Automatic Image Context**")
        st.markdown("Ask questions about analysis results and get answers with relevant images automatically displayed.")

        st.markdown("**Orchestration Flow:**")
        st.code("""
# 1. User asks a question
question = "Show me the analysis for urban_scene_003.jpg"

# 2. Build context from all available data
context = {
    "visual_llm_outputs": results,
    "computational_results": computational_analysis,
    "evaluation_results": model_evaluation,
    "conversation_history": previous_qa_exchanges
}

# 3. LLM selects relevant images (two methods)

# Method A: Explicit image name detection
if "urban_scene_003.jpg" in question:
    relevant_images.append(find_image("urban_scene_003.jpg"))

# Method B: LLM-based selection
image_descriptors = [
    {"image_name": img.name, "descriptor": img.summary}
    for img in results
]

selected_names = await llm_select_relevant_images(
    question=question,
    image_descriptors=image_descriptors,
    qa_model="gpt-5-nano"
)

# 4. Answer question with full context
answer = await answer_followup_question(
    question=question,
    visual_llm_outputs=results,
    computational_results=comp_results,
    evaluation_results=eval_results,
    conversation_history=history,
    qa_model="gpt-5-nano"
)

# 5. Display answer with images
# UI automatically shows:
# - Answer text
# - Relevant images in expanders
# - Model analyses for each image
# - Suggested follow-up actions
        """, language="python")

        st.markdown("**Key Functions:**")
        st.code("""
# Main Q&A function
async def answer_followup_question(
    question, visual_llm_outputs,
    computational_results, evaluation_results,
    conversation_history, qa_model
):
    # Build comprehensive context
    # Select relevant images
    # Generate answer
    # Suggest follow-up actions
    return {
        "answer": str,
        "relevant_images": List[Dict],
        "suggested_actions": List[str]
    }

# LLM-based image selection
async def llm_select_relevant_images(
    question, image_descriptors, qa_model
):
    # LLM analyzes question and image descriptors
    # Returns list of relevant image names
    # Cached for performance
        """, language="python")

        st.markdown("**Question Types:**")
        st.markdown("- **General**: *Which model performed best?*")
        st.markdown("- **Image-specific**: *Show me image_003.jpg*")
        st.markdown("- **Comparison**: *Why did models disagree on this image?*")
        st.markdown("- **Analysis**: *What patterns did you find in the data?*")

        st.markdown("**Features:**")
        st.markdown("- **Automatic Image Display**: Referenced images shown in expanders")
        st.markdown("- **Conversation History**: Maintains context across questions")
        st.markdown("- **Smart Caching**: Image selections cached for performance")
        st.markdown("- **Suggested Actions**: Follow-up questions and actions")

    # Display conversation history using chat UI
    for exchange in st.session_state.test6_qa_history:
        # User message
        with st.chat_message("user"):
            st.markdown(exchange.get("question", ""))

        # Assistant message
        with st.chat_message("assistant"):
            st.markdown(exchange.get("answer", ""))

            # Display relevant images if any
            if exchange.get('relevant_images'):
                st.markdown("**📸 Relevant Images:**")
                for img_data in exchange['relevant_images']:
                    with st.expander(f"🖼️ {img_data['image_name']}", expanded=False):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            try:
                                from PIL import Image
                                img = Image.open(img_data['image_path'])
                                st.image(img, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error loading image: {str(e)}")
                        with col2:
                            st.markdown("**Model Analyses:**")
                            for model_name, analysis in img_data.get('model_results', {}).items():
                                if hasattr(analysis, 'rationale'):
                                    st.markdown(f"**{model_name}:**")
                                    st.caption(f"{analysis.rationale[:200]}...")
                                    if hasattr(analysis, 'confidence'):
                                        st.caption(f"Confidence: {analysis.confidence:.0%}")
                                elif isinstance(analysis, dict):
                                    st.markdown(f"**{model_name}:**")
                                    st.caption(f"{analysis.get('rationale', 'N/A')[:200]}...")

            if exchange.get('suggested_actions'):
                st.caption("**Suggested Actions:**")
                for action in exchange['suggested_actions']:
                    st.caption(f"- {action}")

    # Chat input
    st.markdown("### ❓ Ask a Question")

    st.caption("💡 **Examples:**")
    st.caption("- General: *Which model performed best overall?*")
    st.caption("- Image-specific: *Show me the analysis for urban_scene_003.jpg*")
    st.caption("- Comparison: *Why did GPT-5 and Gemini disagree on image_005?*")

    # Clear history button
    c1, _ = st.columns([1, 6])
    with c1:
        if st.button("🗑️ Clear History"):
            st.session_state.test6_qa_history = []
            st.rerun()

    # Read chat input
    user_prompt = st.chat_input("Ask about the results or reference an image by name (e.g., 'urban_scene_003.jpg')")

    if user_prompt:
        # Render the user message immediately
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.spinner("Thinking..."):
            comp_results = st.session_state.test6_computational_results
            eval_results = st.session_state.test6_evaluation_results

            response = asyncio.run(answer_followup_question(
                question=user_prompt,
                visual_llm_outputs=results,
                computational_results=comp_results.get("execution") if comp_results else None,
                evaluation_results=eval_results,
                conversation_history=st.session_state.test6_qa_history,
                qa_model="gpt-5-nano",
                openai_api_key=_CONFIG.get('OPENAI_API_KEY')
            ))

        # Store the exchange
        st.session_state.test6_qa_history.append({
            "question": user_prompt,
            "answer": response.get("answer", "No answer"),
            "relevant_images": response.get("relevant_images", []),
            "suggested_actions": response.get("suggested_actions", []),
            "timestamp": response.get("timestamp")
        })

        st.rerun()


def _display_export_tab(
    results: List[Dict[str, Any]],
    selected_models: List[str],
    preset_name: str
) -> None:
    """Display export options."""
    st.markdown("#### 💾 Export Results")

    # Prepare comprehensive export data
    export_data = {
        "preset_name": preset_name,
        "timestamp": datetime.now().isoformat(),
        "models": selected_models,
        "num_images": len(results),
        "results": []
    }

    # Convert Pydantic models to dicts
    for result in results:
        export_result = {
            "image_path": result['image_path'],
            "image_name": result['image_name'],
            "model_results": {}
        }

        for model_name, analysis in result.get("model_results", {}).items():
            if hasattr(analysis, 'model_dump'):
                export_result["model_results"][model_name] = analysis.model_dump()
            elif hasattr(analysis, 'dict'):
                export_result["model_results"][model_name] = analysis.dict()
            else:
                export_result["model_results"][model_name] = str(analysis)

        export_data["results"].append(export_result)

    # Add computational results if available
    if st.session_state.test6_computational_results:
        export_data["computational_analysis"] = st.session_state.test6_computational_results

    # Add evaluation results if available
    if st.session_state.test6_evaluation_results:
        export_data["model_evaluation"] = st.session_state.test6_evaluation_results

    # Add Q&A history if available
    if st.session_state.test6_qa_history:
        export_data["qa_history"] = st.session_state.test6_qa_history

    # JSON export
    st.markdown("### 📄 JSON Export")
    json_str = json.dumps(export_data, indent=2)
    st.download_button(
        label="📥 Download Complete Analysis (JSON)",
        data=json_str,
        file_name=f"test6_analysis_{preset_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

    # CSV export
    st.markdown("### 📊 CSV Export")
    csv_data = []
    for result in results:
        for model_name, analysis in result.get("model_results", {}).items():
            response_text = ""
            confidence = 0.0

            if hasattr(analysis, 'rationale'):
                response_text = analysis.rationale
            elif hasattr(analysis, 'raw_response'):
                response_text = analysis.raw_response

            if hasattr(analysis, 'confidence'):
                confidence = analysis.confidence

            csv_data.append({
                "Image": result['image_name'],
                "Model": model_name,
                "Response": response_text,
                "Confidence": confidence
            })

    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_str = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results Table (CSV)",
            data=csv_str,
            file_name=f"test6_results_{preset_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # Summary report
    st.markdown("### 📋 Summary Report")
    st.info("PDF report generation coming soon!")

