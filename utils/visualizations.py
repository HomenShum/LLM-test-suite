"""
Visualization functions for Streamlit dashboard.
Extracted from streamlit_test_v5.py to reduce main file size.
All Plotly chart rendering functions are centralized here.
"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional
from sklearn.metrics import classification_report
import re

# Plotly configuration
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True
}

# Import label normalization from config
from config.scenarios import TEST_FLOWS


def _normalize_label(s):
    """Normalize label for comparison (imported from main file)."""
    from config.scenarios import CANON_MAP
    if not s or not isinstance(s, str):
        return ""
    s_clean = s.strip().lower()
    return CANON_MAP.get(s_clean, s_clean)


def render_test_flow_diagram(test_number: int, test_name: str):
    """Renders an emoji-based visual flow diagram for each test."""
    flow = TEST_FLOWS.get(test_number, "📊 Data → 🤖 Process → 📈 Evaluate")
    st.markdown(f"### {test_name}")
    st.info(f"**Workflow:** {flow}")


def render_kpi_metrics(df: pd.DataFrame, test_type: str = "classification", model_cols: List[str] = None):
    """
    Renders KPI metrics row for test results.

    Args:
        df: Results dataframe
        test_type: Type of test ("classification", "pruning", "agent")
        model_cols: List of model column prefixes for classification tests
    """
    if test_type == "classification" and len(df) > 0:
        cols = st.columns(4)

        with cols[0]:
            st.metric("Total Samples", len(df))

        # Calculate overall accuracy if ground truth exists
        if "classification" in df.columns and model_cols:
            y_true = df["classification"].fillna("").map(_normalize_label).tolist()

            # Find best model accuracy
            best_acc = 0.0
            best_model = "N/A"

            for model_prefix in model_cols:
                result_col = f"classification_result_{model_prefix}"
                if result_col in df.columns:
                    y_pred = df[result_col].fillna("").map(_normalize_label).tolist()
                    valid_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t and p]
                    if valid_indices:
                        y_true_f = [y_true[i] for i in valid_indices]
                        y_pred_f = [y_pred[i] for i in valid_indices]
                        acc = sum(1 for t, p in zip(y_true_f, y_pred_f) if t == p) / len(y_true_f)
                        if acc > best_acc:
                            best_acc = acc
                            best_model = model_prefix

            with cols[1]:
                delta = f"+{(best_acc - 0.5) * 100:.1f}%" if best_acc > 0.5 else None
                st.metric("Best Accuracy", f"{best_acc:.2%}", delta=delta)

        # Calculate average latency
        latency_cols = [c for c in df.columns if c.startswith("latency_") and df[c].notna().any()]
        if latency_cols:
            avg_latency = df[latency_cols].mean().mean()
            with cols[2]:
                st.metric("Avg Latency", f"{avg_latency:.2f}s")

        # Calculate test cost from cost tracker
        ct = st.session_state.cost_tracker
        with cols[3]:
            st.metric("Test Cost", f"${ct.totals['total_cost_usd']:.4f}")

    elif test_type == "pruning" and len(df) > 0:
        total_cases = len(df)
        cols = st.columns(5)
        with cols[0]:
            st.metric("Total Cases", total_cases)

        pruned_accuracy = None
        if "Pruned Correct Bool" in df.columns:
            pruned_accuracy = df["Pruned Correct Bool"].mean()
        elif "Action Correct" in df.columns:
            pruned_accuracy = df["Action Correct"].astype(str).str.contains("✅").mean()

        baseline_accuracy = None
        if "Baseline Correct Bool" in df.columns:
            baseline_accuracy = df["Baseline Correct Bool"].mean()

        with cols[1]:
            if pruned_accuracy is not None:
                st.metric("Pruned Accuracy", f"{pruned_accuracy:.1%}")
            else:
                st.metric("Pruned Accuracy", "n/a")

        with cols[2]:
            if baseline_accuracy is not None:
                delta = None
                if pruned_accuracy is not None:
                    delta = f"{(pruned_accuracy - baseline_accuracy) * 100:.1f} pts"
                st.metric("Baseline Accuracy", f"{baseline_accuracy:.1%}", delta=delta)
            else:
                st.metric("Baseline Accuracy", "n/a")

        with cols[3]:
            if "Key Score (Jaccard)" in df.columns:
                avg_score = df["Key Score (Jaccard)"].mean()
                st.metric("Avg Key Similarity", f"{avg_score:.3f}")
            else:
                st.metric("Avg Key Similarity", "n/a")

        with cols[4]:
            if "Action Shift Bool" in df.columns:
                st.metric("Action Shift Rate", f"{df['Action Shift Bool'].mean():.1%}")
            else:
                st.metric("Action Shift Rate", "n/a")


def render_cost_dashboard():
    """Renders interactive cost tracking visualizations in sidebar."""
    ct = st.session_state.cost_tracker

    if not ct.by_call:
        return  # No cost data to display

    st.subheader("💰 Cost Analytics")

    # Cost distribution by provider (Pie Chart)
    provider_costs = {}
    for call in ct.by_call:
        provider = call.get("provider", "Unknown")
        cost = call.get("total_cost_usd", 0.0)
        provider_costs[provider] = provider_costs.get(provider, 0.0) + cost

    if provider_costs:
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(provider_costs.keys()),
            values=list(provider_costs.values()),
            hole=0.3,
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Cost: $%{value:.4f}<br>%{percent}<extra></extra>'
        )])
        fig_pie.update_layout(
            title="Cost Distribution by Provider",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CONFIG)

    # Cost per call by provider
    st.markdown("**Cost per Call by Provider**")

    if ct.by_call:
        from collections import defaultdict
        provider_call_costs = defaultdict(list)
        for call in ct.by_call:
            provider = call.get('provider', 'Unknown')
            cost = call.get('total_cost_usd', 0)
            provider_call_costs[provider].append(cost)

        avg_costs = {p: sum(costs)/len(costs) for p, costs in provider_call_costs.items()}

        fig_bar = go.Figure(data=[go.Bar(
            x=list(avg_costs.keys()),
            y=list(avg_costs.values()),
            text=[f'${v:.4f}' for v in avg_costs.values()],
            textposition='auto',
            marker_color='#17becf',
            hovertemplate='<b>%{x}</b><br>Avg: $%{y:.4f}/call<extra></extra>'
        )])

        fig_bar.update_layout(
            title="Average Cost per API Call",
            xaxis_title="Provider",
            yaxis_title="Cost (USD)",
            height=250
        )

        st.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CONFIG)

    # Cumulative cost timeline
    if len(ct.by_call) > 1:
        cumulative_costs = []
        cumulative = 0.0
        call_numbers = []

        for i, call in enumerate(ct.by_call, 1):
            cumulative += call.get("total_cost_usd", 0.0)
            cumulative_costs.append(cumulative)
            call_numbers.append(i)

        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=call_numbers,
            y=cumulative_costs,
            mode='lines+markers',
            name='Cumulative Cost',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6),
            hovertemplate='Call #%{x}<br>Total: $%{y:.4f}<extra></extra>'
        ))

        fig_timeline.update_layout(
            title="Cumulative Cost Timeline",
            xaxis_title="API Call Number",
            yaxis_title="Cumulative Cost (USD)",
            height=250,
            margin=dict(l=20, r=20, t=40, b=40),
            hovermode='x unified'
        )
        st.plotly_chart(fig_timeline, use_container_width=True, config=PLOTLY_CONFIG)


def visualize_dataset_composition(df: pd.DataFrame, dataset_type: str = "classification"):
    """
    Visualizes dataset composition with interactive charts.

    Args:
        df: Dataset dataframe
        dataset_type: Type of dataset ("classification", "tool/agent sequence", "context pruning")
    """
    if df is None or df.empty:
        st.info("No data available for visualization.")
        return

    # Normalize dataset type
    kind = (dataset_type or "").strip().lower().replace(" ", "_")

    st.subheader("📊 Dataset Composition")

    # Shared: show a length histogram for the primary text column
    text_col = "query" if "query" in df.columns else ("new_question" if "new_question" in df.columns else None)
    if text_col:
        df_len = df.copy()
        try:
            df_len[text_col] = df_len[text_col].astype(str)
        except Exception:
            pass
        df_len["_text_length"] = df_len[text_col].fillna("").astype(str).str.len()
        fig_hist = go.Figure(data=[go.Histogram(
            x=df_len["_text_length"],
            nbinsx=30,
            marker_color='#2ca02c',
            hovertemplate='Length: %{x}<br>Count: %{y}<extra></extra>'
        )])
        fig_hist.update_layout(
            title=f"{text_col.replace('_',' ').title()} Length Distribution",
            xaxis_title="Length (characters)",
            yaxis_title="Count",
            height=300,
            margin=dict(l=20, r=20, t=40, b=40)
        )
        st.plotly_chart(fig_hist, use_container_width=True, config=PLOTLY_CONFIG)

    # Classification-specific visuals
    if kind in ("classification",) and "classification" in df.columns:
        col1, col2 = st.columns([2, 1])

        with col1:
            class_counts = df["classification"].value_counts()
            fig_dist = go.Figure(data=[go.Bar(
                x=class_counts.index,
                y=class_counts.values,
                text=class_counts.values,
                textposition='auto',
                marker_color='#1f77b4',
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            )])
            fig_dist.update_layout(
                title="Class Distribution",
                xaxis_title="Class Label",
                yaxis_title="Count",
                height=300,
                margin=dict(l=20, r=20, t=40, b=40)
            )
            st.plotly_chart(fig_dist, use_container_width=True, config=PLOTLY_CONFIG)

        with col2:
            # Class Balance Health Gauge
            st.markdown("**Class Balance Health**")

            class_counts = df["classification"].value_counts()
            max_count = class_counts.max()
            min_count = class_counts.min()
            balance_ratio = min_count / max_count if max_count > 0 else 0

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=balance_ratio * 100,
                title={'text': "Balance Score (%)"},
                delta={'reference': 80},  # 80% is good balance threshold
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))

            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True, config=PLOTLY_CONFIG)

    # Tool/Agent Sequence visuals
    if kind in ("tool_agent_sequence", "tool/agent_sequence", "tool_agent", "tool_sequence"):
        # Sequence length distribution (best-effort parsing)
        seq_col = "expected_sequence" if "expected_sequence" in df.columns else None
        if seq_col:
            def _seq_len(val):
                if isinstance(val, list):
                    return len(val)
                if isinstance(val, str):
                    parts = [x.strip() for x in re.split(r"[|;,]+", val) if x.strip()]
                    return len(parts)
                return 0
            try:
                df_seq = df.copy()
                df_seq["_seq_len"] = df_seq[seq_col].apply(_seq_len)
                fig_len = go.Figure(data=[go.Histogram(x=df_seq["_seq_len"], nbinsx=10, marker_color='#9467bd')])
                fig_len.update_layout(title="Sequence Length Distribution", xaxis_title="Length (#steps)", yaxis_title="Count", height=300, margin=dict(l=20, r=20, t=40, b=40))
                st.plotly_chart(fig_len, use_container_width=True, config=PLOTLY_CONFIG)
            except Exception:
                pass

    # Context Pruning visuals
    if kind in ("context_pruning", "pruning", "context"):
        # Action distribution
        if "expected_action" in df.columns:
            action_counts = df["expected_action"].astype(str).value_counts()
            fig_actions = go.Figure(data=[go.Bar(
                x=action_counts.index,
                y=action_counts.values,
                text=action_counts.values,
                textposition='auto',
                marker_color='#ff7f0e',
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            )])
            fig_actions.update_layout(title="Expected Action Distribution", xaxis_title="Action", yaxis_title="Count", height=300, margin=dict(l=20, r=20, t=40, b=40))
            st.plotly_chart(fig_actions, use_container_width=True, config=PLOTLY_CONFIG)

        # Kept keys frequency (top 10)
        if "expected_kept_keys" in df.columns and df["expected_kept_keys"].notna().any():
            try:
                keys = []
                for s in df["expected_kept_keys"].fillna("").astype(str).tolist():
                    parts = [x.strip() for x in s.split(",") if x.strip()]
                    keys.extend(parts)
                if keys:
                    counts = pd.Series(keys).value_counts().head(10)
                    fig_keys = go.Figure(data=[go.Bar(x=counts.index, y=counts.values, marker_color='#17becf')])
                    fig_keys.update_layout(title="Top Expected Kept Keys (Top 10)", xaxis_title="Key", yaxis_title="Count", height=300, margin=dict(l=20, r=20, t=40, b=40))
                    st.plotly_chart(fig_keys, use_container_width=True, config=PLOTLY_CONFIG)
            except Exception:
                pass


def render_model_comparison_chart(df: pd.DataFrame, model_cols: List[str], model_names: List[str] = None):
    """
    Renders an interactive model comparison chart with F1 scores (bars) and latency (line on secondary axis).

    Args:
        df: Results dataframe
        model_cols: List of model column prefixes (e.g., ["openrouter_mistral", "openai", "third"])
        model_names: Optional list of display names for models
    """
    if df.empty or "classification" not in df.columns:
        return

    if model_names is None:
        model_names = model_cols

    # Calculate F1 scores and latencies for each model
    y_true = df["classification"].fillna("").map(_normalize_label).tolist()

    f1_scores = []
    latencies = []
    valid_models = []
    valid_names = []

    for i, model_prefix in enumerate(model_cols):
        result_col = f"classification_result_{model_prefix}"
        latency_col = f"latency_{model_prefix}"

        if result_col in df.columns:
            y_pred = df[result_col].fillna("").map(_normalize_label).tolist()
            valid_indices = [idx for idx, (t, p) in enumerate(zip(y_true, y_pred)) if t and p]

            if valid_indices:
                y_true_f = [y_true[idx] for idx in valid_indices]
                y_pred_f = [y_pred[idx] for idx in valid_indices]

                # Calculate F1 score
                report = classification_report(y_true_f, y_pred_f, output_dict=True, zero_division=0)
                f1 = report.get('macro avg', {}).get('f1-score', 0.0)
                f1_scores.append(f1)

                # Calculate average latency
                if latency_col in df.columns and df[latency_col].notna().any():
                    avg_latency = df[latency_col].mean()
                    latencies.append(avg_latency)
                else:
                    latencies.append(0.0)

                valid_models.append(model_prefix)
                valid_names.append(model_names[i] if i < len(model_names) else model_prefix)

    if not valid_models:
        return

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add F1 score bars
    fig.add_trace(
        go.Bar(
            name="F1 Score",
            x=valid_names,
            y=f1_scores,
            text=[f"{f1:.3f}" for f1 in f1_scores],
            textposition='auto',
            marker_color='#1f77b4',
            hovertemplate='<b>%{x}</b><br>F1 Score: %{y:.4f}<extra></extra>'
        ),
        secondary_y=False
    )

    # Add latency line
    fig.add_trace(
        go.Scatter(
            name="Avg Latency",
            x=valid_names,
            y=latencies,
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=8),
            hovertemplate='<b>%{x}</b><br>Latency: %{y:.2f}s<extra></extra>'
        ),
        secondary_y=True
    )

    # Update axes
    fig.update_xaxes(title_text="Model")
    fig.update_yaxes(title_text="F1 Score", secondary_y=False, range=[0, 1])
    fig.update_yaxes(title_text="Latency (seconds)", secondary_y=True)

    fig.update_layout(
        title="Model Performance Comparison",
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

