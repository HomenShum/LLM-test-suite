"""
Advanced visualization functions for test results and execution tracking.
Extracted from streamlit_test_v5.py to reduce main file size.

This module contains:
- render_model_comparison_chart - Model performance comparison
- render_organized_results - Organized result tabs
- render_progress_replay - Animated progress replay
- render_universal_gantt_chart - Universal Gantt chart for all tests
- render_task_cards - Task card visualization
- render_single_task_card - Single task card rendering
- render_live_agent_status - Live agent status display
- render_agent_task_cards - Agent task card display
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional
from sklearn.metrics import classification_report, confusion_matrix
import time

from utils.execution_tracker import ExecutionTracker
from utils.visualizations import render_kpi_metrics
from utils.data_helpers import _normalize_label

# PLOTLY CONFIG
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True
}


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


def render_organized_results(df: pd.DataFrame, test_type: str = "classification", model_cols: List[str] = None, model_names: List[str] = None):
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
            # KPI metrics
            render_kpi_metrics(df, test_type="classification", model_cols=model_cols)

            st.divider()

            # Model comparison chart
            if model_cols:
                render_model_comparison_chart(df, model_cols, model_names)

        elif test_type == "pruning":
            # Pruning-specific metrics
            if "action_match" in df.columns:
                action_accuracy = df["action_match"].mean()
                st.metric("Action Accuracy", f"{action_accuracy:.2%}")

            if "key_similarity" in df.columns:
                avg_similarity = df["key_similarity"].mean()
                st.metric("Avg Key Similarity", f"{avg_similarity:.3f}")

        elif test_type == "agent":
            # Agent-specific metrics
            if "accuracy" in df.columns:
                avg_accuracy = df["accuracy"].mean()
                st.metric("Average Accuracy", f"{avg_accuracy:.2%}")

    # Tab 2: Performance
    with result_tabs[1]:
        st.subheader("Performance Metrics")

        if test_type == "classification" and model_cols:
            # Per-model performance
            for i, model_prefix in enumerate(model_cols):
                result_col = f"classification_result_{model_prefix}"
                if result_col in df.columns:
                    model_name = model_names[i] if model_names and i < len(model_names) else model_prefix

                    with st.expander(f"📊 {model_name} Performance", expanded=False):
                        y_true = df["classification"].fillna("").map(_normalize_label).tolist()
                        y_pred = df[result_col].fillna("").map(_normalize_label).tolist()

                        valid_indices = [idx for idx, (t, p) in enumerate(zip(y_true, y_pred)) if t and p]

                        if valid_indices:
                            y_true_f = [y_true[idx] for idx in valid_indices]
                            y_pred_f = [y_pred[idx] for idx in valid_indices]

                            # Classification report
                            report = classification_report(y_true_f, y_pred_f, output_dict=True, zero_division=0)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df, use_container_width=True)

                            # Confusion matrix
                            cm = confusion_matrix(y_true_f, y_pred_f)
                            labels = sorted(set(y_true_f + y_pred_f))

                            fig = go.Figure(data=go.Heatmap(
                                z=cm,
                                x=labels,
                                y=labels,
                                colorscale='Blues',
                                text=cm,
                                texttemplate='%{text}',
                                textfont={"size": 12}
                            ))

                            fig.update_layout(
                                title=f"Confusion Matrix - {model_name}",
                                xaxis_title="Predicted",
                                yaxis_title="Actual",
                                height=400
                            )

                            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # Tab 3: Errors
    with result_tabs[2]:
        st.subheader("Error Analysis")

        if test_type == "classification" and model_cols:
            # Show misclassifications
            for i, model_prefix in enumerate(model_cols):
                result_col = f"classification_result_{model_prefix}"
                if result_col in df.columns:
                    model_name = model_names[i] if model_names and i < len(model_names) else model_prefix

                    # Find errors
                    df_copy = df.copy()
                    df_copy['true_norm'] = df_copy["classification"].fillna("").map(_normalize_label)
                    df_copy['pred_norm'] = df_copy[result_col].fillna("").map(_normalize_label)
                    errors = df_copy[df_copy['true_norm'] != df_copy['pred_norm']]

                    if not errors.empty:
                        with st.expander(f"❌ {model_name} Errors ({len(errors)} total)", expanded=False):
                            # Show error samples
                            display_cols = ["query", "classification", result_col]
                            if f"{result_col}_rationale" in errors.columns:
                                display_cols.append(f"{result_col}_rationale")

                            st.dataframe(errors[display_cols].head(10), use_container_width=True)

    # Tab 4: Raw Data
    with result_tabs[3]:
        st.subheader("Raw Data")
        st.dataframe(df, use_container_width=True, height=400)

        # Download button
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results as CSV",
            data=csv_bytes,
            file_name=f"{test_type}_results.csv",
            mime="text/csv"
        )


def render_progress_replay(test_name: str = "classification"):
    """
    Renders an animated replay of the processing progression after async operations complete.
    Uses progress metadata collected during execution.
    """
    if 'last_progress_metadata' not in st.session_state:
        st.info("No execution data yet. Run a test to see the replay.")
        return

    metadata = st.session_state.last_progress_metadata
    if not metadata:
        st.info("No progress data available.")
        return

    st.subheader("🎬 Progress Replay")

    # Create timeline
    events = []
    # Handle both dict and list formats
    if isinstance(metadata, dict):
        metadata_list = metadata.get(test_name, [])
    else:
        metadata_list = metadata if isinstance(metadata, list) else []

    for item in metadata_list:
        # Skip if item is not a dict
        if not isinstance(item, dict):
            continue
        events.append({
            'time': item.get('timestamp', 0),
            'batch': item.get('batch_id', 'unknown'),
            'status': item.get('status', 'unknown'),
            'message': item.get('message', '')
        })

    if not events:
        st.info("No events to replay.")
        return

    # Sort by time
    events.sort(key=lambda x: x['time'])

    # Animation controls
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        speed = st.slider("Replay Speed", 0.1, 2.0, 1.0, 0.1)
    with col2:
        auto_play = st.checkbox("Auto Play", value=False)
    with col3:
        current_idx = st.number_input("Event Index", 0, len(events) - 1, 0)

    # Display current event
    if 0 <= current_idx < len(events):
        event = events[current_idx]
        st.info(f"**Batch {event['batch']}** - {event['status']}: {event['message']}")

        # Progress bar
        progress = (current_idx + 1) / len(events)
        st.progress(progress)

    # Auto-play logic
    if auto_play and current_idx < len(events) - 1:
        time.sleep(1.0 / speed)
        st.rerun()


def render_universal_gantt_chart(test_name: str = "Test 1"):
    """
    Universal Gantt chart renderer for ALL tests using ExecutionTracker events.

    Args:
        test_name: Name of the test to render timeline for
    """
    if 'execution_tracker' not in st.session_state:
        st.info("No execution data yet. Run a test to see the timeline.")
        return

    tracker: ExecutionTracker = st.session_state.execution_tracker
    events = tracker.get_test_events(test_name)

    if not events:
        st.info(f"No execution events for {test_name}. Run the test to see the timeline.")
        return

    st.subheader(f"📊 Execution Timeline - {test_name}")

    # Group events by task
    tasks = {}
    for event in events:
        task_id = event.task_id
        if task_id not in tasks:
            tasks[task_id] = {
                'task_id': task_id,
                'task_name': event.task_name,
                'start_time': event.timestamp,
                'end_time': event.timestamp,
                'status': event.status,
                'events': []
            }
        tasks[task_id]['events'].append(event)
        tasks[task_id]['end_time'] = max(tasks[task_id]['end_time'], event.timestamp)

    # Create Gantt chart
    fig = go.Figure()

    # Color mapping
    status_colors = {
        'started': '#3498db',
        'running': '#2ecc71',
        'completed': '#27ae60',
        'failed': '#e74c3c',
        'pending': '#95a5a6'
    }

    for i, (task_id, task_data) in enumerate(tasks.items()):
        start = task_data['start_time']
        end = task_data['end_time']
        duration = end - start

        color = status_colors.get(task_data['status'], '#95a5a6')

        fig.add_trace(go.Bar(
            name=task_data['task_name'],
            x=[duration],
            y=[task_data['task_name']],
            orientation='h',
            marker=dict(color=color),
            hovertemplate=f"<b>{task_data['task_name']}</b><br>" +
                         f"Duration: {duration:.2f}s<br>" +
                         f"Status: {task_data['status']}<extra></extra>",
            showlegend=False
        ))

    fig.update_layout(
        title=f"Execution Timeline - {test_name}",
        xaxis_title="Duration (seconds)",
        yaxis_title="Task",
        height=max(400, len(tasks) * 40),
        barmode='overlay',
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def render_task_cards(test_name: str, tracker: ExecutionTracker):
    """Render task cards (similar to HTML prototype) with hover details."""

    st.subheader("Task Cards View")

    events = tracker.get_test_events(test_name)
    if not events:
        st.info("No tasks to display.")
        return

    # Group by task (using agent_id as task identifier)
    tasks = {}
    for event in events:
        # ExecutionEvent uses agent_id, not task_id
        task_id = getattr(event, 'agent_id', 'unknown')
        task_name = getattr(event, 'agent_name', task_id)
        status = getattr(event, 'status', 'unknown')
        timestamp = getattr(event, 'timestamp', 0)
        metadata = getattr(event, 'metadata', {})

        if task_id not in tasks:
            tasks[task_id] = {
                'task_id': task_id,
                'task_name': task_name,
                'status': status,
                'start_time': timestamp,
                'end_time': timestamp,
                'metadata': metadata
            }
        else:
            tasks[task_id]['end_time'] = max(tasks[task_id]['end_time'], timestamp)
            tasks[task_id]['status'] = status

    # Render cards in grid
    cols = st.columns(3)
    for i, (task_id, task) in enumerate(tasks.items()):
        with cols[i % 3]:
            render_single_task_card(task_id, task, tracker)


def render_single_task_card(task_id: str, task: Dict, tracker: ExecutionTracker):
    """Render a single task card with hover details."""

    # Status icon
    status_icons = {
        'started': '🔵',
        'running': '🟢',
        'completed': '✅',
        'failed': '❌',
        'pending': '⏳'
    }
    icon = status_icons.get(task['status'], '⚪')

    # Duration
    duration = task['end_time'] - task['start_time']

    # Card container
    with st.container():
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
            <div style="font-size: 20px;">{icon} <b>{task['task_name']}</b></div>
            <div style="color: #666; font-size: 14px;">Status: {task['status']}</div>
            <div style="color: #666; font-size: 14px;">Duration: {duration:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)

        # Expandable details
        with st.expander("View Details"):
            st.json(task['metadata'])


def render_live_agent_status(test_name: str = "classification"):
    """Renders live status cards for running agents/batches."""
    if 'active_batches' not in st.session_state:
        st.session_state.active_batches = {}

    active = st.session_state.active_batches.get(test_name, {})

    if not active:
        st.info("No active batches.")
        return

    st.subheader("🔴 Live Agent Status")

    cols = st.columns(min(3, len(active)))
    for i, (batch_id, batch_info) in enumerate(active.items()):
        with cols[i % 3]:
            status = batch_info.get('status', 'unknown')
            progress = batch_info.get('progress', 0)

            st.metric(f"Batch {batch_id}", status)
            st.progress(progress)


def render_agent_task_cards():
    """Renders task cards showing status of different test runs."""
    st.subheader("Test Run History")

    if 'test_run_history' not in st.session_state:
        st.session_state.test_run_history = []

    history = st.session_state.test_run_history

    if not history:
        st.info("No test runs yet.")
        return

    for run in history[-5:]:  # Show last 5 runs
        with st.expander(f"{run['test_name']} - {run['timestamp']}", expanded=False):
            st.write(f"**Status:** {run['status']}")
            st.write(f"**Duration:** {run['duration']:.2f}s")
            if 'results' in run:
                st.json(run['results'])

