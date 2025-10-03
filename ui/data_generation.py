"""Preparation tab rendering extracted from main app."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections import Counter
from typing import Dict, Any, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.plotly_config import PLOTLY_CONFIG
from utils.visualizations import visualize_dataset_composition
from config.scenarios import SUGGESTED_PROMPTS, SKELETON_COLUMNS
from utils.helpers import enhance_prompt_with_user_input
from utils.model_discovery import OPENAI_MODEL
from utils.data_helpers import save_dataset_to_file
from core.models import PruningDataItem

# These globals are configured at runtime via `configure`.
CLASSIFICATION_DATASET_PATH = ""
TOOL_SEQUENCE_DATASET_PATH = ""
CONTEXT_PRUNING_DATASET_PATH = ""
DEFAULT_DATASET_PROMPTS: Dict[str, Any] = {}
AVAILABLE_MODELS = []
API_ROUTING_MODE = "openrouter"
generate_synthetic_data = None
load_classification_dataset = None
load_tool_sequence_dataset = None
load_context_pruning_dataset = None

def configure(context: Dict[str, Any]) -> None:
    """Configure module-level globals used by the preparation tab."""
    global CLASSIFICATION_DATASET_PATH, TOOL_SEQUENCE_DATASET_PATH, CONTEXT_PRUNING_DATASET_PATH
    global DEFAULT_DATASET_PROMPTS, AVAILABLE_MODELS, API_ROUTING_MODE
    global generate_synthetic_data, load_classification_dataset, load_tool_sequence_dataset, load_context_pruning_dataset
    paths = context.get("dataset_paths", {})
    CLASSIFICATION_DATASET_PATH = paths.get("classification", "")
    TOOL_SEQUENCE_DATASET_PATH = paths.get("tool_sequence", "")
    CONTEXT_PRUNING_DATASET_PATH = paths.get("context_pruning", "")
    DEFAULT_DATASET_PROMPTS = context.get("dataset_prompts", {})
    AVAILABLE_MODELS = context.get("available_models", [])
    API_ROUTING_MODE = context.get("api_routing_mode", "openrouter")
    generate_synthetic_data = context.get("generate_synthetic_data")
    loaders = context.get("loaders", {})
    load_classification_dataset = loaders.get("classification")
    load_tool_sequence_dataset = loaders.get("tool_sequence")
    load_context_pruning_dataset = loaders.get("context_pruning")

def render_preparation_tab(tab) -> None:
    """Render the Preparation (dataset generation) tab."""
    if generate_synthetic_data is None:
        st.error("Data generation is not configured. Call configure() first.")
        return
    with tab:
        st.header("Preparation: Synthetic Data Generation")

        # --- NEW: Display dataset status ---
        st.subheader("📊 Dataset Status")
        col1, col2, col3 = st.columns(3)

        with col1:
            if os.path.exists(CLASSIFICATION_DATASET_PATH):
                df_class = pd.read_csv(CLASSIFICATION_DATASET_PATH)
                st.success(f"✅ Classification\n({len(df_class)} rows)")
            else:
                st.warning("⚠️ Classification\n(Missing)")

        with col2:
            if os.path.exists(TOOL_SEQUENCE_DATASET_PATH):
                df_tool = pd.read_csv(TOOL_SEQUENCE_DATASET_PATH)
                st.success(f"✅ Tool/Agent Seq.\n({len(df_tool)} rows)")
            else:
                st.warning("⚠️ Tool/Agent Seq.\n(Missing)")

        with col3:
            if os.path.exists(CONTEXT_PRUNING_DATASET_PATH):
                df_prune = pd.read_csv(CONTEXT_PRUNING_DATASET_PATH)
                st.success(f"✅ Context Pruning\n({len(df_prune)} rows)")
            else:
                st.warning("⚠️ Context Pruning\n(Missing)")

        st.divider()

        # --- NEW: Dataset Composition Visualization ---
        st.subheader("📊 Dataset Composition Analysis")

        # Create tabs for each dataset type
        dataset_viz_tabs = st.tabs(["Classification", "Tool/Agent Sequence", "Context Pruning"])

        with dataset_viz_tabs[0]:
            if os.path.exists(CLASSIFICATION_DATASET_PATH):
                df_class = pd.read_csv(CLASSIFICATION_DATASET_PATH)
                visualize_dataset_composition(df_class, dataset_type="Classification")
            else:
                st.info("Generate the Classification dataset to see composition analysis.")

        with dataset_viz_tabs[1]:
            if os.path.exists(TOOL_SEQUENCE_DATASET_PATH):
                df_tool = pd.read_csv(TOOL_SEQUENCE_DATASET_PATH)
                visualize_dataset_composition(df_tool, dataset_type="Tool/Agent Sequence")
            else:
                st.info("Generate the Tool/Agent Sequence dataset to see composition analysis.")

        with dataset_viz_tabs[2]:
            if os.path.exists(CONTEXT_PRUNING_DATASET_PATH):
                df_prune = pd.read_csv(CONTEXT_PRUNING_DATASET_PATH)
                visualize_dataset_composition(df_prune, dataset_type="Context Pruning")
            else:
                st.info("Generate the Context Pruning dataset to see composition analysis.")

        st.divider()

        # --- NEW: Cross-Dataset Analytics Section ---
        # Only display if at least one dataset exists
        if any([os.path.exists(p) for p in [CLASSIFICATION_DATASET_PATH, TOOL_SEQUENCE_DATASET_PATH, CONTEXT_PRUNING_DATASET_PATH]]):
            st.subheader("📊 Cross-Dataset Analytics")

            analytics_tabs = st.tabs(["Dataset Comparison", "Generation History", "Quality Metrics", "Token Economics"])

            # Tab 1: Dataset Comparison Matrix
            with analytics_tabs[0]:
                st.markdown("### Dataset Comparison Matrix")

                # Collect statistics for all datasets
                dataset_stats = []

                # Classification dataset
                if os.path.exists(CLASSIFICATION_DATASET_PATH):
                    try:
                        df_class = pd.read_csv(CLASSIFICATION_DATASET_PATH)
                        unique_labels = df_class['classification'].nunique() if 'classification' in df_class.columns else 0
                        avg_query_len = df_class['query'].astype(str).str.len().mean() if 'query' in df_class.columns else 0
                        missing_pct = (df_class.isnull().sum().sum() / df_class.size * 100) if df_class.size > 0 else 0
                        file_size_kb = os.path.getsize(CLASSIFICATION_DATASET_PATH) / 1024

                        dataset_stats.append({
                            'Dataset': 'Classification',
                            'Total Items': len(df_class),
                            'Unique Labels/Actions': unique_labels,
                            'Avg Query Length (chars)': round(avg_query_len, 1),
                            'Missing Values (%)': round(missing_pct, 2),
                            'File Size (KB)': round(file_size_kb, 2)
                        })
                    except Exception as e:
                        st.warning(f"Error reading Classification dataset: {e}")
                else:
                    dataset_stats.append({
                        'Dataset': 'Classification',
                        'Total Items': 'N/A',
                        'Unique Labels/Actions': 'N/A',
                        'Avg Query Length (chars)': 'N/A',
                        'Missing Values (%)': 'N/A',
                        'File Size (KB)': 'N/A'
                    })
        
                    # Tool/Agent Sequence dataset
                    if os.path.exists(TOOL_SEQUENCE_DATASET_PATH):
                        try:
                            df_tool = pd.read_csv(TOOL_SEQUENCE_DATASET_PATH)
                            unique_seqs = df_tool['expected_sequence'].nunique() if 'expected_sequence' in df_tool.columns else 0
                            avg_query_len = df_tool['query'].astype(str).str.len().mean() if 'query' in df_tool.columns else 0
                            missing_pct = (df_tool.isnull().sum().sum() / df_tool.size * 100) if df_tool.size > 0 else 0
                            file_size_kb = os.path.getsize(TOOL_SEQUENCE_DATASET_PATH) / 1024
        
                            dataset_stats.append({
                                'Dataset': 'Tool/Agent Sequence',
                                'Total Items': len(df_tool),
                                'Unique Labels/Actions': unique_seqs,
                                'Avg Query Length (chars)': round(avg_query_len, 1),
                                'Missing Values (%)': round(missing_pct, 2),
                                'File Size (KB)': round(file_size_kb, 2)
                            })
                        except Exception as e:
                            st.warning(f"Error reading Tool/Agent Sequence dataset: {e}")
                    else:
                        dataset_stats.append({
                            'Dataset': 'Tool/Agent Sequence',
                            'Total Items': 'N/A',
                            'Unique Labels/Actions': 'N/A',
                            'Avg Query Length (chars)': 'N/A',
                            'Missing Values (%)': 'N/A',
                            'File Size (KB)': 'N/A'
                        })
        
                    # Context Pruning dataset
                    if os.path.exists(CONTEXT_PRUNING_DATASET_PATH):
                        try:
                            df_prune = pd.read_csv(CONTEXT_PRUNING_DATASET_PATH)
                            unique_actions = df_prune['expected_action'].nunique() if 'expected_action' in df_prune.columns else 0
                            avg_query_len = df_prune['new_question'].astype(str).str.len().mean() if 'new_question' in df_prune.columns else 0
                            missing_pct = (df_prune.isnull().sum().sum() / df_prune.size * 100) if df_prune.size > 0 else 0
                            file_size_kb = os.path.getsize(CONTEXT_PRUNING_DATASET_PATH) / 1024
        
                            dataset_stats.append({
                                'Dataset': 'Context Pruning',
                                'Total Items': len(df_prune),
                                'Unique Labels/Actions': unique_actions,
                                'Avg Query Length (chars)': round(avg_query_len, 1),
                                'Missing Values (%)': round(missing_pct, 2),
                                'File Size (KB)': round(file_size_kb, 2)
                            })
                        except Exception as e:
                            st.warning(f"Error reading Context Pruning dataset: {e}")
                    else:
                        dataset_stats.append({
                            'Dataset': 'Context Pruning',
                            'Total Items': 'N/A',
                            'Unique Labels/Actions': 'N/A',
                            'Avg Query Length (chars)': 'N/A',
                            'Missing Values (%)': 'N/A',
                            'File Size (KB)': 'N/A'
                        })
        
                    # Create DataFrame and display with styling
                    stats_df = pd.DataFrame(dataset_stats)
        
                    # Apply gradient styling only to numeric columns
                    def style_stats(df):
                        styled = df.style
                        numeric_cols = ['Total Items', 'Avg Query Length (chars)', 'File Size (KB)']
                        for col in numeric_cols:
                            if col in df.columns:
                                # Only apply gradient to numeric values (not 'N/A')
                                try:
                                    styled = styled.background_gradient(subset=[col], cmap='Blues', vmin=0)
                                except:
                                    pass
                        return styled
        
                    st.dataframe(style_stats(stats_df), use_container_width=True)
        
                    # Comparison Charts
                    st.markdown("### Visual Comparison")
                    comp_col1, comp_col2 = st.columns(2)
        
                    with comp_col1:
                        # Bar chart: Total items
                        valid_stats = [s for s in dataset_stats if s['Total Items'] != 'N/A']
                        if valid_stats:
                            fig_items = go.Figure(data=[go.Bar(
                                x=[s['Dataset'] for s in valid_stats],
                                y=[s['Total Items'] for s in valid_stats],
                                text=[s['Total Items'] for s in valid_stats],
                                textposition='auto',
                                marker_color='#1f77b4',
                                hovertemplate='<b>%{x}</b><br>Items: %{y}<extra></extra>'
                            )])
                            fig_items.update_layout(
                                title="Total Items Comparison",
                                xaxis_title="Dataset",
                                yaxis_title="Total Items",
                                height=300,
                                margin=dict(l=20, r=20, t=40, b=40)
                            )
                            st.plotly_chart(fig_items, use_container_width=True, config=PLOTLY_CONFIG)
                        else:
                            st.info("No datasets available for comparison")
        
                    with comp_col2:
                        # Bar chart: Average query complexity
                        valid_stats = [s for s in dataset_stats if s['Avg Query Length (chars)'] != 'N/A']
                        if valid_stats:
                            fig_complexity = go.Figure(data=[go.Bar(
                                x=[s['Dataset'] for s in valid_stats],
                                y=[s['Avg Query Length (chars)'] for s in valid_stats],
                                text=[f"{s['Avg Query Length (chars)']:.1f}" for s in valid_stats],
                                textposition='auto',
                                marker_color='#ff7f0e',
                                hovertemplate='<b>%{x}</b><br>Avg Length: %{y:.1f} chars<extra></extra>'
                            )])
                            fig_complexity.update_layout(
                                title="Average Query Complexity",
                                xaxis_title="Dataset",
                                yaxis_title="Avg Character Length",
                                height=300,
                                margin=dict(l=20, r=20, t=40, b=40)
                            )
                            st.plotly_chart(fig_complexity, use_container_width=True, config=PLOTLY_CONFIG)
                        else:
                            st.info("No datasets available for comparison")
        
                # Tab 2: Generation History & Cost Trends
                with analytics_tabs[1]:
                    st.markdown("### Generation History & Cost Trends")
        
                    # Check if generation history exists
                    gen_history = st.session_state.get("generation_history", [])
        
                    if not gen_history:
                        st.info("No generation history available. Generate datasets to see cost trends.")
                    else:
                        # Convert to DataFrame for easier manipulation
                        hist_df = pd.DataFrame(gen_history)
        
                        # Add run number
                        hist_df['run_number'] = range(1, len(hist_df) + 1)
        
                        # Calculate cumulative cost
                        hist_df['cumulative_cost'] = hist_df['cost_usd'].cumsum()
        
                        # 1. Dual-Axis Cost Trend Chart
                        st.markdown("#### Cost Trends Over Time")
        
                        fig_cost_trend = make_subplots(specs=[[{"secondary_y": True}]])
        
                        # Primary axis: Cost per generation run
                        fig_cost_trend.add_trace(
                            go.Scatter(
                                x=hist_df['run_number'],
                                y=hist_df['cost_usd'],
                                mode='lines+markers',
                                name='Cost per Run',
                                line=dict(color='#2ca02c', width=2),
                                marker=dict(size=8),
                                hovertemplate='Run #%{x}<br>Cost: $%{y:.4f}<extra></extra>'
                            ),
                            secondary_y=False
                        )
        
                        # Secondary axis: Cumulative cost
                        fig_cost_trend.add_trace(
                            go.Scatter(
                                x=hist_df['run_number'],
                                y=hist_df['cumulative_cost'],
                                mode='lines',
                                name='Cumulative Cost',
                                line=dict(color='#d62728', width=2, dash='dash'),
                                hovertemplate='Run #%{x}<br>Total: $%{y:.4f}<extra></extra>'
                            ),
                            secondary_y=True
                        )
        
                        # Update axes
                        fig_cost_trend.update_xaxes(title_text="Generation Run Number")
                        fig_cost_trend.update_yaxes(title_text="Cost per Run (USD)", secondary_y=False)
                        fig_cost_trend.update_yaxes(title_text="Cumulative Cost (USD)", secondary_y=True)
        
                        fig_cost_trend.update_layout(
                            height=400,
                            hovermode='x unified',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
        
                        st.plotly_chart(fig_cost_trend, use_container_width=True, config=PLOTLY_CONFIG)
        
                        # 2. Token Usage Trends
                        st.markdown("#### Token Usage Trends")
        
                        fig_tokens = go.Figure()
        
                        fig_tokens.add_trace(go.Scatter(
                            x=hist_df['run_number'],
                            y=hist_df['tokens'],
                            mode='lines+markers',
                            name='Tokens Used',
                            fill='tozeroy',
                            fillcolor='rgba(31, 119, 180, 0.3)',
                            line=dict(color='#1f77b4', width=2),
                            marker=dict(size=6),
                            hovertemplate='Run #%{x}<br>Tokens: %{y:,}<extra></extra>'
                        ))
        
                        fig_tokens.update_layout(
                            title="Token Usage Over Time",
                            xaxis_title="Generation Run Number",
                            yaxis_title="Total Tokens Used",
                            height=300,
                            margin=dict(l=20, r=20, t=40, b=40)
                        )
        
                        st.plotly_chart(fig_tokens, use_container_width=True, config=PLOTLY_CONFIG)
        
                        # 3. Model Efficiency Comparison (if multiple models used)
                        unique_models = hist_df['model'].nunique()
        
                        if unique_models >= 2:
                            st.markdown("#### Model Efficiency Comparison")
        
                            # Calculate efficiency metrics per model
                            model_efficiency = hist_df.groupby('model').agg({
                                'cost_usd': 'sum',
                                'tokens': 'sum',
                                'total_items': 'sum'
                            }).reset_index()
        
                            model_efficiency['cost_per_item'] = model_efficiency['cost_usd'] / model_efficiency['total_items']
                            model_efficiency['tokens_per_item'] = model_efficiency['tokens'] / model_efficiency['total_items']
        
                            eff_col1, eff_col2 = st.columns(2)
        
                            with eff_col1:
                                # Cost per item by model
                                fig_cost_eff = go.Figure(data=[go.Bar(
                                    x=model_efficiency['model'],
                                    y=model_efficiency['cost_per_item'],
                                    text=[f"${v:.4f}" for v in model_efficiency['cost_per_item']],
                                    textposition='auto',
                                    marker_color='#2ca02c',
                                    hovertemplate='<b>%{x}</b><br>Cost/Item: $%{y:.4f}<extra></extra>'
                                )])
                                fig_cost_eff.update_layout(
                                    title="Cost per Item by Model",
                                    xaxis_title="Model",
                                    yaxis_title="Cost per Item (USD)",
                                    height=300,
                                    margin=dict(l=20, r=20, t=40, b=40)
                                )
                                st.plotly_chart(fig_cost_eff, use_container_width=True, config=PLOTLY_CONFIG)
        
                            with eff_col2:
                                # Tokens per item by model
                                fig_tokens_eff = go.Figure(data=[go.Bar(
                                    x=model_efficiency['model'],
                                    y=model_efficiency['tokens_per_item'],
                                    text=[f"{v:.0f}" for v in model_efficiency['tokens_per_item']],
                                    textposition='auto',
                                    marker_color='#1f77b4',
                                    hovertemplate='<b>%{x}</b><br>Tokens/Item: %{y:.0f}<extra></extra>'
                                )])
                                fig_tokens_eff.update_layout(
                                    title="Tokens per Item by Model",
                                    xaxis_title="Model",
                                    yaxis_title="Tokens per Item",
                                    height=300,
                                    margin=dict(l=20, r=20, t=40, b=40)
                                )
                                st.plotly_chart(fig_tokens_eff, use_container_width=True, config=PLOTLY_CONFIG)
        
                # Tab 3: Quality Metrics Dashboard
                with analytics_tabs[2]:
                    st.markdown("### Quality Metrics Dashboard")
        
                    # 3-column layout for each dataset
                    qm_col1, qm_col2, qm_col3 = st.columns(3)
        
                    # --- Column 1: Classification Dataset Quality ---
                    with qm_col1:
                        st.markdown("#### 📋 Classification Dataset")
        
                        if os.path.exists(CLASSIFICATION_DATASET_PATH):
                            try:
                                df_class = pd.read_csv(CLASSIFICATION_DATASET_PATH)
        
                                if not df_class.empty and 'classification' in df_class.columns:
                                    # Calculate balance score (min/max class ratio)
                                    class_counts = df_class['classification'].value_counts()
                                    if len(class_counts) > 1:
                                        balance_score = (class_counts.min() / class_counts.max() * 100) if class_counts.max() > 0 else 0
                                    else:
                                        balance_score = 100  # Perfect balance if only one class
        
                                    # Gauge chart for balance
                                    fig_gauge_class = go.Figure(go.Indicator(
                                        mode="gauge+number",
                                        value=balance_score,
                                        title={'text': "Balance Score"},
                                        gauge={
                                            'axis': {'range': [0, 100]},
                                            'bar': {'color': "darkblue"},
                                            'steps': [
                                                {'range': [0, 50], 'color': "lightgray"},
                                                {'range': [50, 80], 'color': "lightyellow"},
                                                {'range': [80, 100], 'color': "lightgreen"}
                                            ],
                                            'threshold': {
                                                'line': {'color': "red", 'width': 4},
                                                'thickness': 0.75,
                                                'value': 90
                                            }
                                        }
                                    ))
                                    fig_gauge_class.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                                    st.plotly_chart(fig_gauge_class, use_container_width=True, config=PLOTLY_CONFIG)
        
                                    # Metrics below gauge
                                    metric_col1, metric_col2 = st.columns(2)
                                    with metric_col1:
                                        diversity = len(class_counts)
                                        st.metric("Diversity", f"{diversity} classes")
                                    with metric_col2:
                                        completeness = (1 - df_class.isnull().sum().sum() / df_class.size) * 100 if df_class.size > 0 else 0
                                        st.metric("Completeness", f"{completeness:.1f}%")
        
                                    # Keyword analysis for classification
                                    st.markdown("**Top Keywords by Label**")
        
                                    # Simple stop words list
                                    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                                                'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                                                'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                                                'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
                                                'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        
                                    if 'query' in df_class.columns:
                                        # Get top label (most common)
                                        top_label = class_counts.index[0]
                                        label_queries = df_class[df_class['classification'] == top_label]['query'].astype(str)
        
                                        # Extract words
                                        all_words = []
                                        for query in label_queries:
                                            words = re.findall(r'\b[a-z]{3,}\b', query.lower())
                                            all_words.extend([w for w in words if w not in stop_words])
        
                                        if all_words:
                                            word_counts = Counter(all_words).most_common(5)
        
                                            # Horizontal bar chart
                                            fig_keywords = go.Figure(data=[go.Bar(
                                                y=[w[0] for w in word_counts],
                                                x=[w[1] for w in word_counts],
                                                orientation='h',
                                                marker_color='darkblue',
                                                text=[w[1] for w in word_counts],
                                                textposition='auto'
                                            )])
                                            fig_keywords.update_layout(
                                                title=f"Top 5 Keywords: {top_label}",
                                                xaxis_title="Frequency",
                                                yaxis_title="",
                                                height=200,
                                                margin=dict(l=20, r=20, t=40, b=20)
                                            )
                                            st.plotly_chart(fig_keywords, use_container_width=True, config=PLOTLY_CONFIG)
                                        else:
                                            st.info("No keywords extracted")
        
                                else:
                                    st.info("Dataset is empty or missing required columns")
                            except Exception as e:
                                st.warning(f"Error analyzing Classification dataset: {e}")
                        else:
                            st.info("Classification dataset not found")
        
                    # --- Column 2: Tool/Agent Sequence Dataset Quality ---
                    with qm_col2:
                        st.markdown("#### 🔧 Tool/Agent Sequence")
        
                        if os.path.exists(TOOL_SEQUENCE_DATASET_PATH):
                            try:
                                df_tool = pd.read_csv(TOOL_SEQUENCE_DATASET_PATH)
        
                                if not df_tool.empty and 'expected_sequence' in df_tool.columns:
                                    # Calculate average sequence length (0-10 scale)
                                    avg_seq_length = df_tool['expected_sequence'].astype(str).str.split(',').str.len().mean()
                                    # Normalize to 0-100 scale (assume max 10 steps)
                                    balance_score = min((avg_seq_length / 10) * 100, 100) if avg_seq_length > 0 else 0
        
                                    # Gauge chart
                                    fig_gauge_tool = go.Figure(go.Indicator(
                                        mode="gauge+number",
                                        value=balance_score,
                                        title={'text': "Complexity Score"},
                                        gauge={
                                            'axis': {'range': [0, 100]},
                                            'bar': {'color': "darkgreen"},
                                            'steps': [
                                                {'range': [0, 50], 'color': "lightgray"},
                                                {'range': [50, 80], 'color': "lightyellow"},
                                                {'range': [80, 100], 'color': "lightgreen"}
                                            ],
                                            'threshold': {
                                                'line': {'color': "red", 'width': 4},
                                                'thickness': 0.75,
                                                'value': 90
                                            }
                                        }
                                    ))
                                    fig_gauge_tool.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                                    st.plotly_chart(fig_gauge_tool, use_container_width=True, config=PLOTLY_CONFIG)
        
                                    # Metrics below gauge
                                    metric_col1, metric_col2 = st.columns(2)
                                    with metric_col1:
                                        diversity = df_tool['expected_sequence'].nunique()
                                        st.metric("Unique Sequences", f"{diversity}")
                                    with metric_col2:
                                        completeness = (1 - df_tool.isnull().sum().sum() / df_tool.size) * 100 if df_tool.size > 0 else 0
                                        st.metric("Completeness", f"{completeness:.1f}%")
        
                                    # Sequence length distribution
                                    st.markdown("**Sequence Length Distribution**")
                                    seq_lengths = df_tool['expected_sequence'].astype(str).str.split(',').str.len()
        
                                    fig_seq_dist = go.Figure(data=[go.Histogram(
                                        x=seq_lengths,
                                        marker_color='darkgreen',
                                        nbinsx=10
                                    )])
                                    fig_seq_dist.update_layout(
                                        title="Steps per Sequence",
                                        xaxis_title="Number of Steps",
                                        yaxis_title="Frequency",
                                        height=200,
                                        margin=dict(l=20, r=20, t=40, b=20)
                                    )
                                    st.plotly_chart(fig_seq_dist, use_container_width=True, config=PLOTLY_CONFIG)
        
                                else:
                                    st.info("Dataset is empty or missing required columns")
                            except Exception as e:
                                st.warning(f"Error analyzing Tool/Agent dataset: {e}")
                        else:
                            st.info("Tool/Agent dataset not found")
        
                    # --- Column 3: Context Pruning Dataset Quality ---
                    with qm_col3:
                        st.markdown("#### ✂️ Context Pruning")
        
                        if os.path.exists(CONTEXT_PRUNING_DATASET_PATH):
                            try:
                                df_prune = pd.read_csv(CONTEXT_PRUNING_DATASET_PATH)
        
                                if not df_prune.empty and 'expected_action' in df_prune.columns:
                                    # Calculate action balance ratio
                                    action_counts = df_prune['expected_action'].value_counts()
                                    if len(action_counts) > 1:
                                        balance_score = (action_counts.min() / action_counts.max() * 100) if action_counts.max() > 0 else 0
                                    else:
                                        balance_score = 100
        
                                    # Gauge chart
                                    fig_gauge_prune = go.Figure(go.Indicator(
                                        mode="gauge+number",
                                        value=balance_score,
                                        title={'text': "Action Balance"},
                                        gauge={
                                            'axis': {'range': [0, 100]},
                                            'bar': {'color': "darkorange"},
                                            'steps': [
                                                {'range': [0, 50], 'color': "lightgray"},
                                                {'range': [50, 80], 'color': "lightyellow"},
                                                {'range': [80, 100], 'color': "lightgreen"}
                                            ],
                                            'threshold': {
                                                'line': {'color': "red", 'width': 4},
                                                'thickness': 0.75,
                                                'value': 90
                                            }
                                        }
                                    ))
                                    fig_gauge_prune.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                                    st.plotly_chart(fig_gauge_prune, use_container_width=True, config=PLOTLY_CONFIG)
        
                                    # Metrics below gauge
                                    metric_col1, metric_col2 = st.columns(2)
                                    with metric_col1:
                                        diversity = len(action_counts)
                                        st.metric("Action Types", f"{diversity}")
                                    with metric_col2:
                                        completeness = (1 - df_prune.isnull().sum().sum() / df_prune.size) * 100 if df_prune.size > 0 else 0
                                        st.metric("Completeness", f"{completeness:.1f}%")
        
                                    # Action distribution
                                    st.markdown("**Action Distribution**")
        
                                    fig_actions = go.Figure(data=[go.Bar(
                                        y=action_counts.index.tolist(),
                                        x=action_counts.values.tolist(),
                                        orientation='h',
                                        marker_color='darkorange',
                                        text=action_counts.values.tolist(),
                                        textposition='auto'
                                    )])
                                    fig_actions.update_layout(
                                        title="Actions by Type",
                                        xaxis_title="Count",
                                        yaxis_title="",
                                        height=200,
                                        margin=dict(l=20, r=20, t=40, b=20)
                                    )
                                    st.plotly_chart(fig_actions, use_container_width=True, config=PLOTLY_CONFIG)
        
                                else:
                                    st.info("Dataset is empty or missing required columns")
                            except Exception as e:
                                st.warning(f"Error analyzing Context Pruning dataset: {e}")
                        else:
                            st.info("Context Pruning dataset not found")
        
                # Tab 4: Token Economics
                with analytics_tabs[3]:
                    st.markdown("### Token Economics Dashboard")
        
                    # Model pricing reference (per 1M tokens)
                    model_prices = {
                        'gpt-5-mini': {'input': 0.25, 'output': 2.0, 'display': 'GPT-5 Mini'},
                        'gpt-5': {'input': 1.25, 'output': 10.0, 'display': 'GPT-5'},
                        'gemini-2.5-flash': {'input': 0.30, 'output': 2.50, 'display': 'Gemini 2.5 Flash'}
                    }
        
                    # Collect token data from all datasets
                    dataset_token_data = []
        
                    # Helper function to estimate tokens (rough approximation: chars / 4)
                    def estimate_tokens(text_series):
                        return (text_series.astype(str).str.len() / 4).sum()
        
                    # Classification dataset
                    if os.path.exists(CLASSIFICATION_DATASET_PATH):
                        try:
                            df_class = pd.read_csv(CLASSIFICATION_DATASET_PATH)
                            if not df_class.empty and 'query' in df_class.columns:
                                input_tokens = estimate_tokens(df_class['query'])
                                # Assume output is classification label (small)
                                output_tokens = len(df_class) * 5  # ~5 tokens per classification
                                dataset_token_data.append({
                                    'dataset': 'Classification',
                                    'input_tokens': input_tokens,
                                    'output_tokens': output_tokens,
                                    'total_tokens': input_tokens + output_tokens,
                                    'items': len(df_class)
                                })
                        except Exception as e:
                            st.warning(f"Error processing Classification dataset: {e}")
        
                    # Tool/Agent Sequence dataset
                    if os.path.exists(TOOL_SEQUENCE_DATASET_PATH):
                        try:
                            df_tool = pd.read_csv(TOOL_SEQUENCE_DATASET_PATH)
                            if not df_tool.empty and 'query' in df_tool.columns:
                                input_tokens = estimate_tokens(df_tool['query'])
                                # Assume output is sequence (medium)
                                if 'expected_sequence' in df_tool.columns:
                                    output_tokens = estimate_tokens(df_tool['expected_sequence'])
                                else:
                                    output_tokens = len(df_tool) * 20  # ~20 tokens per sequence
                                dataset_token_data.append({
                                    'dataset': 'Tool/Agent Sequence',
                                    'input_tokens': input_tokens,
                                    'output_tokens': output_tokens,
                                    'total_tokens': input_tokens + output_tokens,
                                    'items': len(df_tool)
                                })
                        except Exception as e:
                            st.warning(f"Error processing Tool/Agent dataset: {e}")
        
                    # Context Pruning dataset
                    if os.path.exists(CONTEXT_PRUNING_DATASET_PATH):
                        try:
                            df_prune = pd.read_csv(CONTEXT_PRUNING_DATASET_PATH)
                            if not df_prune.empty:
                                # Input includes conversation history + new question
                                input_tokens = 0
                                if 'conversation_history' in df_prune.columns:
                                    input_tokens += estimate_tokens(df_prune['conversation_history'])
                                if 'new_question' in df_prune.columns:
                                    input_tokens += estimate_tokens(df_prune['new_question'])
                                # Output is action decision (small)
                                output_tokens = len(df_prune) * 10  # ~10 tokens per action
                                dataset_token_data.append({
                                    'dataset': 'Context Pruning',
                                    'input_tokens': input_tokens,
                                    'output_tokens': output_tokens,
                                    'total_tokens': input_tokens + output_tokens,
                                    'items': len(df_prune)
                                })
                        except Exception as e:
                            st.warning(f"Error processing Context Pruning dataset: {e}")
        
                    if dataset_token_data:
                        # 1. Violin plot: Token distribution comparison
                        st.markdown("#### Token Distribution Comparison")
        
                        # Prepare data for violin plot
                        violin_data = []
                        for data in dataset_token_data:
                            # Create distribution (approximate per-item tokens)
                            tokens_per_item = data['total_tokens'] / data['items'] if data['items'] > 0 else 0
                            # Simulate distribution around mean (for visualization)
                            for _ in range(min(int(data['items']), 100)):  # Cap at 100 for performance
                                violin_data.append({
                                    'Dataset': data['dataset'],
                                    'Tokens': tokens_per_item
                                })
        
                        if violin_data:
                            df_violin = pd.DataFrame(violin_data)
        
                            fig_violin = go.Figure()
        
                            for dataset_name in df_violin['Dataset'].unique():
                                dataset_tokens = df_violin[df_violin['Dataset'] == dataset_name]['Tokens']
                                fig_violin.add_trace(go.Violin(
                                    y=dataset_tokens,
                                    name=dataset_name,
                                    box_visible=True,
                                    meanline_visible=True
                                ))
        
                            fig_violin.update_layout(
                                title="Token Distribution by Dataset",
                                yaxis_title="Tokens per Item",
                                height=350,
                                margin=dict(l=20, r=20, t=40, b=40)
                            )
                            st.plotly_chart(fig_violin, use_container_width=True, config=PLOTLY_CONFIG)
        
                        # 2. Cost Calculator
                        st.markdown("#### Cost Calculator")
        
                        selected_model = st.selectbox(
                            "Select Model for Cost Estimation",
                            options=list(model_prices.keys()),
                            format_func=lambda x: model_prices[x]['display']
                        )
        
                        if selected_model:
                            pricing = model_prices[selected_model]
        
                            # Calculate costs for each dataset
                            cost_data = []
                            for data in dataset_token_data:
                                input_cost = (data['input_tokens'] / 1_000_000) * pricing['input']
                                output_cost = (data['output_tokens'] / 1_000_000) * pricing['output']
                                total_cost = input_cost + output_cost
        
                                cost_data.append({
                                    'Dataset': data['dataset'],
                                    'Input Tokens': f"{data['input_tokens']:,.0f}",
                                    'Output Tokens': f"{data['output_tokens']:,.0f}",
                                    'Input Cost': f"${input_cost:.4f}",
                                    'Output Cost': f"${output_cost:.4f}",
                                    'Total Cost': f"${total_cost:.4f}",
                                    'total_cost_numeric': total_cost
                                })
        
                            # Display cost table
                            cost_df = pd.DataFrame(cost_data)
                            display_df = cost_df.drop(columns=['total_cost_numeric'])
        
                            st.dataframe(display_df, use_container_width=True)
        
                            # Total summary
                            total_cost_all = sum([c['total_cost_numeric'] for c in cost_data])
                            st.metric("**Total Estimated Cost**", f"${total_cost_all:.4f}",
                                     help=f"Total cost for all datasets using {model_prices[selected_model]['display']}")
        
                            # 3. Bar chart: Total estimated costs by dataset
                            st.markdown("#### Cost Breakdown by Dataset")
        
                            fig_costs = go.Figure(data=[go.Bar(
                                x=[c['Dataset'] for c in cost_data],
                                y=[c['total_cost_numeric'] for c in cost_data],
                                text=[f"${c['total_cost_numeric']:.4f}" for c in cost_data],
                                textposition='auto',
                                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(cost_data)],
                                hovertemplate='<b>%{x}</b><br>Total Cost: $%{y:.4f}<extra></extra>'
                            )])
        
                            fig_costs.update_layout(
                                title=f"Estimated Costs by Dataset ({model_prices[selected_model]['display']})",
                                xaxis_title="Dataset",
                                yaxis_title="Total Cost (USD)",
                                height=350,
                                margin=dict(l=20, r=20, t=40, b=40)
                            )
                            st.plotly_chart(fig_costs, use_container_width=True, config=PLOTLY_CONFIG)
        
                            # 4. Input vs Output cost comparison
                            st.markdown("#### Input vs Output Cost Analysis")
        
                            fig_io_costs = go.Figure()
        
                            datasets = [c['Dataset'] for c in cost_data]
                            input_costs = [(float(c['Input Cost'].replace('$', ''))) for c in cost_data]
                            output_costs = [(float(c['Output Cost'].replace('$', ''))) for c in cost_data]
        
                            fig_io_costs.add_trace(go.Bar(
                                name='Input Cost',
                                x=datasets,
                                y=input_costs,
                                marker_color='lightblue',
                                text=[f"${v:.4f}" for v in input_costs],
                                textposition='auto'
                            ))
        
                            fig_io_costs.add_trace(go.Bar(
                                name='Output Cost',
                                x=datasets,
                                y=output_costs,
                                marker_color='lightcoral',
                                text=[f"${v:.4f}" for v in output_costs],
                                textposition='auto'
                            ))
        
                            fig_io_costs.update_layout(
                                title="Input vs Output Costs",
                                xaxis_title="Dataset",
                                yaxis_title="Cost (USD)",
                                barmode='group',
                                height=300,
                                margin=dict(l=20, r=20, t=40, b=40)
                            )
                            st.plotly_chart(fig_io_costs, use_container_width=True, config=PLOTLY_CONFIG)
        
                    else:
                        st.info("No datasets available for token economics analysis. Generate datasets to see cost estimates.")
        
            # Define use case configurations
            USE_CASE_CONFIGS = {
                "🚀 Development (Fast)": {
                    "description": "Quick iteration, prompt engineering, initial testing",
                    "total_items": 200,
                    "time_estimate": "2-5 minutes",
                    "cost_estimate": "$0.10-0.30",
                    "statistical_power": "Low-Medium",
                    "config": [
                        ("Classification", 100, CLASSIFICATION_DATASET_PATH),
                        ("Tool/Agent Sequence", 50, TOOL_SEQUENCE_DATASET_PATH),
                        ("Context Pruning", 50, CONTEXT_PRUNING_DATASET_PATH)
                    ]
                },
                "🧪 Testing (Recommended)": {
                    "description": "Pre-production validation, reliable metrics, stakeholder demos",
                    "total_items": 550,
                    "time_estimate": "10-15 minutes",
                    "cost_estimate": "$0.25-0.75",
                    "statistical_power": "Medium-High",
                    "config": [
                        ("Classification", 300, CLASSIFICATION_DATASET_PATH),
                        ("Tool/Agent Sequence", 150, TOOL_SEQUENCE_DATASET_PATH),
                        ("Context Pruning", 100, CONTEXT_PRUNING_DATASET_PATH)
                    ]
                },
                "🏭 Production (Robust)": {
                    "description": "Final validation, regulatory compliance, production deployment",
                    "total_items": 950,
                    "time_estimate": "20-40 minutes",
                    "cost_estimate": "$0.50-1.50",
                    "statistical_power": "High",
                    "config": [
                        ("Classification", 500, CLASSIFICATION_DATASET_PATH),
                        ("Tool/Agent Sequence", 250, TOOL_SEQUENCE_DATASET_PATH),
                        ("Context Pruning", 200, CONTEXT_PRUNING_DATASET_PATH)
                    ]
                }
            }
        
            # Use case selector
            selected_use_case = st.selectbox(
                "Select Use Case",
                options=list(USE_CASE_CONFIGS.keys()),
                index=0,  # Default to Development
                help="Choose based on your testing needs and time constraints"
            )
        
            # Display use case details
            use_case_info = USE_CASE_CONFIGS[selected_use_case]
        
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            with col_info1:
                st.metric("Total Items", use_case_info["total_items"])
            with col_info2:
                st.metric("Time Estimate", use_case_info["time_estimate"])
            with col_info3:
                st.metric("Est. Cost", use_case_info["cost_estimate"])
            with col_info4:
                st.metric("Statistical Power", use_case_info["statistical_power"])
        
            st.info(f"**Use for:** {use_case_info['description']}")
        
            # Show breakdown
            with st.expander("📊 Dataset Breakdown"):
                for data_type, size, _ in use_case_info["config"]:
                    st.write(f"- **{data_type}**: {size} items")
        
            # --- Quick Generate: Choose model used for all datasets ---
            default_quick_model = "openai/gpt-5-mini"
            if default_quick_model not in AVAILABLE_MODELS:
                default_quick_model = AVAILABLE_MODELS[0] if AVAILABLE_MODELS else OPENAI_MODEL
            default_quick_index = AVAILABLE_MODELS.index(default_quick_model) if default_quick_model in AVAILABLE_MODELS else 0
        
            quick_gen_model = st.selectbox(
                "Generation Model (Quick Generate)",
                options=AVAILABLE_MODELS,
                index=default_quick_index,
                help="Model used for Quick Generate across all selected datasets."
            )
            st.session_state["quick_gen_model"] = quick_gen_model
            st.caption(f"Using model: {quick_gen_model} via {API_ROUTING_MODE}")
        
            # Generate button
            if st.button(f"🔄 Generate All Datasets ({selected_use_case})", use_container_width=True, type="primary"):
                try:
                    datasets_config = use_case_info["config"]
                    quick_gen_model = st.session_state.get("quick_gen_model", quick_gen_model)
                    # Snapshot totals for whole batch and prepare meta collection
                    ct0 = st.session_state.cost_tracker if "cost_tracker" in st.session_state else None
                    batch_pre_totals = dict(ct0.totals) if ct0 else {}
                    batch_pre_calls = len(ct0.by_call) if ct0 else 0
                    meta_files: List[str] = []
        
                    all_success = True
        
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
        
                    total_datasets = len(datasets_config)
        
                    for idx, (data_type, size, path) in enumerate(datasets_config):
                        # Snapshot cost totals before this dataset
                        pre_totals = dict(st.session_state.cost_tracker.totals) if "cost_tracker" in st.session_state else {}
                        pre_calls = len(st.session_state.cost_tracker.by_call) if "cost_tracker" in st.session_state else 0
        
                        status_text.text(f"Generating {data_type} dataset ({size} items)... [{idx+1}/{total_datasets}]")
                        progress_bar.progress((idx) / total_datasets)
        
                        try:
                            prompt = DEFAULT_DATASET_PROMPTS[data_type]
                            df = asyncio.run(generate_synthetic_data(prompt, size, data_type, quick_gen_model))
        
                            if not df.empty:
                                df.to_csv(path, index=False)
                                # Write sidecar meta for traceability
                                # Compute cost deltas for this dataset
                                try:
                                    ct = st.session_state.cost_tracker
                                    delta_usage = {
                                        "prompt_tokens": max(ct.totals.get("prompt_tokens", 0) - pre_totals.get("prompt_tokens", 0), 0),
                                        "completion_tokens": max(ct.totals.get("completion_tokens", 0) - pre_totals.get("completion_tokens", 0), 0),
                                        "total_tokens": max(ct.totals.get("total_tokens", 0) - pre_totals.get("total_tokens", 0), 0),
                                    }
                                    delta_cost = {
                                        "input_cost_usd": round(max(ct.totals.get("input_cost_usd", 0.0) - pre_totals.get("input_cost_usd", 0.0), 0.0), 4),
                                        "output_cost_usd": round(max(ct.totals.get("output_cost_usd", 0.0) - pre_totals.get("output_cost_usd", 0.0), 0.0), 4),
                                        "total_cost_usd": round(max(ct.totals.get("total_cost_usd", 0.0) - pre_totals.get("total_cost_usd", 0.0), 0.0), 4),
                                    }
                                    delta_calls = max(len(ct.by_call) - pre_calls, 0)
                                except Exception:
                                    delta_usage, delta_cost, delta_calls = {}, {}, 0
        
                                try:
                                    meta_path = path.replace(".csv", ".meta.json")
                                    with open(meta_path, "w", encoding="utf-8") as f:
                                        json.dump({
                                            "model": quick_gen_model,
                                            "routing_mode": API_ROUTING_MODE,
                                            "when": pd.Timestamp.utcnow().isoformat(),
                                            "dataset_type": data_type,
                                            "size": size,
                                            "use_case": selected_use_case,
                                            "calls": delta_calls,
                                            "usage": delta_usage,
                                            "cost": delta_cost,
                                        }, f, indent=2)
                                    meta_files.append(meta_path)
                                except Exception:
                                    pass
                                st.success(f"✅ Generated {data_type} dataset ({len(df)} rows) using {quick_gen_model} [{API_ROUTING_MODE}]")
                            else:
                                st.warning(f"⚠️ Failed to generate {data_type} dataset")
                                all_success = False
                        except Exception as e:
                            st.error(f"Error generating {data_type} dataset: {e}")
                            all_success = False
        
                    progress_bar.progress(1.0)
                    status_text.text("Generation complete!")
        
                    if all_success:
                        # Reload datasets into session state
                        st.session_state.df = load_classification_dataset()
                        st.session_state.agent_df = load_tool_sequence_dataset()
                        st.session_state.pruning_df = load_context_pruning_dataset()
        
                        # Compute batch deltas for tokens, cost, and calls
                        try:
                            ct = st.session_state.cost_tracker
                            d_tokens = max(ct.totals.get("total_tokens", 0) - batch_pre_totals.get("total_tokens", 0), 0)
                            d_cost = round(max(ct.totals.get("total_cost_usd", 0.0) - batch_pre_totals.get("total_cost_usd", 0.0), 0.0), 4)
                            d_calls = max(len(ct.by_call) - batch_pre_calls, 0)
                        except Exception:
                            d_tokens, d_cost, d_calls = 0, 0.0, 0
        
                        # Persist meta file list and summary for the expander/history
                        st.session_state["last_generated_meta_files"] = meta_files
                        summary = {
                            "when": pd.Timestamp.utcnow().isoformat(),
                            "use_case": selected_use_case,
                            "model": quick_gen_model,
                            "routing": API_ROUTING_MODE,
                            "total_items": use_case_info['total_items'],
                            "tokens": d_tokens,
                            "cost_usd": d_cost,
                            "calls": d_calls,
                            "meta_files": meta_files,
                        }
                        st.session_state["last_generation_summary"] = summary
                        if "generation_history" not in st.session_state:
                            st.session_state["generation_history"] = []
                        st.session_state["generation_history"].append(summary)
        
                        st.success(f"✅ All datasets generated successfully! Total: {use_case_info['total_items']} items using {quick_gen_model} [{API_ROUTING_MODE}] • Tokens: +{d_tokens:,} • Cost: +${d_cost:.4f} • Calls: +{d_calls}")
                        st.balloons()
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to generate datasets: {e}")
        
            # Compact expander to view last generation metadata
            if st.session_state.get("last_generated_meta_files"):
                with st.expander("🧾 View last generation metadata"):
                    files = st.session_state.get("last_generated_meta_files", [])
                    # Compact summary table
                    rows = []
                    for p in files:
                        try:
                            with open(p, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                                rows.append({
                                    "dataset_type": meta.get("dataset_type"),
                                    "size": meta.get("size"),
                                    "total_tokens": ((meta.get("usage") or {}).get("total_tokens")),
                                    "total_cost_usd": ((meta.get("cost") or {}).get("total_cost_usd")),
                                })
                        except Exception:
                            pass
                    if rows:
                        try:
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=min(300, 60+28*len(rows)))
                        except Exception:
                            st.write(rows)
                    # Raw JSON entries
                    for p in files:
                        try:
                            st.markdown(f"**{os.path.basename(p)}**")
                            with open(p, "r", encoding="utf-8") as f:
                                st.json(json.load(f))
                        except Exception as e:
                            st.text(f"{p}: {e}")
        
            # History expander
            if st.session_state.get("generation_history"):
                with st.expander("🗂️ Generation history"):
                    hist = st.session_state.get("generation_history", [])
                    try:
                        dfh = pd.DataFrame(hist)
                        st.dataframe(dfh[["when","use_case","model","routing","total_items","tokens","cost_usd","calls"]].sort_values("when", ascending=False), use_container_width=True)
                    except Exception:
                        st.write(hist)
        
            st.divider()
        
            # --- PATCH 20: Restructure Tab 0 UI for dynamic prompt generation ---
            # Initialize prompt key
            if 'generation_prompt_text' not in st.session_state:
                st.session_state['generation_prompt_text'] = SUGGESTED_PROMPTS["Classification"][0]
        
            st.subheader("🎯 Generate Individual Dataset")
            c_model, c_size, c_type = st.columns(3)
        
            with c_model:
                # --- UPDATED: Flexible model selection from all available models ---
                default_gen_model = "openai/gpt-5-mini"
                if default_gen_model not in AVAILABLE_MODELS:
                    default_gen_model = AVAILABLE_MODELS[0] if AVAILABLE_MODELS else OPENAI_MODEL
        
                default_index = AVAILABLE_MODELS.index(default_gen_model) if default_gen_model in AVAILABLE_MODELS else 0
        
                gen_model = st.selectbox(
                    "Generation Model",
                    options=AVAILABLE_MODELS,
                    index=default_index,
                    help="Select any model to generate synthetic data. Defaults to gpt-5-mini."
                )
        
            with c_size:
                size_options = {5: "5 Pairs", 25: "25 Pairs", 100: "100 Pairs", 1000: "1000 Pairs"}
                data_size_label = st.selectbox("Dataset Size", options=list(size_options.values()), index=2)
                data_size = list(size_options.keys())[list(size_options.values()).index(data_size_label)]
        
            with c_type:
                # --- PATCH 17: Added "Context Pruning" option ---
                data_type = st.selectbox("Data Type", ["Classification", "Tool/Agent Sequence", "Context Pruning"])
        
            # Define the helper function for random prompt selection (user-input-aware)
            def set_random_prompt(current_data_type):
                """Generate random prompt that's aware of user's existing input."""
                import random
        
                prompts = SUGGESTED_PROMPTS.get(current_data_type, [])
                if not prompts:
                    st.session_state['generation_prompt_text'] = f"Generate synthetic data for the '{current_data_type}' task."
                    return
        
                # Get current user input
                current_input = st.session_state.get('generation_prompt_text', '')
        
                # Select base prompt
                base_prompt = random.choice(prompts)
        
                # Build context
                context = {
                    'dataset_size': st.session_state.get('data_size', 100),
                    'columns': ['query', 'classification'] if current_data_type == 'Classification' else ['query', 'expected_sequence'],
                    'suggested_approach': base_prompt.split('\n')[0] if '\n' in base_prompt else base_prompt[:100],
                    'success_criteria': '85%+ accuracy on test set'
                }
        
                # Enhance with user input
                enhanced = enhance_prompt_with_user_input(base_prompt, current_input, context)
        
                st.session_state['generation_prompt_text'] = enhanced
        
            # 1. Random Prompt Generator Button
            if st.button(f"🎲 Randomly Generate Prompt for {data_type}", use_container_width=True):
                set_random_prompt(data_type)
                st.rerun() # Rerun to update the text area immediately
        
            # 2. Text Area (linked to session state)
            st.text_area(
                "Describe the dataset requirements:",
                height=150,
                key='generation_prompt_text',
                help="Provide a detailed description of the type of queries, the required complexity, and the target labels/sequences."
            )
        
            # 3. Suggestion Pills
            st.markdown("##### Suggested Starting Points (Click to use):")
            selected_prompts = SUGGESTED_PROMPTS.get(data_type, [])
        
            if selected_prompts:
                cols = st.columns(len(selected_prompts))
                for i, prompt in enumerate(selected_prompts):
                    with cols[i]:
                        if st.button(prompt[:30] + "...", key=f"pill_{data_type}_{i}", help=prompt, use_container_width=True):
                            st.session_state['generation_prompt_text'] = prompt
                            st.rerun() # Rerun to update the text area
            else:
                st.info("No suggestions available for this data type.")
        
            # 4. Execution Button (reads from session state)
            if st.button("Generate & Save Dataset", use_container_width=True):
                generation_prompt_used = st.session_state.get('generation_prompt_text', "").strip()
                if not generation_prompt_used:
                    st.warning("Please provide a dataset description prompt.")
                else:
                    # Need to run generation synchronously in Streamlit context
                    with st.spinner(f"Generating {data_size} {data_type} items..."):
                        new_df = asyncio.run(generate_synthetic_data(generation_prompt_used, data_size, data_type, gen_model))
        
                    if not new_df.empty:
        
                        # Check if we are generating for the classification tests (Test 1-3)
                        if data_type == "Classification" and "classification" in new_df.columns:
                            # Re-initialize the dataframe with all necessary classification skeleton columns
                            st.session_state.df = new_df[["query", "classification"]].copy()
                            for col in SKELETON_COLUMNS:
                                if col not in st.session_state.df.columns:
                                    st.session_state.df[col] = None
        
                            # Save to file
                            save_dataset_to_file(new_df[["query", "classification"]], data_type, model_used=gen_model, routing_mode=API_ROUTING_MODE)
                            st.success(f"Generated {len(new_df)} classification items using {gen_model} [{API_ROUTING_MODE}] and updated the main dataset (Tests 1-3).")
        
                            # Visualize generated dataset
                            st.subheader("📊 Generated Dataset Analysis")
                            visualize_dataset_composition(new_df, dataset_type=data_type)
                            st.dataframe(new_df.head(10), use_container_width=True)
        
                        # If it's an Agent test set, save it separately/use a different structure
                        elif data_type == "Tool/Agent Sequence" and "expected_sequence" in new_df.columns:
                            # Save agent data to a specific session state key
                            st.session_state.agent_df = new_df[["query", "expected_sequence"]].copy()
        
                            # Save to file
                            save_dataset_to_file(new_df, data_type, model_used=gen_model, routing_mode=API_ROUTING_MODE)
                            st.success(f"Generated {len(new_df)} agent sequence items using {gen_model} [{API_ROUTING_MODE}]. Ready for Test 5.")
        
                            # Visualize generated dataset
                            st.subheader("📊 Generated Dataset Analysis")
                            visualize_dataset_composition(new_df, dataset_type=data_type)
                            st.dataframe(new_df.head(10), use_container_width=True)
        
                        # --- PATCH 17: Save Pruning Data to session state ---
                        elif data_type == "Context Pruning" and "new_question" in new_df.columns:
                            # Test 4 requires specific columns defined in PruningDataItem
                            required_cols = list(PruningDataItem.model_fields.keys())
                            st.session_state.pruning_df = new_df[[c for c in required_cols if c in new_df.columns]].copy()
        
                            # Save to file
                            save_dataset_to_file(new_df, data_type, model_used=gen_model, routing_mode=API_ROUTING_MODE)
                            st.success(f"Generated {len(new_df)} context pruning cases using {gen_model} [{API_ROUTING_MODE}]. Ready for Test 4.")
        
                            # Visualize generated dataset
                            st.subheader("📊 Generated Dataset Analysis")
                            visualize_dataset_composition(new_df, dataset_type=data_type)
                            st.dataframe(new_df.head(10), use_container_width=True)
                        # ----------------------------------------------------
        
                        st.dataframe(new_df, use_container_width=True)
        
                        # --- Force rerun to update the Dataset Status display ---
                        st.rerun()
                    else:
                        st.error("Data generation failed to produce valid output.")
            # -------------------------------------------------------------
        
        
        # ---------- Test 1 ----------
