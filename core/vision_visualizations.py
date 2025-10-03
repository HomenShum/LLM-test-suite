"""
Visualization functions for Test 6: Visual LLM Testing.

Creates Plotly charts for:
- Human vs LLM rating comparisons
- Artifact frequency analysis
- Model agreement heatmaps
- Performance metrics
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any
import numpy as np

from core.models import VisualLLMAnalysis, VisualLLMComparisonResult


def create_rating_comparison_scatter(
    results: List[Dict[str, Any]],
    rating_type: str = "movement_rating",
    title: str = None
) -> go.Figure:
    """
    Create scatter plot comparing human vs LLM ratings.
    
    Args:
        results: List of result dicts with human_ratings and model_results
        rating_type: Type of rating to compare
        title: Chart title
    
    Returns:
        Plotly Figure object
    """
    if title is None:
        title = f"Human vs LLM {rating_type.replace('_', ' ').title()} Comparison"
    
    # Prepare data
    data = []
    for result in results:
        avatar_id = result.get("avatar_id", "unknown")
        human_rating = result.get("human_ratings", {}).get(rating_type.replace("_rating", ""))
        
        for model_name, analysis in result.get("model_results", {}).items():
            llm_rating = analysis.get(rating_type)
            
            if human_rating is not None and llm_rating is not None:
                data.append({
                    "avatar_id": avatar_id,
                    "model": model_name,
                    "human_rating": human_rating,
                    "llm_rating": llm_rating
                })
    
    if not data:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for comparison",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    df = pd.DataFrame(data)
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x="human_rating",
        y="llm_rating",
        color="model",
        hover_data=["avatar_id"],
        title=title,
        labels={
            "human_rating": "Human Rating",
            "llm_rating": "LLM Rating"
        },
        trendline="ols"  # Add trend line
    )
    
    # Add diagonal line (perfect agreement)
    fig.add_trace(go.Scatter(
        x=[1, 5],
        y=[1, 5],
        mode="lines",
        name="Perfect Agreement",
        line=dict(dash="dash", color="gray")
    ))
    
    fig.update_layout(
        xaxis=dict(range=[0.5, 5.5]),
        yaxis=dict(range=[0.5, 5.5]),
        height=500
    )
    
    return fig


def create_artifact_frequency_chart(
    results: List[Dict[str, Any]],
    min_frequency: int = 1
) -> go.Figure:
    """
    Create bar chart showing artifact detection frequency.
    
    Args:
        results: List of result dicts with model_results
        min_frequency: Minimum frequency to include
    
    Returns:
        Plotly Figure object
    """
    # Count artifacts
    artifact_counts = {}
    
    for result in results:
        for model_name, analysis in result.get("model_results", {}).items():
            for artifact in analysis.get("detected_artifacts", []):
                artifact_lower = artifact.lower()
                if artifact_lower not in artifact_counts:
                    artifact_counts[artifact_lower] = {"count": 0, "original": artifact}
                artifact_counts[artifact_lower]["count"] += 1
    
    # Filter by minimum frequency
    filtered = {
        k: v for k, v in artifact_counts.items()
        if v["count"] >= min_frequency
    }
    
    if not filtered:
        fig = go.Figure()
        fig.add_annotation(
            text="No artifacts detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Sort by frequency
    sorted_artifacts = sorted(
        filtered.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )
    
    artifacts = [v["original"] for k, v in sorted_artifacts]
    counts = [v["count"] for k, v in sorted_artifacts]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=artifacts,
            y=counts,
            marker_color='indianred',
            text=counts,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Artifact Detection Frequency",
        xaxis_title="Artifact Type",
        yaxis_title="Detection Count",
        height=400
    )
    
    return fig


def create_model_agreement_heatmap(
    results: List[Dict[str, Any]]
) -> go.Figure:
    """
    Create heatmap showing agreement between models.
    
    Args:
        results: List of result dicts with model_results
    
    Returns:
        Plotly Figure object
    """
    # Get all model names
    all_models = set()
    for result in results:
        all_models.update(result.get("model_results", {}).keys())
    
    model_list = sorted(all_models)
    
    if len(model_list) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 models for comparison",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Calculate pairwise agreement
    agreement_matrix = np.zeros((len(model_list), len(model_list)))
    
    for i, model_i in enumerate(model_list):
        for j, model_j in enumerate(model_list):
            if i == j:
                agreement_matrix[i][j] = 1.0
            else:
                agreements = []
                
                for result in results:
                    analysis_i = result.get("model_results", {}).get(model_i)
                    analysis_j = result.get("model_results", {}).get(model_j)
                    
                    if analysis_i and analysis_j:
                        # Compare ratings
                        for rating_type in ["movement_rating", "visual_quality_rating", "artifact_presence_rating"]:
                            rating_i = analysis_i.get(rating_type)
                            rating_j = analysis_j.get(rating_type)
                            
                            if rating_i is not None and rating_j is not None:
                                # Calculate agreement (1 - normalized difference)
                                diff = abs(rating_i - rating_j)
                                agreement = 1.0 - (diff / 4.0)  # Max diff is 4 (5-1)
                                agreements.append(agreement)
                
                if agreements:
                    agreement_matrix[i][j] = sum(agreements) / len(agreements)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=agreement_matrix,
        x=model_list,
        y=model_list,
        colorscale='RdYlGn',
        text=np.round(agreement_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Agreement")
    ))
    
    fig.update_layout(
        title="Model Agreement Heatmap",
        xaxis_title="Model",
        yaxis_title="Model",
        height=500
    )
    
    return fig


def create_confidence_distribution(
    results: List[Dict[str, Any]]
) -> go.Figure:
    """
    Create histogram showing confidence score distribution.
    
    Args:
        results: List of result dicts with model_results
    
    Returns:
        Plotly Figure object
    """
    # Collect confidence scores
    confidence_data = []
    
    for result in results:
        for model_name, analysis in result.get("model_results", {}).items():
            confidence = analysis.get("confidence")
            if confidence is not None:
                confidence_data.append({
                    "model": model_name,
                    "confidence": confidence
                })
    
    if not confidence_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No confidence data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    df = pd.DataFrame(confidence_data)
    
    # Create histogram
    fig = px.histogram(
        df,
        x="confidence",
        color="model",
        nbins=20,
        title="Confidence Score Distribution",
        labels={"confidence": "Confidence Score"},
        barmode="overlay",
        opacity=0.7
    )
    
    fig.update_layout(
        xaxis=dict(range=[0, 1]),
        height=400
    )

    return fig


def create_performance_dashboard(
    results: List[Dict[str, Any]]
) -> go.Figure:
    """
    Create comprehensive dashboard with multiple metrics.

    Args:
        results: List of result dicts with model_results

    Returns:
        Plotly Figure object with subplots
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Average Ratings by Model",
            "Artifact Detection Rate",
            "Rating Distribution",
            "Model Performance Summary"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "box"}, {"type": "table"}]
        ]
    )

    # Collect data
    model_ratings = {}
    model_artifacts = {}
    all_ratings = []

    for result in results:
        for model_name, analysis in result.get("model_results", {}).items():
            if model_name not in model_ratings:
                model_ratings[model_name] = []
                model_artifacts[model_name] = 0

            # Collect ratings
            for rating_type in ["movement_rating", "visual_quality_rating", "artifact_presence_rating"]:
                rating = analysis.get(rating_type)
                if rating is not None:
                    model_ratings[model_name].append(rating)
                    all_ratings.append({
                        "model": model_name,
                        "rating": rating,
                        "type": rating_type
                    })

            # Count artifacts
            artifacts = analysis.get("detected_artifacts", [])
            if artifacts and artifacts != ["none"] and artifacts != ["none detected"]:
                model_artifacts[model_name] += len(artifacts)

    # Subplot 1: Average ratings
    models = list(model_ratings.keys())
    avg_ratings = [
        sum(ratings) / len(ratings) if ratings else 0
        for ratings in model_ratings.values()
    ]

    fig.add_trace(
        go.Bar(x=models, y=avg_ratings, name="Avg Rating", marker_color='lightblue'),
        row=1, col=1
    )

    # Subplot 2: Artifact detection rate
    artifact_counts = [model_artifacts.get(m, 0) for m in models]

    fig.add_trace(
        go.Bar(x=models, y=artifact_counts, name="Artifacts", marker_color='coral'),
        row=1, col=2
    )

    # Subplot 3: Rating distribution (box plot)
    if all_ratings:
        df_ratings = pd.DataFrame(all_ratings)
        for model in models:
            model_data = df_ratings[df_ratings["model"] == model]["rating"]
            fig.add_trace(
                go.Box(y=model_data, name=model),
                row=2, col=1
            )

    # Subplot 4: Performance summary table
    summary_data = []
    for model in models:
        ratings = model_ratings.get(model, [])
        artifacts = model_artifacts.get(model, 0)

        summary_data.append([
            model,
            f"{sum(ratings) / len(ratings):.2f}" if ratings else "N/A",
            str(artifacts),
            str(len(results))
        ])

    fig.add_trace(
        go.Table(
            header=dict(values=["Model", "Avg Rating", "Artifacts", "Images"]),
            cells=dict(values=list(zip(*summary_data)) if summary_data else [[], [], [], []])
        ),
        row=2, col=2
    )

    fig.update_layout(
        title_text="Visual LLM Performance Dashboard",
        showlegend=False,
        height=800
    )

    return fig


def create_correlation_matrix(
    results: List[Dict[str, Any]]
) -> go.Figure:
    """
    Create correlation matrix between different rating types.

    Args:
        results: List of result dicts with model_results

    Returns:
        Plotly Figure object
    """
    # Collect all ratings
    rating_data = {
        "movement": [],
        "visual_quality": [],
        "artifact_presence": []
    }

    for result in results:
        for model_name, analysis in result.get("model_results", {}).items():
            movement = analysis.get("movement_rating")
            visual = analysis.get("visual_quality_rating")
            artifacts = analysis.get("artifact_presence_rating")

            if all(r is not None for r in [movement, visual, artifacts]):
                rating_data["movement"].append(movement)
                rating_data["visual_quality"].append(visual)
                rating_data["artifact_presence"].append(artifacts)

    if not rating_data["movement"]:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for correlation analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig

    # Calculate correlation matrix
    df = pd.DataFrame(rating_data)
    corr_matrix = df.corr()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=["Movement", "Visual Quality", "Artifact Presence"],
        y=["Movement", "Visual Quality", "Artifact Presence"],
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 14},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title="Rating Correlation Matrix",
        height=500
    )

    return fig


