"""
Visual LLM Results Synthesis Module

Provides comprehensive analysis synthesis:
- Normalized agreement metrics with visualization
- Model ranking and complementary strengths analysis
- Prompt optimization suggestions
- Actionable insights and recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import pearsonr, spearmanr


def normalize_agreement_data(
    results: List[Dict[str, Any]],
    selected_models: List[str]
) -> pd.DataFrame:
    """
    Normalize sparse agreement data by filling missing values and handling NaN.
    
    Args:
        results: List of analysis results
        selected_models: List of model identifiers
        
    Returns:
        Normalized DataFrame with agreement scores
    """
    # Extract pairwise agreements
    agreement_data = []
    
    for result in results:
        image_name = result.get('image_name', 'unknown')
        model_results = result.get('model_results', {})
        
        # Get all model pairs
        models = list(model_results.keys())
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                # Calculate agreement based on confidence similarity
                conf1 = model_results[model1].confidence if hasattr(model_results[model1], 'confidence') else 0
                conf2 = model_results[model2].confidence if hasattr(model_results[model2], 'confidence') else 0
                
                # Agreement score: 1 - abs(conf1 - conf2)
                agreement = 1.0 - abs(conf1 - conf2)
                
                agreement_data.append({
                    'image': image_name,
                    'model1': model1,
                    'model2': model2,
                    'agreement': agreement,
                    'conf1': conf1,
                    'conf2': conf2
                })
    
    df = pd.DataFrame(agreement_data)
    
    # Fill missing values with 0 (no agreement data)
    df['agreement'] = df['agreement'].fillna(0.0)
    df['conf1'] = df['conf1'].fillna(0.0)
    df['conf2'] = df['conf2'].fillna(0.0)
    
    return df


def create_agreement_heatmap(agreement_df: pd.DataFrame) -> go.Figure:
    """
    Create interactive heatmap of model agreement.
    
    Args:
        agreement_df: Normalized agreement DataFrame
        
    Returns:
        Plotly figure
    """
    # Pivot to create matrix
    models = sorted(set(agreement_df['model1'].unique()) | set(agreement_df['model2'].unique()))
    
    # Create symmetric matrix
    matrix = np.zeros((len(models), len(models)))
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i == j:
                matrix[i, j] = 1.0  # Perfect self-agreement
            else:
                # Get agreement score
                subset = agreement_df[
                    ((agreement_df['model1'] == model1) & (agreement_df['model2'] == model2)) |
                    ((agreement_df['model1'] == model2) & (agreement_df['model2'] == model1))
                ]
                if not subset.empty:
                    matrix[i, j] = subset['agreement'].mean()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=models,
        y=models,
        colorscale='RdYlGn',
        zmid=0.5,
        text=np.round(matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Agreement")
    ))
    
    fig.update_layout(
        title="Model Agreement Heatmap",
        xaxis_title="Model",
        yaxis_title="Model",
        height=500
    )
    
    return fig


def create_confidence_correlation_plot(
    results: List[Dict[str, Any]]
) -> go.Figure:
    """
    Create scatter plot showing confidence vs agreement correlation.
    
    Args:
        results: List of analysis results
        
    Returns:
        Plotly figure with correlation visualization
    """
    # Extract data
    data_points = []
    
    for result in results:
        model_results = result.get('model_results', {})
        
        # Calculate average confidence
        confidences = [
            m.confidence if hasattr(m, 'confidence') else 0
            for m in model_results.values()
        ]
        avg_conf = np.mean(confidences) if confidences else 0
        
        # Calculate agreement (std dev of confidences, inverted)
        agreement = 1.0 - (np.std(confidences) if len(confidences) > 1 else 0)
        
        data_points.append({
            'image': result.get('image_name', 'unknown'),
            'avg_confidence': avg_conf,
            'agreement': agreement,
            'num_models': len(confidences)
        })
    
    df = pd.DataFrame(data_points)

    # Calculate correlation with error handling
    corr_text = "Insufficient data for correlation"
    if len(df) > 2:
        try:
            # Check if we have enough variance
            if df['avg_confidence'].std() > 0.01 and df['agreement'].std() > 0.01:
                corr, p_value = pearsonr(df['avg_confidence'], df['agreement'])
                corr_text = f"Pearson r = {corr:.3f} (p = {p_value:.3f})"
            else:
                corr_text = "Insufficient variance in data"
        except (ValueError, RuntimeError):
            corr_text = "Correlation calculation failed"
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='avg_confidence',
        y='agreement',
        size='num_models',
        hover_data=['image'],
        title=f"Confidence vs Agreement Correlation<br><sub>{corr_text}</sub>",
        labels={
            'avg_confidence': 'Average Confidence',
            'agreement': 'Model Agreement',
            'num_models': 'Number of Models'
        }
    )
    
    # Add trend line with error handling
    if len(df) > 2:
        try:
            # Check if we have enough variance in the data
            if df['avg_confidence'].std() > 0.01 and df['agreement'].std() > 0.01:
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
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            # Skip trend line if fitting fails
            pass

    return fig


def rank_models_by_task(
    results: List[Dict[str, Any]],
    task_description: str
) -> Dict[str, Any]:
    """
    Rank models based on performance for specific task.
    
    Args:
        results: List of analysis results
        task_description: Original task description
        
    Returns:
        Dict with rankings, scores, and complementary strengths
    """
    model_scores = defaultdict(lambda: {
        'total_confidence': 0.0,
        'count': 0,
        'avg_confidence': 0.0,
        'consistency': 0.0,
        'detail_score': 0.0
    })
    
    # Collect metrics for each model
    for result in results:
        model_results = result.get('model_results', {})
        
        for model_name, analysis in model_results.items():
            conf = analysis.confidence if hasattr(analysis, 'confidence') else 0
            rationale_len = len(analysis.rationale) if hasattr(analysis, 'rationale') else 0
            artifacts = len(analysis.detected_artifacts) if hasattr(analysis, 'detected_artifacts') else 0
            
            model_scores[model_name]['total_confidence'] += conf
            model_scores[model_name]['count'] += 1
            model_scores[model_name]['detail_score'] += (rationale_len / 1000.0) + (artifacts * 0.1)
    
    # Calculate averages and rankings
    rankings = []
    for model_name, scores in model_scores.items():
        if scores['count'] > 0:
            avg_conf = scores['total_confidence'] / scores['count']
            avg_detail = scores['detail_score'] / scores['count']
            
            # Overall score: weighted combination
            overall_score = (avg_conf * 0.6) + (avg_detail * 0.4)
            
            rankings.append({
                'model': model_name,
                'overall_score': overall_score,
                'avg_confidence': avg_conf,
                'avg_detail_score': avg_detail,
                'num_analyses': scores['count']
            })
    
    # Sort by overall score
    rankings.sort(key=lambda x: x['overall_score'], reverse=True)
    
    # Identify complementary strengths
    strengths = identify_complementary_strengths(results)
    
    return {
        'rankings': rankings,
        'best_model': rankings[0]['model'] if rankings else None,
        'complementary_strengths': strengths,
        'task_description': task_description
    }


def identify_complementary_strengths(
    results: List[Dict[str, Any]]
) -> Dict[str, List[str]]:
    """
    Identify what each model is uniquely good at.
    
    Args:
        results: List of analysis results
        
    Returns:
        Dict mapping model names to their unique strengths
    """
    strengths = defaultdict(list)
    
    # Analyze each model's characteristics
    model_characteristics = defaultdict(lambda: {
        'high_confidence_count': 0,
        'detailed_rationale_count': 0,
        'artifact_detection_count': 0,
        'total_count': 0
    })
    
    for result in results:
        model_results = result.get('model_results', {})
        
        for model_name, analysis in model_results.items():
            chars = model_characteristics[model_name]
            chars['total_count'] += 1
            
            # High confidence
            if hasattr(analysis, 'confidence') and analysis.confidence > 0.8:
                chars['high_confidence_count'] += 1
            
            # Detailed rationale
            if hasattr(analysis, 'rationale') and len(analysis.rationale) > 500:
                chars['detailed_rationale_count'] += 1
            
            # Artifact detection
            if hasattr(analysis, 'detected_artifacts') and len(analysis.detected_artifacts) > 0:
                chars['artifact_detection_count'] += 1
    
    # Determine strengths based on relative performance
    for model_name, chars in model_characteristics.items():
        total = chars['total_count']
        if total == 0:
            continue
        
        if chars['high_confidence_count'] / total > 0.7:
            strengths[model_name].append("High confidence predictions")
        
        if chars['detailed_rationale_count'] / total > 0.6:
            strengths[model_name].append("Detailed explanations")
        
        if chars['artifact_detection_count'] / total > 0.5:
            strengths[model_name].append("Artifact detection")
    
    return dict(strengths)


def generate_prompt_improvements(
    results: List[Dict[str, Any]],
    task_description: str,
    model_rankings: Dict[str, Any]
) -> Dict[str, str]:
    """
    Generate improved prompts for each model based on performance.

    Args:
        results: List of analysis results
        task_description: Original task description
        model_rankings: Model ranking results

    Returns:
        Dict mapping model names to improved prompts
    """
    improved_prompts = {}

    # Analyze common issues
    for ranking in model_rankings['rankings']:
        model_name = ranking['model']
        avg_conf = ranking['avg_confidence']

        # Base prompt
        base_prompt = task_description

        # Add model-specific enhancements
        enhancements = []

        if avg_conf < 0.7:
            enhancements.append("Be specific and confident in your analysis.")

        if ranking['avg_detail_score'] < 0.5:
            enhancements.append("Provide detailed explanations for your findings.")

        # Combine
        if enhancements:
            improved_prompt = f"{base_prompt}\n\nAdditional guidance:\n" + "\n".join(f"- {e}" for e in enhancements)
        else:
            improved_prompt = base_prompt

        improved_prompts[model_name] = improved_prompt

    return improved_prompts


def create_comprehensive_synthesis(
    results: List[Dict[str, Any]],
    task_description: str,
    selected_models: List[str]
) -> Dict[str, Any]:
    """
    Create comprehensive synthesis of all results.

    Args:
        results: List of analysis results
        task_description: Original task description
        selected_models: List of model identifiers

    Returns:
        Complete synthesis with all metrics, rankings, and recommendations
    """
    # Normalize agreement data
    agreement_df = normalize_agreement_data(results, selected_models)

    # Rank models
    model_rankings = rank_models_by_task(results, task_description)

    # Generate improved prompts
    improved_prompts = generate_prompt_improvements(
        results, task_description, model_rankings
    )

    # Create visualizations
    agreement_heatmap = create_agreement_heatmap(agreement_df)
    confidence_correlation = create_confidence_correlation_plot(results)

    # Generate actionable insights
    insights = generate_actionable_insights(
        results, model_rankings, agreement_df
    )

    return {
        'agreement_data': agreement_df,
        'model_rankings': model_rankings,
        'improved_prompts': improved_prompts,
        'visualizations': {
            'agreement_heatmap': agreement_heatmap,
            'confidence_correlation': confidence_correlation
        },
        'insights': insights,
        'summary': {
            'total_images': len(results),
            'total_models': len(selected_models),
            'best_model': model_rankings['best_model'],
            'avg_agreement': agreement_df['agreement'].mean() if not agreement_df.empty else 0,
            'task_description': task_description
        }
    }


def generate_actionable_insights(
    results: List[Dict[str, Any]],
    model_rankings: Dict[str, Any],
    agreement_df: pd.DataFrame
) -> List[Dict[str, str]]:
    """
    Generate actionable insights from analysis.

    Args:
        results: List of analysis results
        model_rankings: Model ranking results
        agreement_df: Agreement DataFrame

    Returns:
        List of insight dicts with category, insight, and action
    """
    insights = []

    # Best model insight
    if model_rankings['best_model']:
        insights.append({
            'category': 'Model Selection',
            'insight': f"{model_rankings['best_model']} performed best overall",
            'action': f"Use {model_rankings['best_model']} as primary model for this task type",
            'priority': 'high'
        })

    # Agreement insight
    avg_agreement = agreement_df['agreement'].mean() if not agreement_df.empty else 0
    if avg_agreement < 0.6:
        insights.append({
            'category': 'Model Agreement',
            'insight': f"Low model agreement ({avg_agreement:.1%}) indicates task ambiguity",
            'action': "Refine task description or use ensemble voting",
            'priority': 'high'
        })
    elif avg_agreement > 0.8:
        insights.append({
            'category': 'Model Agreement',
            'insight': f"High model agreement ({avg_agreement:.1%}) indicates clear task",
            'action': "Consider using single model to reduce costs",
            'priority': 'medium'
        })

    # Complementary strengths
    strengths = model_rankings.get('complementary_strengths', {})
    if len(strengths) > 1:
        insights.append({
            'category': 'Ensemble Strategy',
            'insight': "Models have complementary strengths",
            'action': "Use ensemble approach: " + ", ".join([
                f"{model} for {', '.join(s)}"
                for model, s in list(strengths.items())[:2]
            ]),
            'priority': 'medium'
        })

    # Flagged images
    low_conf_images = []
    for result in results:
        model_results = result.get('model_results', {})
        avg_conf = np.mean([
            m.confidence if hasattr(m, 'confidence') else 0
            for m in model_results.values()
        ])
        if avg_conf < 0.5:
            low_conf_images.append(result.get('image_name', 'unknown'))

    if low_conf_images:
        insights.append({
            'category': 'Quality Control',
            'insight': f"{len(low_conf_images)} images have low confidence",
            'action': f"Review these images manually: {', '.join(low_conf_images[:3])}{'...' if len(low_conf_images) > 3 else ''}",
            'priority': 'high'
        })

    return insights


