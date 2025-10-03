"""
Test 6: Synthesis Results Display

Displays comprehensive synthesis of visual LLM analysis results including:
- Normalized agreement visualizations
- Model rankings and complementary strengths
- Prompt optimization suggestions
- Actionable insights
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime

from core.visual_results_synthesis import create_comprehensive_synthesis
from utils.plotly_config import PLOTLY_CONFIG


def display_synthesis_results(
    results: List[Dict[str, Any]],
    task_description: str,
    selected_models: List[str]
) -> None:
    """
    Display comprehensive synthesis of analysis results.
    
    Args:
        results: List of analysis results
        task_description: Original task description
        selected_models: List of model identifiers
    """
    st.markdown("## 🎯 Comprehensive Analysis Synthesis")
    
    with st.spinner("Generating comprehensive synthesis..."):
        synthesis = create_comprehensive_synthesis(
            results, task_description, selected_models
        )
    
    # Summary metrics at top
    display_summary_metrics(synthesis['summary'])
    
    st.divider()
    
    # Create tabs for different views
    tabs = st.tabs([
        "📊 Model Rankings",
        "🤝 Agreement Analysis",
        "💡 Insights & Actions",
        "✨ Prompt Optimization",
        "📈 Visualizations"
    ])
    
    # Tab 1: Model Rankings
    with tabs[0]:
        display_model_rankings(synthesis['model_rankings'])
    
    # Tab 2: Agreement Analysis
    with tabs[1]:
        display_agreement_analysis(synthesis['agreement_data'])
    
    # Tab 3: Insights & Actions
    with tabs[2]:
        display_actionable_insights(synthesis['insights'])
    
    # Tab 4: Prompt Optimization
    with tabs[3]:
        display_prompt_optimization(synthesis['improved_prompts'], task_description)
    
    # Tab 5: Visualizations
    with tabs[4]:
        display_visualizations(synthesis['visualizations'])


def display_summary_metrics(summary: Dict[str, Any]) -> None:
    """Display summary metrics in columns."""
    st.markdown("### 📋 Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Images",
            summary['total_images'],
            help="Number of images analyzed"
        )
    
    with col2:
        st.metric(
            "Models Tested",
            summary['total_models'],
            help="Number of visual LLM models"
        )
    
    with col3:
        st.metric(
            "Best Model",
            summary['best_model'],
            help="Highest performing model"
        )
    
    with col4:
        st.metric(
            "Avg Agreement",
            f"{summary['avg_agreement']:.1%}",
            help="Average inter-model agreement"
        )


def display_model_rankings(rankings_data: Dict[str, Any]) -> None:
    """Display model rankings and complementary strengths."""
    st.markdown("### 🏆 Model Performance Rankings")
    
    rankings = rankings_data['rankings']
    
    if not rankings:
        st.warning("No ranking data available")
        return
    
    # Create ranking table
    ranking_df = pd.DataFrame(rankings)
    ranking_df['rank'] = range(1, len(ranking_df) + 1)
    
    # Format for display
    display_df = ranking_df[[
        'rank', 'model', 'overall_score', 'avg_confidence', 
        'avg_detail_score', 'num_analyses'
    ]].copy()
    
    display_df.columns = [
        'Rank', 'Model', 'Overall Score', 'Avg Confidence',
        'Detail Score', 'Analyses'
    ]
    
    # Format numbers
    display_df['Overall Score'] = display_df['Overall Score'].apply(lambda x: f"{x:.3f}")
    display_df['Avg Confidence'] = display_df['Avg Confidence'].apply(lambda x: f"{x:.1%}")
    display_df['Detail Score'] = display_df['Detail Score'].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Complementary strengths
    st.markdown("### 🌟 Complementary Strengths")
    
    strengths = rankings_data.get('complementary_strengths', {})
    
    if strengths:
        for model, strength_list in strengths.items():
            with st.expander(f"**{model}**", expanded=True):
                for strength in strength_list:
                    st.markdown(f"✅ {strength}")
    else:
        st.info("No distinct complementary strengths identified")


def display_agreement_analysis(agreement_df: pd.DataFrame) -> None:
    """Display agreement analysis with statistics."""
    st.markdown("### 🤝 Inter-Model Agreement Analysis")
    
    if agreement_df.empty:
        st.warning("No agreement data available")
        return
    
    # Overall statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Mean Agreement",
            f"{agreement_df['agreement'].mean():.1%}",
            help="Average agreement across all model pairs"
        )
    
    with col2:
        st.metric(
            "Std Dev",
            f"{agreement_df['agreement'].std():.3f}",
            help="Variability in agreement scores"
        )
    
    with col3:
        st.metric(
            "Min Agreement",
            f"{agreement_df['agreement'].min():.1%}",
            help="Lowest agreement between any model pair"
        )
    
    # Detailed agreement table
    st.markdown("#### Pairwise Agreement Details")
    
    # Group by model pair and calculate average
    pair_agreement = agreement_df.groupby(['model1', 'model2'])['agreement'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count')
    ]).reset_index()
    
    pair_agreement.columns = ['Model 1', 'Model 2', 'Mean Agreement', 'Std Dev', 'Count']
    pair_agreement['Mean Agreement'] = pair_agreement['Mean Agreement'].apply(lambda x: f"{x:.1%}")
    pair_agreement['Std Dev'] = pair_agreement['Std Dev'].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(pair_agreement, use_container_width=True, hide_index=True)


def display_actionable_insights(insights: List[Dict[str, str]]) -> None:
    """Display actionable insights organized by priority."""
    st.markdown("### 💡 Actionable Insights")
    
    if not insights:
        st.info("No specific insights generated")
        return
    
    # Group by priority
    high_priority = [i for i in insights if i.get('priority') == 'high']
    medium_priority = [i for i in insights if i.get('priority') == 'medium']
    low_priority = [i for i in insights if i.get('priority') == 'low']
    
    # Display high priority first
    if high_priority:
        st.markdown("#### 🔴 High Priority")
        for insight in high_priority:
            with st.container():
                st.markdown(f"**{insight['category']}**")
                st.info(f"**Insight:** {insight['insight']}")
                st.success(f"**Action:** {insight['action']}")
                st.divider()
    
    # Medium priority
    if medium_priority:
        with st.expander("🟡 Medium Priority", expanded=False):
            for insight in medium_priority:
                st.markdown(f"**{insight['category']}**")
                st.info(f"**Insight:** {insight['insight']}")
                st.success(f"**Action:** {insight['action']}")
                st.divider()
    
    # Low priority
    if low_priority:
        with st.expander("🟢 Low Priority", expanded=False):
            for insight in low_priority:
                st.markdown(f"**{insight['category']}**")
                st.info(f"**Insight:** {insight['insight']}")
                st.success(f"**Action:** {insight['action']}")
                st.divider()


def display_prompt_optimization(
    improved_prompts: Dict[str, str],
    original_prompt: str
) -> None:
    """Display prompt optimization suggestions."""
    st.markdown("### ✨ Prompt Optimization Suggestions")
    
    # Show original prompt
    with st.expander("📝 Original Prompt", expanded=False):
        st.code(original_prompt, language="text")
    
    # Show improved prompts for each model
    st.markdown("#### Model-Specific Optimized Prompts")
    
    for model_name, improved_prompt in improved_prompts.items():
        with st.expander(f"**{model_name}**", expanded=False):
            st.code(improved_prompt, language="text")
            
            # Show diff if different
            if improved_prompt != original_prompt:
                st.markdown("**Changes:**")
                additions = improved_prompt.replace(original_prompt, "").strip()
                if additions:
                    st.success(f"Added: {additions}")
            else:
                st.info("No changes recommended - prompt is already optimal")


def display_visualizations(visualizations: Dict[str, Any]) -> None:
    """Display all visualizations."""
    st.markdown("### 📈 Visual Analysis")
    
    # Agreement heatmap
    st.markdown("#### Model Agreement Heatmap")
    st.plotly_chart(
        visualizations['agreement_heatmap'],
        use_container_width=True,
        config=PLOTLY_CONFIG
    )
    
    st.markdown("""
    **Interpretation:**
    - Green cells indicate high agreement between models
    - Red cells indicate low agreement (models disagree)
    - Diagonal is always 1.0 (perfect self-agreement)
    """)
    
    st.divider()
    
    # Confidence correlation
    st.markdown("#### Confidence vs Agreement Correlation")
    st.plotly_chart(
        visualizations['confidence_correlation'],
        use_container_width=True,
        config=PLOTLY_CONFIG
    )
    
    st.markdown("""
    **Interpretation:**
    - Positive correlation: Higher confidence → Higher agreement
    - Negative correlation: Higher confidence → Lower agreement (overconfidence)
    - No correlation: Confidence and agreement are independent
    """)

