"""Reporting helpers extracted from the Streamlit app."""

import asyncio
from typing import List, Optional, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from core.api_clients import generate_text_async
from utils.data_helpers import _normalize_label
from utils.plotly_config import PLOTLY_CONFIG


def generate_classification_report(
    y_true: List[str],
    y_pred: List[str],
    model_name: str,
    explain: bool = False
) -> Optional[Dict[str, Dict[str, float]]]:
    """Render a classification report, confusion matrix, and optional LLM explanation."""
    y_true_norm = [_normalize_label(s) for s in y_true]
    y_pred_norm = [_normalize_label(s) for s in y_pred]

    valid_indices = [i for i, (t, p) in enumerate(zip(y_true_norm, y_pred_norm)) if t and p]
    if not valid_indices:
        st.warning(f"No valid predictions for {model_name} to generate a report.")
        return None

    y_true_filtered = [y_true_norm[i] for i in valid_indices]
    y_pred_filtered = [y_pred_norm[i] for i in valid_indices]

    report_dict = classification_report(
        y_true_filtered,
        y_pred_filtered,
        output_dict=True,
        zero_division=0
    )

    st.subheader(f"Classification Report: {model_name}")
    st.dataframe(pd.DataFrame(report_dict).transpose().style.format("{:.2f}"))

    st.subheader(f"Confusion Matrix: {model_name}")
    labels = sorted(list(set(y_true_filtered) | set(y_pred_filtered)))
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=labels)

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 12},
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
        colorbar=dict(title="Count")
    ))
    fig_cm.update_layout(
        title=f"Confusion Matrix: {model_name}",
        xaxis_title="Predicted",
        yaxis_title="True",
        height=max(400, len(labels) * 50),
        width=max(500, len(labels) * 50),
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'}
    )
    st.plotly_chart(fig_cm, use_container_width=True, config=PLOTLY_CONFIG)

    st.subheader(f"Per-Class Performance Radar: {model_name}")
    class_metrics = []
    for label in labels:
        if label in report_dict:
            class_metrics.append({
                'Class': label,
                'Precision': report_dict[label]['precision'],
                'Recall': report_dict[label]['recall'],
                'F1-Score': report_dict[label]['f1-score']
            })

    if class_metrics:
        fig_radar = go.Figure()
        for metric in ['Precision', 'Recall', 'F1-Score']:
            fig_radar.add_trace(go.Scatterpolar(
                r=[m[metric] for m in class_metrics],
                theta=[m['Class'] for m in class_metrics],
                fill='toself',
                name=metric
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig_radar, use_container_width=True, config=PLOTLY_CONFIG)

    if explain:
        with st.spinner(f"Asking an LLM to explain the '{model_name}' confusion matrix..."):
            try:
                cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                cm_string = cm_df.to_string()
                prompt = f"""
As a data science expert, analyze this confusion matrix for the '{model_name}' model.

**Confusion Matrix:**
```
{cm_string}
```

**Instructions:**
1.  **Overall Performance:** Briefly state how well the model is performing in general.
2.  **Strengths:** Identify which classes the model predicts most accurately (high values on the diagonal).
3.  **Weaknesses/Confusions:** Point out the most significant misclassifications (large off-diagonal values). For each, clearly state 'The model confused class A (True) with class B (Predicted) X times.'
4.  **Conclusion:** Provide a concise summary and a potential next step for improving the model based on these confusions.

Keep the explanation clear, concise, and easy to understand. Use markdown for formatting.
"""
                explanation = asyncio.run(generate_text_async(prompt))
                st.markdown(explanation)
            except Exception as exc:  # pragma: no cover - falls back to warning in Streamlit context
                st.warning(f"Could not generate LLM explanation for the confusion matrix: {exc}")

    return report_dict

