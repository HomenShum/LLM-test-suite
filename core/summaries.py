"""Summary helpers extracted from the Streamlit app."""

from __future__ import annotations

from typing import Any, Dict, Optional

import asyncio
import streamlit as st
from openai import AsyncOpenAI

from core.models import TestSummaryAndRefinement
from core.api_clients import openai_structured_json, openrouter_json

_CONFIG: Dict[str, Any] = {}

def configure(context: Dict[str, Any]) -> None:
    """Store summary-related configuration."""
    _CONFIG.clear()
    _CONFIG.update(context)

async def get_structured_summary_and_refinement(test_report: str, refinement_code: Optional[str] = None) -> TestSummaryAndRefinement:
    """Generate structured summary and refinement suggestions using the configured providers."""

    system_prompt = (
        "You are an expert AI performance analyst and prompt engineer. "
        "Analyze the provided test report and suggest actionable improvements, "
        "providing the refined Python code or prompt structure if applicable."
    )
    summary_schema = TestSummaryAndRefinement.model_json_schema()

    instruction_text = """
    Analyze the test report and suggest improvements.
    You MUST return JSON with the following required keys:
    1. 'findings_summary' (string): A concise summary of the test results and model performance.
    2. 'key_suggestions' (array of strings): Actionable suggestions for prompt, architecture, or tool improvement.
    3. 'suggested_improvement_code' (optional string): The refined Python code, agent framework, or prompt string.
    4. 'suggested_improvement_prompt_reasoning' (optional string): Detailed reasoning for the refinement.
    Ensure your JSON output strictly adheres to these exact key names.
    """

    payload = {
        "test_report": test_report,
        "refinement_artifact": refinement_code if refinement_code else "N/A",
        "instructions": instruction_text,
    }

    use_openai = _CONFIG.get("use_openai", False)
    openai_api_key = _CONFIG.get("OPENAI_API_KEY")
    openai_model = _CONFIG.get("OPENAI_MODEL")
    use_openrouter = _CONFIG.get("use_ollama", False)
    openrouter_model = _CONFIG.get("OPENROUTER_MODEL")
    openrouter_api_key = _CONFIG.get("OPENROUTER_API_KEY")

    try:
        if use_openai and openai_api_key and openai_model:
            client = AsyncOpenAI(api_key=openai_api_key)
            raw_result = await openai_structured_json(client, openai_model, system_prompt, payload)
        elif use_openrouter and openrouter_api_key and openrouter_model:
            raw_result = await openrouter_json(
                openrouter_model,
                system_prompt,
                payload,
                "summary_refinement",
                summary_schema,
            )
        else:
            return TestSummaryAndRefinement(
                findings_summary="No LLM provider configured for summarization.",
                key_suggestions=["Check API keys."],
                suggested_improvement_code=None,
            )

        return TestSummaryAndRefinement.model_validate(raw_result)

    except Exception as exc:  # pragma: no cover - Streamlit UI feedback
        st.error(f"Error generating structured summary: {exc}")
        return TestSummaryAndRefinement(
            findings_summary=f"Failed to generate structured summary due to error: {exc}",
            key_suggestions=[],
            suggested_improvement_code=None,
        )

async def display_final_summary_for_test(test_name: str, report_text: str, artifact: Optional[str] = None) -> None:
    st.divider()
    st.subheader(f"Final Analysis & Refinement for {test_name}")

    summary_result = await get_structured_summary_and_refinement(report_text, artifact)

    st.markdown(f"**Summary:** {summary_result.findings_summary}")
    st.markdown("**Key Suggestions:**")
    st.json(summary_result.key_suggestions)

    if summary_result.suggested_improvement_code:
        st.subheader("Refined Prompt/Code Suggestion")
        suggestion = summary_result.suggested_improvement_code
        language = "python" if suggestion and ("class " in suggestion or "def " in suggestion) else "markdown"
        st.code(suggestion, language=language)
    if summary_result.suggested_improvement_prompt_reasoning:
        st.subheader("Reasoning for Refinement")
        st.markdown(summary_result.suggested_improvement_prompt_reasoning)
