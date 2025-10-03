"""
Pydantic models for data validation and structured outputs.
Extracted from streamlit_test_v5.py to reduce main file size.

This module contains:
- Classification models
- Data generation schemas
- Test summary models
- Validation artifact models
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Classification(BaseModel):
    """Basic classification result with rationale."""
    classification_result: str
    rationale: str


class ClassificationWithConf(Classification):
    """Classification result with confidence score."""
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SyntheticDataItem(BaseModel):
    """Schema for synthetic classification data generation."""
    query: str = Field(description="The user query or input text.")
    classification: str = Field(description="The expected classification label.")


class ToolCallSequenceItem(BaseModel):
    """Schema for tool/agent sequence data generation."""
    query: str = Field(description="The user query requiring a specific tool sequence.")
    expected_sequence: List[str] = Field(description="The sequential list of tool names or actions expected.")


class PruningDataItem(BaseModel):
    """Schema for context pruning data generation."""
    instruction: str = Field(description="System instruction/persona given to the LLM.")
    summary: str = Field(description="Summary of the conversation history so far.")
    user_msgs: str = Field(description="Pipe-separated (||) list of previous user messages.")
    agent_resps: str = Field(description="Pipe-separated (||) list of previous agent responses.")
    tool_logs: str = Field(description="Pipe-separated (||) list of previous tool call logs/results.")
    new_question: str = Field(description="The latest user query.")
    expected_action: str = Field(description="The ground truth next action (general_answer, kb_lookup, tool_call).")
    expected_kept_keys: str = Field(description="Comma-separated list of context keys expected to be kept (e.g., instruction, summary).")


class TestSummaryAndRefinement(BaseModel):
    """Schema for test summary and improvement suggestions."""
    findings_summary: str = Field(description="A concise summary of the test results and model performance.")
    key_suggestions: List[str] = Field(description="Actionable suggestions for prompt, architecture, or tool improvement.")
    suggested_improvement_code: Optional[str] = Field(default=None, description="The refined Python code, agent framework, or prompt string based on suggestions, ready for the next test iteration.")
    suggested_improvement_prompt_reasoning: Optional[str] = Field(default=None, description="Detailed reasoning for the suggested code or prompt improvement.")


class FactualConstraint(BaseModel):
    """Schema for individual policy constraints extracted from validation failures."""
    constraint_id: str = Field(description="Unique identifier for the constraint (e.g., 'NO_PHD_CLAIM')")
    constraint_text: str = Field(description="The explicit rule/fact to enforce")
    priority: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score (0.0-1.0)")
    severity: str = Field(default="MEDIUM", description="Severity level: CRITICAL, HIGH, MEDIUM, LOW")


class ValidationResultArtifact(BaseModel):
    """Schema for ValidatorAgent output with policy update information."""
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall validation confidence (0.0-1.0)")
    final_verdict: str = Field(description="Verification status: 'Verified', 'Hallucination Detected', or 'Inconclusive'")
    policy_updates: List[FactualConstraint] = Field(default_factory=list, description="New constraints to add to memory")
    report_summary: str = Field(description="Cleaned summary for ContentGenerator")
    red_flags: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Detected hallucination risks")


def convert_validation_to_artifact(validation_result: Dict[str, Any]) -> ValidationResultArtifact:
    """
    Convert ValidatorAgent's validation_result to ValidationResultArtifact schema.

    This enables type-safe policy updates while maintaining backward compatibility
    with the existing validation_result format.
    """
    # Map verdict to final_verdict
    verdict = validation_result.get("verdict", "UNKNOWN")
    if verdict == "VERIFIED":
        final_verdict = "Verified"
    elif verdict == "FAILED":
        final_verdict = "Hallucination Detected"
    else:
        final_verdict = "Inconclusive"

    # Convert policy_suggestions to FactualConstraint objects
    policy_updates = []
    for idx, suggestion in enumerate(validation_result.get("policy_suggestions", [])):
        # Extract constraint ID from suggestion text (e.g., "CONSTRAINT: No PhD claims" -> "NO_PHD_CLAIMS")
        constraint_id = f"CONSTRAINT_{idx + 1}"
        if ":" in suggestion:
            # Try to extract meaningful ID from constraint text
            constraint_text = suggestion.split(":", 1)[1].strip()
            # Generate ID from first few words
            words = constraint_text.upper().replace(",", "").replace(".", "").split()[:3]
            constraint_id = "_".join(words)
        else:
            constraint_text = suggestion

        # Determine severity based on hallucination risk
        red_flags = validation_result.get("red_flags", [])
        max_risk = max([rf.get("hallucination_risk", 0.0) for rf in red_flags], default=0.0)

        if max_risk >= 0.8:
            severity = "CRITICAL"
            priority = 0.9
        elif max_risk >= 0.6:
            severity = "HIGH"
            priority = 0.7
        elif max_risk >= 0.4:
            severity = "MEDIUM"
            priority = 0.5
        else:
            severity = "LOW"
            priority = 0.3

        policy_updates.append(FactualConstraint(
            constraint_id=constraint_id,
            constraint_text=constraint_text,
            priority=priority,
            severity=severity
        ))

    return ValidationResultArtifact(
        confidence_score=validation_result.get("confidence_score", 0.5),
        final_verdict=final_verdict,
        policy_updates=policy_updates,
        report_summary=validation_result.get("summary", "No summary available"),
        red_flags=validation_result.get("red_flags", [])
    )


# ============================================================
# Visual LLM Testing Models (Test 6)
# ============================================================

class VisualLLMAnalysis(BaseModel):
    """Schema for visual LLM analysis results."""
    model_name: str = Field(description="Name of the visual LLM model used")
    movement_rating: Optional[float] = Field(default=None, ge=1.0, le=5.0, description="Movement quality rating (1-5)")
    visual_quality_rating: Optional[float] = Field(default=None, ge=1.0, le=5.0, description="Visual quality rating (1-5)")
    artifact_presence_rating: Optional[float] = Field(default=None, ge=1.0, le=5.0, description="Artifact presence rating (1-5, 5=no artifacts)")
    detected_artifacts: List[str] = Field(default_factory=list, description="List of detected artifacts")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in analysis")
    rationale: str = Field(description="Explanation of the analysis")
    raw_response: Optional[str] = Field(default=None, description="Raw LLM response for debugging")


class VRAvatarTestRow(BaseModel):
    """Schema for VR avatar test data (Mode A)."""
    avatar_id: str = Field(description="Unique identifier for the avatar")
    video_path: Optional[str] = Field(default=None, description="Path to video file")
    screenshot_path: Optional[str] = Field(default=None, description="Path to screenshot file")
    human_movement_rating: Optional[float] = Field(default=None, ge=1.0, le=5.0)
    human_visual_rating: Optional[float] = Field(default=None, ge=1.0, le=5.0)
    human_comfort_rating: Optional[float] = Field(default=None, ge=1.0, le=5.0)
    bug_description: Optional[str] = Field(default=None, description="Human-reported bug description")


class VisualLLMComparisonResult(BaseModel):
    """Schema for multi-model visual LLM comparison results."""
    image_id: str = Field(description="Unique identifier for the image")
    image_path: str = Field(description="Path to the image file")
    model_results: Dict[str, VisualLLMAnalysis] = Field(description="Results from each visual LLM model")
    agreement_score: float = Field(ge=0.0, le=1.0, description="Agreement score across models")
    consensus_artifacts: List[str] = Field(default_factory=list, description="Artifacts detected by multiple models")
    timestamp: str = Field(description="Timestamp of analysis")

