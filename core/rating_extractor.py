"""
Rating extraction from visual LLM responses.

Parses LLM text responses to extract structured ratings (1-5 scale),
detected artifacts, and confidence scores.
"""

import re
from typing import Dict, List, Optional, Tuple
from core.models import VisualLLMAnalysis


def extract_rating(text: str, rating_name: str) -> Optional[float]:
    """
    Extract a rating (1-5 scale) from text.
    
    Args:
        text: LLM response text
        rating_name: Name of the rating to extract (e.g., "Movement", "Visual Quality")
    
    Returns:
        Rating value (1.0-5.0) or None if not found
    """
    # Patterns to match ratings
    patterns = [
        # "Movement Rating: 4.5" or "Movement: 4.5"
        rf"{rating_name}\s*(?:Rating)?:\s*([0-5](?:\.[0-9]+)?)",
        # "Movement Quality: 4/5" or "Movement: 4/5"
        rf"{rating_name}\s*(?:Quality)?:\s*([0-5])(?:\.[0-9]+)?(?:/5)?",
        # "Movement - 4.5" or "Movement - 4/5"
        rf"{rating_name}\s*-\s*([0-5](?:\.[0-9]+)?)",
        # "[Movement: 4.5]" or "(Movement: 4.5)"
        rf"[\[\(]{rating_name}:\s*([0-5](?:\.[0-9]+)?)[\]\)]",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                rating = float(match.group(1))
                # Clamp to 1-5 range
                return max(1.0, min(5.0, rating))
            except ValueError:
                continue
    
    return None


def extract_artifacts(text: str, known_artifacts: List[str] = None) -> List[str]:
    """
    Extract detected artifacts from text.
    
    Args:
        text: LLM response text
        known_artifacts: List of known artifact types to look for
    
    Returns:
        List of detected artifacts
    """
    if known_artifacts is None:
        known_artifacts = [
            "red lines in eyes",
            "red lines",
            "eye artifacts",
            "finger movement issues",
            "finger issues",
            "finger glitches",
            "feet not moving",
            "static feet",
            "avatar distortions",
            "body distortions",
            "mesh distortions",
            "clothing distortions",
            "texture issues",
            "clipping",
            "z-fighting",
            "polygon artifacts",
            "animation glitches",
            "rigging issues"
        ]
    
    detected = []
    text_lower = text.lower()
    
    # Look for explicit artifact lists
    artifact_section_patterns = [
        r"detected artifacts?:\s*(.+?)(?:\n|$)",
        r"artifacts? (?:found|detected|present):\s*(.+?)(?:\n|$)",
        r"issues? (?:found|detected):\s*(.+?)(?:\n|$)",
    ]
    
    for pattern in artifact_section_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
        if match:
            artifact_text = match.group(1)
            # Split by common delimiters
            items = re.split(r'[,;•\-\n]', artifact_text)
            for item in items:
                item = item.strip()
                if item and len(item) > 3:  # Ignore very short items
                    detected.append(item)
    
    # Also check for known artifacts mentioned anywhere in text
    for artifact in known_artifacts:
        if artifact.lower() in text_lower and artifact not in detected:
            detected.append(artifact)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_detected = []
    for artifact in detected:
        artifact_lower = artifact.lower()
        if artifact_lower not in seen:
            seen.add(artifact_lower)
            unique_detected.append(artifact)
    
    return unique_detected


def extract_confidence(text: str) -> float:
    """
    Extract confidence score from text.

    Args:
        text: LLM response text

    Returns:
        Confidence score (0.0-1.0), defaults to 0.8 if not found
    """
    patterns = [
        # Percentage formats (check these first to avoid partial matches)
        r"confidence\s*(?:score)?:\s*([0-9]{1,3}(?:\.[0-9]+)?)%",
        r"([0-9]{1,3}(?:\.[0-9]+)?)%\s*confidence",
        # Decimal formats (0.XX or 0.X)
        r"confidence\s*(?:score)?:\s*(0\.[0-9]+)",
        r"certainty:\s*(0\.[0-9]+)",
        # Standard formats (without percentage)
        r"confidence:\s*([0-9](?:\.[0-9]+)?)",
        r"certainty:\s*([0-9](?:\.[0-9]+)?)",
        # Rating out of 5 or 10
        r"confidence:\s*([0-9])/5",
        r"confidence:\s*([0-9]{1,2})/10",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))

                # Handle different scales
                if "/" in pattern:
                    # Rating out of 5 or 10
                    if "/5" in pattern:
                        value = value / 5.0
                    elif "/10" in pattern:
                        value = value / 10.0
                elif value > 1.0:
                    # Percentage (convert to 0-1)
                    value = value / 100.0

                # Clamp to 0-1 range
                result = max(0.0, min(1.0, value))

                # If we got 0.0, check if it's explicitly stated or just missing
                # If it's explicitly "0.00%" or "0%", use fallback instead
                if result == 0.0 and ("0.00%" in text or "0%" in text):
                    # This is likely an error or placeholder, use fallback
                    continue

                return result
            except (ValueError, IndexError):
                continue

    # Default confidence based on response quality and content
    # Higher confidence for detailed, structured responses
    if "rating" in text.lower() and "artifact" in text.lower():
        # Structured response with ratings and artifacts
        return 0.85
    elif len(text) > 300:
        return 0.80  # Very detailed response
    elif len(text) > 200:
        return 0.75  # Detailed response
    elif len(text) > 100:
        return 0.70  # Moderate response
    else:
        return 0.60  # Brief response


def parse_visual_llm_response(
    raw_response: str,
    model_name: str,
    known_artifacts: List[str] = None
) -> VisualLLMAnalysis:
    """
    Parse a visual LLM response into structured VisualLLMAnalysis.
    
    Args:
        raw_response: Raw text response from LLM
        model_name: Name of the model that generated the response
        known_artifacts: List of known artifact types to look for
    
    Returns:
        VisualLLMAnalysis object with extracted data
    """
    # Extract ratings
    movement_rating = extract_rating(raw_response, "Movement")
    visual_quality_rating = extract_rating(raw_response, "Visual Quality")
    artifact_presence_rating = extract_rating(raw_response, "Artifact Presence")
    
    # Also try alternative names
    if movement_rating is None:
        movement_rating = extract_rating(raw_response, "Motion")
    if visual_quality_rating is None:
        visual_quality_rating = extract_rating(raw_response, "Quality")
    if artifact_presence_rating is None:
        artifact_presence_rating = extract_rating(raw_response, "Artifacts")
    
    # Extract artifacts
    detected_artifacts = extract_artifacts(raw_response, known_artifacts)
    
    # Extract confidence
    confidence = extract_confidence(raw_response)
    
    # Extract rationale (everything after "Rationale:" or similar)
    rationale_patterns = [
        r"rationale:\s*(.+)",
        r"explanation:\s*(.+)",
        r"analysis:\s*(.+)",
    ]
    
    rationale = raw_response  # Default to full response
    for pattern in rationale_patterns:
        match = re.search(pattern, raw_response, re.IGNORECASE | re.DOTALL)
        if match:
            rationale = match.group(1).strip()
            break
    
    return VisualLLMAnalysis(
        model_name=model_name,
        movement_rating=movement_rating,
        visual_quality_rating=visual_quality_rating,
        artifact_presence_rating=artifact_presence_rating,
        detected_artifacts=detected_artifacts,
        confidence=confidence,
        rationale=rationale,
        raw_response=raw_response
    )


def calculate_agreement_score(analyses: List[VisualLLMAnalysis]) -> float:
    """
    Calculate agreement score across multiple model analyses.
    
    Args:
        analyses: List of VisualLLMAnalysis from different models
    
    Returns:
        Agreement score (0.0-1.0), where 1.0 = perfect agreement
    """
    if len(analyses) < 2:
        return 1.0
    
    scores = []
    
    # Rating agreement (average standard deviation across all ratings)
    rating_types = ['movement_rating', 'visual_quality_rating', 'artifact_presence_rating']
    
    for rating_type in rating_types:
        ratings = [getattr(a, rating_type) for a in analyses if getattr(a, rating_type) is not None]
        if len(ratings) >= 2:
            # Calculate standard deviation
            mean = sum(ratings) / len(ratings)
            variance = sum((r - mean) ** 2 for r in ratings) / len(ratings)
            std_dev = variance ** 0.5
            # Convert to agreement score (lower std_dev = higher agreement)
            # Max std_dev for 1-5 scale is 2.0, so normalize
            agreement = 1.0 - (std_dev / 2.0)
            scores.append(max(0.0, agreement))
    
    # Artifact agreement (Jaccard similarity)
    all_artifacts = set()
    for analysis in analyses:
        all_artifacts.update(a.lower() for a in analysis.detected_artifacts)
    
    if all_artifacts:
        # Calculate pairwise Jaccard similarity
        jaccard_scores = []
        for i in range(len(analyses)):
            for j in range(i + 1, len(analyses)):
                set_i = set(a.lower() for a in analyses[i].detected_artifacts)
                set_j = set(a.lower() for a in analyses[j].detected_artifacts)
                
                if set_i or set_j:
                    intersection = len(set_i & set_j)
                    union = len(set_i | set_j)
                    jaccard = intersection / union if union > 0 else 0.0
                    jaccard_scores.append(jaccard)
        
        if jaccard_scores:
            scores.append(sum(jaccard_scores) / len(jaccard_scores))
    
    # Overall agreement
    return sum(scores) / len(scores) if scores else 0.5


def find_consensus_artifacts(analyses: List[VisualLLMAnalysis], threshold: float = 0.5) -> List[str]:
    """
    Find artifacts detected by multiple models (consensus).
    
    Args:
        analyses: List of VisualLLMAnalysis from different models
        threshold: Minimum fraction of models that must detect an artifact (0.0-1.0)
    
    Returns:
        List of consensus artifacts
    """
    if not analyses:
        return []
    
    # Count artifact occurrences
    artifact_counts = {}
    for analysis in analyses:
        for artifact in analysis.detected_artifacts:
            artifact_lower = artifact.lower()
            if artifact_lower not in artifact_counts:
                artifact_counts[artifact_lower] = {"count": 0, "original": artifact}
            artifact_counts[artifact_lower]["count"] += 1
    
    # Filter by threshold
    min_count = len(analyses) * threshold
    consensus = [
        data["original"]
        for artifact, data in artifact_counts.items()
        if data["count"] >= min_count
    ]
    
    # Sort by frequency
    consensus.sort(key=lambda a: artifact_counts[a.lower()]["count"], reverse=True)
    
    return consensus

