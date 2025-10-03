"""
Dynamic Visual Analysis Examples

Demonstrates how the dynamic schema adaptation system works
across different visual analysis tasks WITHOUT hardcoded schemas.

Examples:
1. VR Avatar Artifact Detection
2. General Image Analysis (people, objects, actions)
3. Medical Imaging Analysis
4. Document/Screenshot Analysis
5. Emotion Detection in Images
"""

import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock visual LLM responses for different tasks
# In production, these would come from actual API calls


# ============================================================
# Example 1: VR Avatar Artifact Detection
# ============================================================

VR_AVATAR_RESPONSES = [
    {
        "image_name": "avatar_001.png",
        "model_results": {
            "GPT-5 Vision": {
                "movement_rating": 4.5,
                "visual_quality_rating": 4.0,
                "artifact_presence_rating": 3.5,
                "detected_artifacts": ["red lines in eyes", "finger clipping"],
                "confidence": 0.85,
                "rationale": "Avatar shows good overall quality with minor artifacts in eye rendering and finger movement."
            },
            "Gemini 2.5 Vision": {
                "movement_rating": 4.2,
                "visual_quality_rating": 4.3,
                "artifact_presence_rating": 3.8,
                "detected_artifacts": ["red lines in eyes"],
                "confidence": 0.82,
                "rationale": "High visual quality, smooth movement, minor red line artifacts detected."
            }
        }
    },
    {
        "image_name": "avatar_002.png",
        "model_results": {
            "GPT-5 Vision": {
                "movement_rating": 3.0,
                "visual_quality_rating": 2.5,
                "artifact_presence_rating": 2.0,
                "detected_artifacts": ["severe distortion", "texture flickering", "feet not touching ground"],
                "confidence": 0.90,
                "rationale": "Multiple severe artifacts detected affecting both visual quality and movement."
            },
            "Gemini 2.5 Vision": {
                "movement_rating": 3.2,
                "visual_quality_rating": 2.8,
                "artifact_presence_rating": 2.2,
                "detected_artifacts": ["distortion", "texture issues"],
                "confidence": 0.88,
                "rationale": "Significant quality issues and artifacts present."
            }
        }
    }
]


# ============================================================
# Example 2: General Image Analysis
# ============================================================

GENERAL_IMAGE_RESPONSES = [
    {
        "image_name": "park_scene.jpg",
        "model_results": {
            "GPT-5 Vision": {
                "people_count": 5,
                "detected_objects": ["bench", "tree", "dog", "bicycle"],
                "detected_actions": ["walking", "sitting", "playing"],
                "scene_type": "outdoor park",
                "weather_condition": "sunny",
                "time_of_day": "afternoon",
                "confidence": 0.92,
                "description": "A sunny afternoon in a park with people engaged in various activities."
            },
            "Gemini 2.5 Vision": {
                "people_count": 6,
                "detected_objects": ["bench", "tree", "dog", "bicycle", "fountain"],
                "detected_actions": ["walking", "sitting", "playing", "jogging"],
                "scene_type": "park",
                "weather_condition": "clear",
                "time_of_day": "midday",
                "confidence": 0.89,
                "description": "Park scene with multiple people and activities on a clear day."
            }
        }
    },
    {
        "image_name": "office_meeting.jpg",
        "model_results": {
            "GPT-5 Vision": {
                "people_count": 8,
                "detected_objects": ["table", "chairs", "laptop", "whiteboard", "projector"],
                "detected_actions": ["presenting", "listening", "taking notes"],
                "scene_type": "office meeting room",
                "formality_level": "business formal",
                "confidence": 0.94,
                "description": "Formal business meeting with presentation in progress."
            },
            "Gemini 2.5 Vision": {
                "people_count": 7,
                "detected_objects": ["conference table", "laptop", "whiteboard", "projector"],
                "detected_actions": ["presenting", "discussing"],
                "scene_type": "meeting room",
                "formality_level": "formal",
                "confidence": 0.91,
                "description": "Professional meeting environment with active discussion."
            }
        }
    }
]


# ============================================================
# Example 3: Medical Imaging Analysis
# ============================================================

MEDICAL_IMAGE_RESPONSES = [
    {
        "image_name": "xray_001.png",
        "model_results": {
            "GPT-5 Vision": {
                "image_type": "chest x-ray",
                "quality_score": 4.5,
                "detected_abnormalities": ["small nodule in right lung", "slight opacity"],
                "abnormality_severity": "mild",
                "requires_followup": True,
                "confidence": 0.78,
                "clinical_notes": "Small nodule detected, recommend follow-up imaging in 3 months."
            },
            "Gemini 2.5 Vision": {
                "image_type": "chest radiograph",
                "quality_score": 4.3,
                "detected_abnormalities": ["nodule right lung"],
                "abnormality_severity": "mild to moderate",
                "requires_followup": True,
                "confidence": 0.75,
                "clinical_notes": "Nodule present, clinical correlation recommended."
            }
        }
    }
]


# ============================================================
# Example 4: Document/Screenshot Analysis
# ============================================================

DOCUMENT_IMAGE_RESPONSES = [
    {
        "image_name": "invoice_001.png",
        "model_results": {
            "GPT-5 Vision": {
                "document_type": "invoice",
                "has_logo": True,
                "has_date": True,
                "has_total": True,
                "detected_fields": ["invoice_number", "date", "total_amount", "items"],
                "completeness_score": 4.8,
                "readability_score": 4.5,
                "confidence": 0.96,
                "summary": "Complete invoice with all required fields clearly visible."
            },
            "Gemini 2.5 Vision": {
                "document_type": "invoice",
                "has_logo": True,
                "has_date": True,
                "has_total": True,
                "detected_fields": ["invoice_number", "date", "total", "line_items", "tax"],
                "completeness_score": 4.9,
                "readability_score": 4.7,
                "confidence": 0.94,
                "summary": "Well-formatted invoice with excellent readability."
            }
        }
    }
]


# ============================================================
# Example 5: Emotion Detection
# ============================================================

EMOTION_DETECTION_RESPONSES = [
    {
        "image_name": "portrait_001.jpg",
        "model_results": {
            "GPT-5 Vision": {
                "face_count": 1,
                "primary_emotion": "happy",
                "emotion_confidence": 0.92,
                "detected_emotions": ["happy", "excited"],
                "age_estimate": "25-30",
                "gender_estimate": "female",
                "facial_expression_intensity": 4.5,
                "confidence": 0.88,
                "description": "Young woman displaying genuine happiness with bright smile."
            },
            "Gemini 2.5 Vision": {
                "face_count": 1,
                "primary_emotion": "joy",
                "emotion_confidence": 0.90,
                "detected_emotions": ["joy", "contentment"],
                "age_estimate": "20-30",
                "gender_estimate": "female",
                "facial_expression_intensity": 4.3,
                "confidence": 0.86,
                "description": "Portrait showing joyful expression."
            }
        }
    }
]


# ============================================================
# Demonstration Functions
# ============================================================

def demonstrate_field_introspection(task_name: str, responses: List[Dict[str, Any]]):
    """
    Demonstrate how field introspection works for a given task.
    """
    from core.dynamic_visual_analysis import introspect_analysis_fields
    
    print(f"\n{'='*60}")
    print(f"TASK: {task_name}")
    print(f"{'='*60}\n")
    
    # Introspect fields
    numerical_fields, categorical_fields, descriptive_fields = introspect_analysis_fields(responses)
    
    print("DETECTED FIELDS:")
    print(f"\n📊 Numerical Fields ({len(numerical_fields)}):")
    for field in sorted(numerical_fields):
        print(f"  - {field}")
    
    print(f"\n📋 Categorical/List Fields ({len(categorical_fields)}):")
    for field in sorted(categorical_fields):
        print(f"  - {field}")
    
    print(f"\n📝 Descriptive Fields ({len(descriptive_fields)}):")
    for field in sorted(descriptive_fields):
        print(f"  - {field}")
    
    print(f"\n✅ RESULT: Analysis will adapt to these {len(numerical_fields) + len(categorical_fields) + len(descriptive_fields)} fields")
    print(f"   - Distribution analysis for {len(numerical_fields)} numerical fields")
    print(f"   - Frequency analysis for {len(categorical_fields)} categorical fields")
    print(f"   - Text analysis for {len(descriptive_fields)} descriptive fields")


def demonstrate_adaptive_prompt(task_name: str, task_description: str, responses: List[Dict[str, Any]]):
    """
    Demonstrate how the analysis prompt adapts to detected fields.
    """
    from core.dynamic_visual_analysis import introspect_analysis_fields, create_adaptive_analysis_prompt
    
    print(f"\n{'='*60}")
    print(f"ADAPTIVE PROMPT FOR: {task_name}")
    print(f"{'='*60}\n")
    
    # Introspect fields
    numerical_fields, categorical_fields, descriptive_fields = introspect_analysis_fields(responses)
    
    # Create adaptive prompt
    prompt = create_adaptive_analysis_prompt(
        numerical_fields=numerical_fields,
        categorical_fields=categorical_fields,
        descriptive_fields=descriptive_fields,
        task_description=task_description
    )
    
    print(prompt)


if __name__ == "__main__":
    # Run all demonstrations
    
    print("\n" + "="*80)
    print("DYNAMIC VISUAL ANALYSIS - FIELD INTROSPECTION DEMONSTRATION")
    print("="*80)
    
    # Example 1: VR Avatar
    demonstrate_field_introspection(
        "VR Avatar Artifact Detection",
        VR_AVATAR_RESPONSES
    )
    
    # Example 2: General Images
    demonstrate_field_introspection(
        "General Image Analysis",
        GENERAL_IMAGE_RESPONSES
    )
    
    # Example 3: Medical Imaging
    demonstrate_field_introspection(
        "Medical Imaging Analysis",
        MEDICAL_IMAGE_RESPONSES
    )
    
    # Example 4: Documents
    demonstrate_field_introspection(
        "Document/Screenshot Analysis",
        DOCUMENT_IMAGE_RESPONSES
    )
    
    # Example 5: Emotions
    demonstrate_field_introspection(
        "Emotion Detection",
        EMOTION_DETECTION_RESPONSES
    )
    
    print("\n" + "="*80)
    print("ADAPTIVE PROMPT GENERATION DEMONSTRATION")
    print("="*80)
    
    # Show adaptive prompt for one example
    demonstrate_adaptive_prompt(
        "General Image Analysis",
        "Analyze images to detect people, objects, and activities",
        GENERAL_IMAGE_RESPONSES
    )
    
    print("\n" + "="*80)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nKEY TAKEAWAY:")
    print("The system automatically detects fields from actual LLM outputs")
    print("and adapts the analysis plan WITHOUT any hardcoded schemas!")

