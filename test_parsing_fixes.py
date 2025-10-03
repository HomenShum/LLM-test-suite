"""
Test script to verify visual LLM response parsing improvements.

This script tests the parsing logic with various response formats to ensure
all models' outputs are correctly parsed.
"""

import json
from core.visual_llm_clients import _parse_visual_analysis
from core.rating_extractor import parse_visual_llm_response


# Test cases representing different model response formats
TEST_CASES = {
    "gpt5_json_clean": {
        "response": """{
  "movement_rating": 3.5,
  "visual_quality_rating": 4.0,
  "artifact_presence_rating": 2.5,
  "detected_artifacts": ["red lines in eyes", "texture glitches"],
  "confidence": 0.85,
  "rationale": "The product shows clear defects in the eye region."
}""",
        "model": "GPT-5 Vision",
        "expected_confidence": 0.85
    },
    
    "gpt5_json_markdown": {
        "response": """```json
{
  "movement_rating": 4.0,
  "visual_quality_rating": 4.5,
  "artifact_presence_rating": 3.0,
  "detected_artifacts": ["minor scratches"],
  "confidence": 0.90,
  "rationale": "Overall good quality with minor issues."
}
```""",
        "model": "GPT-5 Vision",
        "expected_confidence": 0.90
    },
    
    "gemini_text_format": {
        "response": """Based on the image provided, the product appears to be in excellent condition with no visible defects, scratches, dents, discoloration, or manufacturing flaws.

Product Quality Rating: 5/5

Confidence: 0.00%""",
        "model": "Gemini 2.5 Vision",
        "expected_confidence": 0.75  # Should use fallback
    },
    
    "gemini_structured": {
        "response": """Analysis of the image:

Movement Rating: 3.0/5
Visual Quality: 4.5/5
Artifacts: 2.0/5

Detected Issues:
- Red lines in eyes
- Minor texture problems

Confidence: 85%

The avatar shows some quality issues but is generally acceptable.""",
        "model": "Gemini 2.5 Vision",
        "expected_confidence": 0.85
    },
    
    "llama_detailed": {
        "response": """The product appears to be a high-quality softbox lighting kit with a Neewer brand logo. The image shows a well-designed and manufactured product with no visible defects, scratches, dents, or discoloration. The visual quality is excellent, and there are no noticeable artifacts.

Confidence: 90.00%""",
        "model": "Llama 3.2 Vision",
        "expected_confidence": 0.90
    },
    
    "claude_json_with_text": {
        "response": """Here's my analysis:

{
  "movement_rating": null,
  "visual_quality_rating": 5.0,
  "artifact_presence_rating": 5.0,
  "detected_artifacts": [],
  "confidence": 0.95,
  "rationale": "Excellent product quality with no defects detected."
}

The product is in pristine condition.""",
        "model": "Claude 4.5 Vision",
        "expected_confidence": 0.95
    },
    
    "malformed_json": {
        "response": """Unable to parse structured response.

Confidence: 0.00%""",
        "model": "GPT-5 Vision",
        "expected_confidence": 0.60  # Should use fallback
    }
}


def test_parsing():
    """Test parsing with various response formats."""
    print("="*80)
    print("VISUAL LLM RESPONSE PARSING TEST")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for test_name, test_case in TEST_CASES.items():
        print(f"\n{'='*80}")
        print(f"Test: {test_name}")
        print(f"Model: {test_case['model']}")
        print(f"{'='*80}")
        
        # Parse the response
        try:
            result = _parse_visual_analysis(
                test_case['response'],
                test_case['model']
            )
            
            print(f"\n✅ Parsing successful!")
            print(f"   Model Name: {result.model_name}")
            print(f"   Movement Rating: {result.movement_rating}")
            print(f"   Visual Quality: {result.visual_quality_rating}")
            print(f"   Artifact Rating: {result.artifact_presence_rating}")
            print(f"   Detected Artifacts: {result.detected_artifacts}")
            print(f"   Confidence: {result.confidence:.2%}")
            print(f"   Rationale: {result.rationale[:100]}..." if len(result.rationale) > 100 else f"   Rationale: {result.rationale}")
            
            # Check confidence
            expected_conf = test_case['expected_confidence']
            actual_conf = result.confidence
            
            # Allow 10% tolerance
            if abs(actual_conf - expected_conf) <= 0.10:
                print(f"\n✅ Confidence check PASSED (expected ~{expected_conf:.2%}, got {actual_conf:.2%})")
                passed += 1
            else:
                print(f"\n⚠️ Confidence check FAILED (expected ~{expected_conf:.2%}, got {actual_conf:.2%})")
                failed += 1
                
        except Exception as e:
            print(f"\n❌ Parsing FAILED with error: {e}")
            failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total Tests: {len(TEST_CASES)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/len(TEST_CASES)*100:.1f}%")
    print(f"{'='*80}")
    
    return passed, failed


if __name__ == "__main__":
    test_parsing()

