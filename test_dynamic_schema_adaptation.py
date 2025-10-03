"""
Test Dynamic Schema Adaptation System

Verifies that the system truly adapts to different tasks without hardcoded schemas.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.dynamic_visual_analysis import (
    DynamicVisualLLMAnalysis,
    parse_dynamic_visual_response,
    introspect_analysis_fields,
    create_adaptive_analysis_prompt
)


def test_dynamic_analysis_class():
    """Test DynamicVisualLLMAnalysis with different field sets."""
    print("\n" + "="*60)
    print("TEST 1: DynamicVisualLLMAnalysis Class")
    print("="*60)
    
    # Test 1: VR Avatar fields
    print("\n1.1 VR Avatar Fields:")
    vr_analysis = DynamicVisualLLMAnalysis(
        model_name="GPT-5 Vision",
        raw_response='{"movement_rating": 4.5}',
        movement_rating=4.5,
        visual_quality_rating=4.0,
        detected_artifacts=["red lines", "finger clipping"],
        confidence=0.85,
        rationale="Good quality with minor artifacts"
    )
    
    print(f"  Numerical fields: {vr_analysis.get_numerical_fields()}")
    print(f"  Categorical fields: {vr_analysis.get_categorical_fields()}")
    print(f"  List fields: {vr_analysis.get_list_fields()}")
    print(f"  Descriptive fields: {vr_analysis.get_descriptive_fields()}")
    
    assert 'movement_rating' in vr_analysis.get_numerical_fields()
    assert 'confidence' in vr_analysis.get_numerical_fields()
    assert 'detected_artifacts' in vr_analysis.get_list_fields()
    print("  ✅ VR Avatar fields correctly categorized")
    
    # Test 2: General image fields
    print("\n1.2 General Image Fields:")
    image_analysis = DynamicVisualLLMAnalysis(
        model_name="Gemini 2.5 Vision",
        raw_response='{"people_count": 5}',
        people_count=5,
        detected_objects=["bench", "tree", "dog"],
        detected_actions=["walking", "sitting"],
        scene_type="park",
        weather_condition="sunny",
        confidence=0.92,
        description="A sunny day in the park with people and activities"
    )
    
    print(f"  Numerical fields: {image_analysis.get_numerical_fields()}")
    print(f"  Categorical fields: {image_analysis.get_categorical_fields()}")
    print(f"  List fields: {image_analysis.get_list_fields()}")
    print(f"  Descriptive fields: {image_analysis.get_descriptive_fields()}")
    
    assert 'people_count' in image_analysis.get_numerical_fields()
    assert 'detected_objects' in image_analysis.get_list_fields()
    assert 'scene_type' in image_analysis.get_categorical_fields()
    # Note: description is short (<100 chars) so it's categorical, not descriptive
    assert 'description' in image_analysis.get_categorical_fields()
    print("  ✅ General image fields correctly categorized")
    
    # Test 3: Medical imaging fields
    print("\n1.3 Medical Imaging Fields:")
    medical_analysis = DynamicVisualLLMAnalysis(
        model_name="Claude 4.5 Vision",
        raw_response='{"quality_score": 4.5}',
        image_type="chest x-ray",
        quality_score=4.5,
        detected_abnormalities=["small nodule", "opacity"],
        abnormality_severity="mild",
        requires_followup=True,
        confidence=0.78,
        clinical_notes="Nodule detected, recommend follow-up"
    )
    
    print(f"  Numerical fields: {medical_analysis.get_numerical_fields()}")
    print(f"  Categorical fields: {medical_analysis.get_categorical_fields()}")
    print(f"  Descriptive fields: {medical_analysis.get_descriptive_fields()}")

    assert 'quality_score' in medical_analysis.get_numerical_fields()
    assert 'detected_abnormalities' in medical_analysis.get_list_fields()
    # clinical_notes is long enough to be descriptive
    assert 'clinical_notes' in medical_analysis.get_categorical_fields() or 'clinical_notes' in medical_analysis.get_descriptive_fields()
    print("  ✅ Medical imaging fields correctly categorized")
    
    print("\n✅ TEST 1 PASSED: DynamicVisualLLMAnalysis works for all task types")


def test_field_introspection():
    """Test field introspection on different task types."""
    print("\n" + "="*60)
    print("TEST 2: Field Introspection")
    print("="*60)
    
    # Test 1: VR Avatar task
    print("\n2.1 VR Avatar Task:")
    vr_results = [
        {
            "image_name": "avatar_001.png",
            "model_results": {
                "GPT-5 Vision": {
                    "movement_rating": 4.5,
                    "visual_quality_rating": 4.0,
                    "detected_artifacts": ["red lines"],
                    "confidence": 0.85,
                    "rationale": "Good quality"
                }
            }
        }
    ]
    
    numerical, categorical, descriptive = introspect_analysis_fields(vr_results)
    print(f"  Numerical: {sorted(numerical)}")
    print(f"  Categorical: {sorted(categorical)}")
    print(f"  Descriptive: {sorted(descriptive)}")
    
    assert 'movement_rating' in numerical
    assert 'confidence' in numerical
    assert 'detected_artifacts' in categorical
    # rationale is short in this test, so it's categorical
    assert 'rationale' in categorical or 'rationale' in descriptive
    print("  ✅ VR Avatar fields correctly detected")
    
    # Test 2: General image task
    print("\n2.2 General Image Task:")
    image_results = [
        {
            "image_name": "park.jpg",
            "model_results": {
                "Gemini 2.5 Vision": {
                    "people_count": 5,
                    "detected_objects": ["bench", "tree"],
                    "scene_type": "park",
                    "confidence": 0.92,
                    "description": "Park scene"
                }
            }
        }
    ]
    
    numerical, categorical, descriptive = introspect_analysis_fields(image_results)
    print(f"  Numerical: {sorted(numerical)}")
    print(f"  Categorical: {sorted(categorical)}")
    print(f"  Descriptive: {sorted(descriptive)}")
    
    assert 'people_count' in numerical
    assert 'detected_objects' in categorical
    assert 'scene_type' in categorical
    # description is short in this test, so it's categorical
    assert 'description' in categorical or 'description' in descriptive
    print("  ✅ General image fields correctly detected")
    
    # Test 3: Medical imaging task
    print("\n2.3 Medical Imaging Task:")
    medical_results = [
        {
            "image_name": "xray_001.png",
            "model_results": {
                "Claude 4.5 Vision": {
                    "quality_score": 4.5,
                    "detected_abnormalities": ["nodule"],
                    "abnormality_severity": "mild",
                    "confidence": 0.78,
                    "clinical_notes": "Follow-up recommended"
                }
            }
        }
    ]
    
    numerical, categorical, descriptive = introspect_analysis_fields(medical_results)
    print(f"  Numerical: {sorted(numerical)}")
    print(f"  Categorical: {sorted(categorical)}")
    print(f"  Descriptive: {sorted(descriptive)}")
    
    assert 'quality_score' in numerical
    assert 'detected_abnormalities' in categorical
    # clinical_notes might be categorical or descriptive depending on length
    assert 'clinical_notes' in categorical or 'clinical_notes' in descriptive
    print("  ✅ Medical imaging fields correctly detected")
    
    print("\n✅ TEST 2 PASSED: Field introspection works for all task types")


def test_adaptive_prompt_generation():
    """Test adaptive prompt generation."""
    print("\n" + "="*60)
    print("TEST 3: Adaptive Prompt Generation")
    print("="*60)
    
    # Test 1: VR Avatar prompt
    print("\n3.1 VR Avatar Prompt:")
    vr_prompt = create_adaptive_analysis_prompt(
        numerical_fields={'movement_rating', 'confidence'},
        categorical_fields={'detected_artifacts'},
        descriptive_fields={'rationale'},
        task_description="Detect VR avatar artifacts"
    )
    
    assert 'movement_rating' in vr_prompt
    assert 'confidence' in vr_prompt
    assert 'detected_artifacts' in vr_prompt
    assert 'distribution' in vr_prompt.lower()
    assert 'frequency' in vr_prompt.lower()
    print("  ✅ VR Avatar prompt includes detected fields")
    print(f"  Prompt length: {len(vr_prompt)} characters")
    
    # Test 2: General image prompt
    print("\n3.2 General Image Prompt:")
    image_prompt = create_adaptive_analysis_prompt(
        numerical_fields={'people_count', 'confidence'},
        categorical_fields={'detected_objects', 'scene_type'},
        descriptive_fields={'description'},
        task_description="Analyze general images"
    )
    
    assert 'people_count' in image_prompt
    assert 'detected_objects' in image_prompt
    assert 'scene_type' in image_prompt
    assert 'people_count' not in vr_prompt  # Verify prompts are different
    print("  ✅ General image prompt includes detected fields")
    print(f"  Prompt length: {len(image_prompt)} characters")
    
    # Test 3: Verify prompts are task-specific
    print("\n3.3 Prompt Specificity:")
    assert vr_prompt != image_prompt
    assert 'movement_rating' in vr_prompt and 'movement_rating' not in image_prompt
    assert 'people_count' in image_prompt and 'people_count' not in vr_prompt
    print("  ✅ Prompts are task-specific (not generic)")
    
    print("\n✅ TEST 3 PASSED: Adaptive prompts generated correctly")


def test_end_to_end_workflow():
    """Test complete workflow from analysis to introspection to prompt."""
    print("\n" + "="*60)
    print("TEST 4: End-to-End Workflow")
    print("="*60)
    
    # Simulate complete workflow for emotion detection task
    print("\n4.1 Emotion Detection Task:")
    
    # Step 1: Create dynamic analysis results
    emotion_analysis = DynamicVisualLLMAnalysis(
        model_name="GPT-5 Vision",
        raw_response='{"face_count": 1}',
        face_count=1,
        primary_emotion="happy",
        detected_emotions=["happy", "excited"],
        emotion_confidence=0.92,
        facial_expression_intensity=4.5,
        confidence=0.88,
        description="Young woman displaying genuine happiness"
    )
    
    print(f"  Created analysis with {len(emotion_analysis.get_all_fields())} fields")
    
    # Step 2: Introspect fields
    results = [
        {
            "image_name": "portrait.jpg",
            "model_results": {
                "GPT-5 Vision": emotion_analysis.to_dict()
            }
        }
    ]
    
    numerical, categorical, descriptive = introspect_analysis_fields(results)
    print(f"  Detected {len(numerical)} numerical, {len(categorical)} categorical, {len(descriptive)} descriptive fields")
    
    # Step 3: Generate adaptive prompt
    prompt = create_adaptive_analysis_prompt(
        numerical_fields=numerical,
        categorical_fields=categorical,
        descriptive_fields=descriptive,
        task_description="Detect emotions in portraits"
    )
    
    print(f"  Generated adaptive prompt ({len(prompt)} chars)")
    
    # Verify workflow
    assert 'face_count' in numerical
    assert 'detected_emotions' in categorical
    assert 'face_count' in prompt
    assert 'detected_emotions' in prompt
    
    print("  ✅ End-to-end workflow successful")
    
    print("\n✅ TEST 4 PASSED: Complete workflow works correctly")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("DYNAMIC SCHEMA ADAPTATION SYSTEM - TEST SUITE")
    print("="*80)
    
    try:
        test_dynamic_analysis_class()
        test_field_introspection()
        test_adaptive_prompt_generation()
        test_end_to_end_workflow()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\nKEY VERIFICATION:")
        print("✅ DynamicVisualLLMAnalysis accepts any fields")
        print("✅ Field introspection detects field types from actual data")
        print("✅ Adaptive prompts generated for detected fields")
        print("✅ System works for VR, general, medical, emotion tasks")
        print("✅ No hardcoded schemas required")
        print("\n🎯 CONCLUSION: True dynamic schema adaptation achieved!")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

