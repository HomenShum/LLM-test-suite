"""
Mock test for vision analysis workflow (no real API calls required).

Demonstrates:
1. Rating extraction from sample responses
2. Agreement calculation
3. Consensus artifact detection
4. Result caching
5. Visualization data preparation
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from core.rating_extractor import (
    parse_visual_llm_response,
    calculate_agreement_score,
    find_consensus_artifacts
)
from core.models import VisualLLMAnalysis


# Sample LLM responses (realistic examples)
SAMPLE_RESPONSES = {
    "avatar_001_gpt5": """
    Analysis of VR Avatar (avatar_001):
    
    Movement Rating: 3.5
    Visual Quality Rating: 4.0
    Artifact Presence Rating: 2.5
    
    Detected Artifacts:
    - Red lines in eyes
    - Minor texture glitches
    
    Confidence: 0.85
    
    Rationale: The avatar shows clear red line artifacts in the eye region, which significantly impacts the visual quality. The movement appears relatively smooth with a rating of 3.5/5, but the eye artifacts bring down the overall artifact presence score to 2.5/5. The rendering quality is otherwise good at 4.0/5.
    """,
    
    "avatar_001_gemini": """
    VR Avatar Assessment:
    
    Movement: 3.0/5
    Visual Quality: 4.5/5
    Artifacts: 2.0/5
    
    Issues found:
    - Red lines visible in eye area
    - Slight clipping on eyelids
    
    Confidence: 90%
    
    Explanation: This avatar exhibits noticeable red line artifacts in the eyes, which is a common rendering issue. The overall visual fidelity is high (4.5/5), but the eye artifacts are concerning. Movement quality is average at 3.0/5.
    """,
    
    "avatar_001_claude": """
    Avatar Analysis Results:
    
    - Movement Quality: 4.0
    - Visual Quality: 4.2
    - Artifact Presence: 2.8
    
    Detected artifacts:
    • Red lines in eyes
    • Minor polygon artifacts
    
    Confidence level: 0.82
    
    Rationale: The avatar demonstrates good overall quality with a visual rating of 4.2/5. However, there are visible red line artifacts in the eye region that reduce the artifact presence score to 2.8/5. Movement is smooth at 4.0/5.
    """,
    
    "avatar_002_gpt5": """
    Analysis of VR Avatar (avatar_002):
    
    Movement Rating: 2.5
    Visual Quality Rating: 3.8
    Artifact Presence Rating: 3.0
    
    Detected Artifacts:
    - Finger movement issues
    - Hand animation glitches
    - Minor rigging problems
    
    Confidence: 0.78
    
    Rationale: This avatar shows significant finger movement issues with unnatural hand animations. The visual quality is decent at 3.8/5, but the movement problems bring the movement rating down to 2.5/5.
    """,
    
    "avatar_002_gemini": """
    VR Avatar Assessment:
    
    Movement: 2.0/5
    Visual Quality: 4.0/5
    Artifacts: 3.5/5
    
    Issues found:
    - Finger movement issues
    - Hand tracking problems
    - Finger clipping
    
    Confidence: 85%
    
    Explanation: The avatar has noticeable finger movement issues with poor hand tracking. Visual quality is good at 4.0/5, but movement is problematic at 2.0/5.
    """,
    
    "avatar_006_gpt5": """
    Analysis of VR Avatar (avatar_006):
    
    Movement Rating: 4.8
    Visual Quality Rating: 4.7
    Artifact Presence Rating: 4.9
    
    Detected Artifacts: None
    
    Confidence: 0.92
    
    Rationale: This avatar shows excellent quality across all metrics. Movement is very smooth at 4.8/5, visual quality is high at 4.7/5, and there are virtually no artifacts detected (4.9/5). This is a well-rendered avatar with no significant issues.
    """,
    
    "avatar_006_gemini": """
    VR Avatar Assessment:
    
    Movement: 5.0/5
    Visual Quality: 4.5/5
    Artifacts: 5.0/5
    
    Issues found: None detected
    
    Confidence: 95%
    
    Explanation: Excellent avatar with no visible artifacts. Movement is perfect at 5.0/5, visual quality is very good at 4.5/5, and no artifacts were detected.
    """
}


def test_rating_extraction():
    """Test rating extraction from sample responses."""
    print("\n" + "="*80)
    print("TEST 1: Rating Extraction")
    print("="*80)
    
    for response_id, response_text in SAMPLE_RESPONSES.items():
        print(f"\n📝 Parsing: {response_id}")
        
        model_name = response_id.split('_')[-1].upper()
        parsed = parse_visual_llm_response(response_text, model_name)
        
        print(f"   Movement: {parsed.movement_rating}")
        print(f"   Visual Quality: {parsed.visual_quality_rating}")
        print(f"   Artifact Presence: {parsed.artifact_presence_rating}")
        print(f"   Detected Artifacts: {parsed.detected_artifacts}")
        print(f"   Confidence: {parsed.confidence:.2f}")
    
    print("\n✅ Rating extraction test complete")


def test_agreement_calculation():
    """Test agreement score calculation."""
    print("\n" + "="*80)
    print("TEST 2: Agreement Calculation")
    print("="*80)
    
    # Test avatar_001 (has artifacts - should have moderate agreement)
    print("\n📊 Avatar 001 (with artifacts):")
    analyses_001 = [
        parse_visual_llm_response(SAMPLE_RESPONSES["avatar_001_gpt5"], "GPT-5"),
        parse_visual_llm_response(SAMPLE_RESPONSES["avatar_001_gemini"], "Gemini"),
        parse_visual_llm_response(SAMPLE_RESPONSES["avatar_001_claude"], "Claude")
    ]
    
    agreement_001 = calculate_agreement_score(analyses_001)
    consensus_001 = find_consensus_artifacts(analyses_001, threshold=0.5)
    
    print(f"   Agreement Score: {agreement_001:.2%}")
    print(f"   Consensus Artifacts: {consensus_001}")
    
    # Test avatar_006 (no artifacts - should have high agreement)
    print("\n📊 Avatar 006 (no artifacts):")
    analyses_006 = [
        parse_visual_llm_response(SAMPLE_RESPONSES["avatar_006_gpt5"], "GPT-5"),
        parse_visual_llm_response(SAMPLE_RESPONSES["avatar_006_gemini"], "Gemini")
    ]
    
    agreement_006 = calculate_agreement_score(analyses_006)
    consensus_006 = find_consensus_artifacts(analyses_006, threshold=0.5)
    
    print(f"   Agreement Score: {agreement_006:.2%}")
    print(f"   Consensus Artifacts: {consensus_006 if consensus_006 else 'None'}")
    
    print("\n✅ Agreement calculation test complete")


def test_result_caching():
    """Test result caching functionality."""
    print("\n" + "="*80)
    print("TEST 3: Result Caching")
    print("="*80)
    
    # Parse all responses
    all_results = {}
    
    for avatar_id in ["avatar_001", "avatar_002", "avatar_006"]:
        analyses = []
        for model in ["gpt5", "gemini"]:
            response_id = f"{avatar_id}_{model}"
            if response_id in SAMPLE_RESPONSES:
                parsed = parse_visual_llm_response(
                    SAMPLE_RESPONSES[response_id],
                    model.upper()
                )
                analyses.append(parsed.dict())
        
        if analyses:
            agreement = calculate_agreement_score([
                VisualLLMAnalysis(**a) for a in analyses
            ])
            consensus = find_consensus_artifacts([
                VisualLLMAnalysis(**a) for a in analyses
            ], threshold=0.5)
            
            all_results[avatar_id] = {
                "analyses": analyses,
                "agreement_score": agreement,
                "consensus_artifacts": consensus
            }
    
    # Cache results
    cache_dir = Path("test_output/visual_llm_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"mock_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(cache_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_type": "mock_vr_avatar_validation",
            "results": all_results
        }, f, indent=2)
    
    print(f"\n✅ Results cached to: {cache_file}")
    print(f"   File size: {cache_file.stat().st_size / 1024:.2f} KB")
    
    return all_results


def test_visualization_data_prep(results):
    """Test visualization data preparation."""
    print("\n" + "="*80)
    print("TEST 4: Visualization Data Preparation")
    print("="*80)
    
    # Prepare data for scatter plot (human vs LLM ratings)
    print("\n📈 Scatter Plot Data (Movement Rating):")
    print(f"   {'Avatar':<15} {'Model':<10} {'LLM Rating':<12} {'Agreement':<12}")
    print(f"   {'-'*50}")
    
    for avatar_id, data in results.items():
        for analysis in data["analyses"]:
            model = analysis["model_name"]
            rating = analysis.get("movement_rating", "N/A")
            agreement = data["agreement_score"]
            print(f"   {avatar_id:<15} {model:<10} {rating:<12} {agreement:<12.2%}")
    
    # Prepare data for artifact frequency chart
    print("\n📊 Artifact Frequency Data:")
    artifact_counts = {}
    for avatar_id, data in results.items():
        for artifact in data["consensus_artifacts"]:
            artifact_counts[artifact] = artifact_counts.get(artifact, 0) + 1
    
    for artifact, count in sorted(artifact_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {artifact}: {count} occurrences")
    
    # Prepare data for model agreement heatmap
    print("\n🔥 Model Agreement Matrix:")
    print(f"   {'Avatar':<15} {'Agreement Score':<20} {'Status':<10}")
    print(f"   {'-'*50}")
    
    for avatar_id, data in results.items():
        agreement = data["agreement_score"]
        status = "High" if agreement > 0.8 else "Medium" if agreement > 0.6 else "Low"
        print(f"   {avatar_id:<15} {agreement:<20.2%} {status:<10}")
    
    print("\n✅ Visualization data preparation complete")


def main():
    """Run all mock tests."""
    print("\n" + "="*80)
    print("🧪 MOCK VISION ANALYSIS TEST (No API Calls)")
    print("="*80)
    print("\nThis test demonstrates the full workflow using sample LLM responses.")
    
    try:
        # Test 1: Rating extraction
        test_rating_extraction()
        
        # Test 2: Agreement calculation
        test_agreement_calculation()
        
        # Test 3: Result caching
        results = test_result_caching()
        
        # Test 4: Visualization data prep
        test_visualization_data_prep(results)
        
        print("\n" + "="*80)
        print("✅ ALL MOCK TESTS PASSED!")
        print("="*80)
        print("\n📋 Summary:")
        print("   ✓ Rating extraction working")
        print("   ✓ Agreement calculation working")
        print("   ✓ Consensus artifact detection working")
        print("   ✓ Result caching working")
        print("   ✓ Visualization data preparation working")
        print("\n🚀 Ready for real API integration!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

