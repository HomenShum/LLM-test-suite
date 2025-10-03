"""
Test actual vision analysis with real API calls.

This script:
1. Uses sample test images
2. Calls real vision LLM APIs
3. Extracts structured ratings
4. Calculates agreement scores
5. Caches results locally
"""

import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.visual_llm_clients import (
    analyze_image_with_gpt-5-mini,
    analyze_image_with_gemini_vision,
    analyze_image_with_claude_vision,
    build_vr_avatar_analysis_prompt
)
from core.rating_extractor import (
    parse_visual_llm_response,
    calculate_agreement_score,
    find_consensus_artifacts
)
from core.models import VisualLLMAnalysis
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


async def test_single_image_analysis(
    image_path: str,
    prompt: str,
    use_models: list = None
):
    """
    Test vision analysis on a single image with multiple models.
    
    Args:
        image_path: Path to test image
        prompt: Analysis prompt
        use_models: List of models to use (default: all available)
    """
    print(f"\n{'='*80}")
    print(f"Testing Image: {image_path}")
    print(f"{'='*80}")
    
    if use_models is None:
        use_models = ["gpt-5-mini", "gemini", "claude"]
    
    # Get API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    gemini_key = os.getenv('GEMINI_API_KEY')
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    
    analyses = []
    
    # Test GPT-4V
    if "gpt-5-mini" in use_models and openai_key:
        print("\n🤖 Testing GPT-5 Vision...")
        try:
            result = await analyze_image_with_gpt-5-mini(
                image_path=image_path,
                prompt=prompt,
                model=None,  # Use recommended
                openai_api_key=openai_key
            )
            
            # Parse response to extract ratings
            parsed = parse_visual_llm_response(
                result.raw_response,
                result.model_name
            )
            
            print(f"   ✓ Analysis complete")
            print(f"   Movement: {parsed.movement_rating}")
            print(f"   Visual Quality: {parsed.visual_quality_rating}")
            print(f"   Artifact Presence: {parsed.artifact_presence_rating}")
            print(f"   Detected Artifacts: {parsed.detected_artifacts}")
            print(f"   Confidence: {parsed.confidence:.2f}")
            
            analyses.append(parsed)
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    # Test Gemini Vision
    if "gemini" in use_models and gemini_key:
        print("\n🤖 Testing Gemini 2.5 Vision...")
        try:
            result = await analyze_image_with_gemini_vision(
                image_path=image_path,
                prompt=prompt,
                model=None,  # Use recommended
                gemini_api_key=gemini_key
            )
            
            # Parse response to extract ratings
            parsed = parse_visual_llm_response(
                result.raw_response,
                result.model_name
            )
            
            print(f"   ✓ Analysis complete")
            print(f"   Movement: {parsed.movement_rating}")
            print(f"   Visual Quality: {parsed.visual_quality_rating}")
            print(f"   Artifact Presence: {parsed.artifact_presence_rating}")
            print(f"   Detected Artifacts: {parsed.detected_artifacts}")
            print(f"   Confidence: {parsed.confidence:.2f}")
            
            analyses.append(parsed)
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    # Test Claude Vision
    if "claude" in use_models and openrouter_key:
        print("\n🤖 Testing Claude 4.5 Vision...")
        try:
            result = await analyze_image_with_claude_vision(
                image_path=image_path,
                prompt=prompt,
                model=None,  # Use recommended
                openrouter_api_key=openrouter_key
            )
            
            # Parse response to extract ratings
            parsed = parse_visual_llm_response(
                result.raw_response,
                result.model_name
            )
            
            print(f"   ✓ Analysis complete")
            print(f"   Movement: {parsed.movement_rating}")
            print(f"   Visual Quality: {parsed.visual_quality_rating}")
            print(f"   Artifact Presence: {parsed.artifact_presence_rating}")
            print(f"   Detected Artifacts: {parsed.detected_artifacts}")
            print(f"   Confidence: {parsed.confidence:.2f}")
            
            analyses.append(parsed)
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    # Calculate agreement
    if len(analyses) >= 2:
        print(f"\n📊 Multi-Model Analysis:")
        agreement = calculate_agreement_score(analyses)
        consensus = find_consensus_artifacts(analyses, threshold=0.5)
        
        print(f"   Agreement Score: {agreement:.2%}")
        print(f"   Consensus Artifacts: {consensus}")
    
    return analyses


async def test_vr_avatar_dataset():
    """Test VR avatar validation workflow (Mode A)."""
    print("\n" + "="*80)
    print("TEST: VR Avatar Validation (Mode A)")
    print("="*80)
    
    # Sample VR avatar images
    test_images = [
        "test_dataset/visual_llm_images/avatar_001.png",  # Red lines in eyes
        "test_dataset/visual_llm_images/avatar_002.png",  # Finger movement issues
        "test_dataset/visual_llm_images/avatar_006.png",  # No artifacts
    ]
    
    # Build VR avatar analysis prompt
    prompt = build_vr_avatar_analysis_prompt()
    
    all_results = {}
    
    for image_path in test_images:
        if Path(image_path).exists():
            analyses = await test_single_image_analysis(
                image_path=image_path,
                prompt=prompt,
                use_models=["gpt-5-mini", "gemini"]  # Test with 2 models for speed
            )
            all_results[image_path] = [a.dict() for a in analyses]
        else:
            print(f"\n⚠️  Image not found: {image_path}")
    
    # Cache results
    cache_dir = Path("test_output/visual_llm_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"vr_avatar_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(cache_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_type": "vr_avatar_validation",
            "results": all_results
        }, f, indent=2)
    
    print(f"\n✅ Results cached to: {cache_file}")
    
    return all_results


async def test_general_visual_comparison():
    """Test general visual comparison workflow (Mode B)."""
    print("\n" + "="*80)
    print("TEST: General Visual Comparison (Mode B)")
    print("="*80)
    
    # Sample general test images
    test_images = [
        ("test_dataset/visual_llm_images/test_objects.png", "Identify and describe all objects in this image."),
        ("test_dataset/visual_llm_images/test_landscape.png", "Describe the scene and identify key elements."),
    ]
    
    all_results = {}
    
    for image_path, custom_prompt in test_images:
        if Path(image_path).exists():
            analyses = await test_single_image_analysis(
                image_path=image_path,
                prompt=custom_prompt,
                use_models=["gpt-5-mini", "gemini"]  # Test with 2 models for speed
            )
            all_results[image_path] = [a.dict() for a in analyses]
        else:
            print(f"\n⚠️  Image not found: {image_path}")
    
    # Cache results
    cache_dir = Path("test_output/visual_llm_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"general_visual_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(cache_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_type": "general_visual_comparison",
            "results": all_results
        }, f, indent=2)
    
    print(f"\n✅ Results cached to: {cache_file}")
    
    return all_results


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("🧪 ACTUAL VISION ANALYSIS TEST WITH REAL API CALLS")
    print("="*80)
    
    # Check API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    gemini_key = os.getenv('GEMINI_API_KEY')
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    
    print("\n🔑 API Key Status:")
    print(f"   OpenAI: {'✓' if openai_key else '✗'}")
    print(f"   Gemini: {'✓' if gemini_key else '✗'}")
    print(f"   OpenRouter: {'✓' if openrouter_key else '✗'}")
    
    if not (openai_key or gemini_key or openrouter_key):
        print("\n❌ No API keys found! Please set API keys in .env file.")
        return
    
    try:
        # Test Mode A: VR Avatar Validation
        await test_vr_avatar_dataset()
        
        # Test Mode B: General Visual Comparison
        await test_general_visual_comparison()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED!")
        print("="*80)
        print("\nResults have been cached in test_output/visual_llm_cache/")
        print("You can now use these cached results for follow-up analysis without re-running expensive API calls.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

