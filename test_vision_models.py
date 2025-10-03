"""
Test script for Test 6: Vision Model Discovery and Cost Comparison

This script tests:
1. Vision model discovery from OpenRouter
2. Model caching functionality
3. Cost comparison calculations
4. Recommended model selection
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.vision_model_discovery import (
    fetch_openrouter_vision_models,
    get_recommended_vision_models,
    get_vision_models_by_provider,
    get_vision_model_info,
    get_all_vision_model_ids
)


def test_vision_model_discovery():
    """Test 1: Vision model discovery from OpenRouter"""
    print("=" * 80)
    print("TEST 1: Vision Model Discovery")
    print("=" * 80)
    
    # Fetch all vision models
    all_models = fetch_openrouter_vision_models()
    
    print(f"\n✅ Found {len(all_models)} vision-capable models from OpenRouter")
    
    # Show sample models
    print("\n📋 Sample Models (first 5):")
    for i, (model_id, model_info) in enumerate(list(all_models.items())[:5]):
        pricing = model_info.get("pricing", {})
        print(f"\n{i+1}. {model_info.get('name', model_id)}")
        print(f"   ID: {model_id}")
        print(f"   Provider: {model_info.get('provider', 'unknown')}")
        print(f"   Context: {model_info.get('context_length', 0):,} tokens")
        print(f"   Pricing:")
        print(f"     - Prompt: ${pricing.get('prompt', 0) * 1_000_000:.4f}/1M tokens")
        print(f"     - Completion: ${pricing.get('completion', 0) * 1_000_000:.4f}/1M tokens")
        print(f"     - Image: ${pricing.get('image', 0):.4f} per image")
    
    return all_models


def test_recommended_models(all_models):
    """Test 2: Recommended model selection"""
    print("\n" + "=" * 80)
    print("TEST 2: Recommended Model Selection")
    print("=" * 80)
    
    recommended = get_recommended_vision_models()
    
    print(f"\n✅ Found {len(recommended)} recommended models")
    
    print("\n🎯 Recommended Models by Provider:")
    for provider, model_id in recommended.items():
        model_info = all_models.get(model_id, {})
        pricing = model_info.get("pricing", {})
        
        print(f"\n{provider.upper()}:")
        print(f"  Model: {model_info.get('name', model_id)}")
        print(f"  ID: {model_id}")
        print(f"  Context: {model_info.get('context_length', 0):,} tokens")
        print(f"  Pricing:")
        print(f"    - Prompt: ${pricing.get('prompt', 0) * 1_000_000:.4f}/1M tokens")
        print(f"    - Completion: ${pricing.get('completion', 0) * 1_000_000:.4f}/1M tokens")
        print(f"    - Image: ${pricing.get('image', 0):.4f} per image")
    
    return recommended


def test_cost_comparison(recommended, all_models):
    """Test 3: Cost comparison calculations"""
    print("\n" + "=" * 80)
    print("TEST 3: Cost Comparison")
    print("=" * 80)
    
    print("\n💰 Cost Estimation (per image, assuming 1000 prompt + 500 completion tokens):")
    
    total_cost = 0.0
    costs = []
    
    for provider, model_id in recommended.items():
        model_info = all_models.get(model_id, {})
        pricing = model_info.get("pricing", {})
        
        # Calculate cost per image
        prompt_cost = pricing.get("prompt", 0) * 1000  # 1000 tokens
        completion_cost = pricing.get("completion", 0) * 500  # 500 tokens
        image_cost = pricing.get("image", 0)  # Per image
        
        model_total = prompt_cost + completion_cost + image_cost
        total_cost += model_total
        
        costs.append({
            "provider": provider,
            "model": model_info.get("name", model_id),
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "image_cost": image_cost,
            "total": model_total
        })
    
    # Sort by total cost
    costs.sort(key=lambda x: x["total"])
    
    print("\n📊 Cost Breakdown (sorted by total cost):")
    for cost in costs:
        print(f"\n{cost['provider'].upper()} - {cost['model']}")
        print(f"  Prompt:     ${cost['prompt_cost']:.6f}")
        print(f"  Completion: ${cost['completion_cost']:.6f}")
        print(f"  Image:      ${cost['image_cost']:.6f}")
        print(f"  TOTAL:      ${cost['total']:.6f}")
    
    print(f"\n💵 Total Cost (all 4 models): ${total_cost:.6f} per image")
    print(f"💵 Total Cost (100 images):   ${total_cost * 100:.4f}")
    print(f"💵 Total Cost (1000 images):  ${total_cost * 1000:.2f}")
    
    # Find cheapest and most expensive
    cheapest = costs[0]
    most_expensive = costs[-1]
    
    print(f"\n🏆 Cheapest:  {cheapest['provider'].upper()} - ${cheapest['total']:.6f}/image")
    print(f"💎 Most Expensive: {most_expensive['provider'].upper()} - ${most_expensive['total']:.6f}/image")
    print(f"📈 Price Difference: {(most_expensive['total'] / cheapest['total'] - 1) * 100:.1f}% more expensive")


def test_provider_filtering(all_models):
    """Test 4: Filter models by provider"""
    print("\n" + "=" * 80)
    print("TEST 4: Provider Filtering")
    print("=" * 80)
    
    providers = ["openai", "google", "anthropic", "meta-llama"]
    
    for provider in providers:
        models = get_vision_models_by_provider(provider)
        print(f"\n{provider.upper()}: {len(models)} models")
        
        # Show top 3 cheapest models
        sorted_models = sorted(
            models.items(),
            key=lambda x: x[1].get("pricing", {}).get("prompt", 999)
        )[:3]
        
        print(f"  Top 3 cheapest (by prompt price):")
        for model_id, model_info in sorted_models:
            pricing = model_info.get("pricing", {})
            prompt_price = pricing.get("prompt", 0) * 1_000_000
            print(f"    - {model_info.get('name', model_id)}: ${prompt_price:.4f}/1M tokens")


def test_cache_functionality():
    """Test 5: Cache functionality"""
    print("\n" + "=" * 80)
    print("TEST 5: Cache Functionality")
    print("=" * 80)
    
    cache_file = Path("pricing_cache/openrouter_vision_models.json")
    
    if cache_file.exists():
        print(f"\n✅ Cache file exists: {cache_file}")
        
        # Read cache metadata
        with open(cache_file, 'r') as f:
            data = json.load(f)
        
        cached_at = data.get("cached_at", "unknown")
        num_models = len(data.get("models", {}))
        
        print(f"   Cached at: {cached_at}")
        print(f"   Models cached: {num_models}")
        print(f"   File size: {cache_file.stat().st_size / 1024:.2f} KB")
    else:
        print(f"\n❌ Cache file not found: {cache_file}")


def test_specific_model_info():
    """Test 6: Get specific model info"""
    print("\n" + "=" * 80)
    print("TEST 6: Specific Model Info")
    print("=" * 80)
    
    test_models = [
        "openai/gpt-5-nano",
        "google/gemini-2.5-flash-lite",
        "anthropic/claude-3.5-sonnet",
        "meta-llama/llama-3.2-90b-vision-instruct"
    ]
    
    for model_id in test_models:
        info = get_vision_model_info(model_id)
        
        if info:
            print(f"\n✅ {model_id}")
            print(f"   Name: {info.get('name', 'N/A')}")
            print(f"   Context: {info.get('context_length', 0):,} tokens")
            pricing = info.get("pricing", {})
            print(f"   Pricing: ${pricing.get('prompt', 0) * 1_000_000:.4f}/1M tokens")
        else:
            print(f"\n❌ {model_id} - Not found")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("🧪 TEST 6: VISION MODEL DISCOVERY & COST COMPARISON")
    print("=" * 80)
    
    try:
        # Test 1: Discovery
        all_models = test_vision_model_discovery()
        
        # Test 2: Recommendations
        recommended = test_recommended_models(all_models)
        
        # Test 3: Cost comparison
        test_cost_comparison(recommended, all_models)
        
        # Test 4: Provider filtering
        test_provider_filtering(all_models)
        
        # Test 5: Cache
        test_cache_functionality()
        
        # Test 6: Specific model info
        test_specific_model_info()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

