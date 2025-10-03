"""
Test 6: Visual LLM Model Comparison and Artifact Detection

This module provides two modes:
- Mode A: VR Avatar Validation Workflow
- Mode B: General Visual LLM Comparison Framework
"""

import asyncio
import json
import os
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from core.visual_llm_clients import (
    analyze_image_multi_model,
    build_vr_avatar_analysis_prompt,
    build_general_visual_analysis_prompt,
    get_default_vision_models
)
from core.vision_model_discovery import (
    fetch_openrouter_vision_models,
    get_vision_models_by_provider,
    get_vision_model_info
)
from core.image_collector import (
    search_and_download_images,
    get_cached_images_for_preset,
    get_cache_info,
    clear_preset_cache
)
from core.analysis_history import AnalysisHistoryManager
from core.models import VisualLLMAnalysis, VRAvatarTestRow, VisualLLMComparisonResult
from utils.plotly_config import PLOTLY_CONFIG


# Module-level configuration
_CONFIGURED = False
_CONFIG = {}

def configure(context: Dict[str, Any]) -> None:
    """Configure module with global variables from main app."""
    global _CONFIGURED, _CONFIG
    _CONFIG.clear()
    _CONFIG.update(context)
    _CONFIGURED = True


def render_model_selection_ui() -> Dict[str, Any]:
    """
    Render advanced model selection UI with cost comparison.

    Returns:
        Dict containing:
        - selected_models: List of model identifiers
        - model_configs: Dict mapping identifiers to full model configs
        - total_estimated_cost: Estimated cost per image
    """
    st.subheader("🤖 Visual LLM Model Selection")

    # Get all vision models
    all_models = fetch_openrouter_vision_models()

    # Selection mode
    selection_mode = st.radio(
        "Selection Mode",
        options=["Quick Select (Recommended)", "Advanced (Choose Specific Models)"],
        horizontal=True,
        key="test6_selection_mode"
    )

    selected_models = []
    model_configs = {}

    if selection_mode == "Quick Select (Recommended)":
        # Quick selection with recommended models
        recommended = get_default_vision_models()

        # Display recommended models info
        with st.expander("ℹ️ Recommended Models (Auto-Selected)", expanded=False):
            st.markdown("""
            These models are automatically selected based on:
            - **Cost-effectiveness** (best price/performance ratio)
            - **Availability** (currently active on OpenRouter)
            - **Capabilities** (vision + text input/output)
            """)

            # Create comparison table
            comparison_data = []
            for provider, model_id in recommended.items():
                model_info = all_models.get(model_id, {})
                pricing = model_info.get("pricing", {})
                comparison_data.append({
                    "Provider": provider.title(),
                    "Model": model_info.get("name", model_id),
                    "Context": f"{model_info.get('context_length', 0):,}",
                    "Prompt ($/1M)": f"${pricing.get('prompt', 0) * 1_000_000:.4f}",
                    "Completion ($/1M)": f"${pricing.get('completion', 0) * 1_000_000:.4f}",
                    "Image ($)": f"${pricing.get('image', 0):.4f}"
                })

            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Checkboxes for quick selection
        col1, col2 = st.columns(2)
        with col1:
            gpt_model = recommended.get("openai", "gpt-5-nano")
            gpt_info = all_models.get(gpt_model, {})
            gpt_price = gpt_info.get("pricing", {}).get("prompt", 0) * 1_000_000
            use_gpt = st.checkbox(
                f"GPT-5 Vision ({gpt_model.split('/')[-1]}) - ${gpt_price:.4f}/1M tokens",
                value=True,
                key="test6_gpt5"
            )

            gemini_model = recommended.get("google", "gemini-2.5-flash-lite")
            gemini_info = all_models.get(gemini_model, {})
            gemini_price = gemini_info.get("pricing", {}).get("prompt", 0) * 1_000_000
            use_gemini = st.checkbox(
                f"Gemini 2.5 Vision ({gemini_model.split('/')[-1]}) - ${gemini_price:.4f}/1M tokens",
                value=True,
                key="test6_gemini"
            )

        with col2:
            claude_model = recommended.get("anthropic", "anthropic/claude-sonnet-4.5")
            claude_info = all_models.get(claude_model, {})
            claude_price = claude_info.get("pricing", {}).get("prompt", 0) * 1_000_000
            use_claude = st.checkbox(
                f"Claude 4.5 Vision ({claude_model.split('/')[-1]}) - ${claude_price:.4f}/1M tokens",
                value=True,
                key="test6_claude"
            )

            llama_model = recommended.get("meta-llama", "llama-3.2-90b-vision-instruct")
            llama_info = all_models.get(llama_model, {})
            llama_price = llama_info.get("pricing", {}).get("prompt", 0) * 1_000_000
            use_llama = st.checkbox(
                f"Llama 3.2 Vision ({llama_model.split('/')[-1]}) - ${llama_price:.4f}/1M tokens",
                value=False,
                key="test6_llama"
            )

        # Build selected models
        if use_gpt:
            selected_models.append("gpt5")
            model_configs["gpt5"] = gpt_info
        if use_gemini:
            selected_models.append("gemini")
            model_configs["gemini"] = gemini_info
        if use_claude:
            selected_models.append("claude")
            model_configs["claude"] = claude_info
        if use_llama:
            selected_models.append("llama")
            model_configs["llama"] = llama_info

    else:
        # Advanced selection - show all models by provider
        st.markdown("### 🔍 Advanced Model Selection")

        # Group models by provider
        providers = {}
        for model_id, model_info in all_models.items():
            provider = model_info.get("provider", "unknown")
            if provider not in providers:
                providers[provider] = []
            providers[provider].append((model_id, model_info))

        # Sort providers
        provider_order = ["openai", "google", "anthropic", "meta-llama", "other"]
        sorted_providers = sorted(
            providers.keys(),
            key=lambda p: provider_order.index(p) if p in provider_order else 999
        )

        # Limit to top 5 models per provider for UI performance
        st.info("ℹ️ Showing top 5 most cost-effective models per provider. Use Quick Select for recommended models.")

        for provider in sorted_providers[:4]:  # Show only top 4 providers
            st.markdown(f"#### {provider.title()}")
            models = providers[provider][:5]  # Top 5 models

            # Sort by pricing (cheapest first)
            models.sort(key=lambda x: x[1].get("pricing", {}).get("prompt", 999))

            # Display models in compact format
            for model_id, model_info in models:
                pricing = model_info.get("pricing", {})
                prompt_price = pricing.get("prompt", 0) * 1_000_000

                selected = st.checkbox(
                    f"{model_info.get('name', model_id)} - ${prompt_price:.4f}/1M tokens",
                    key=f"test6_advanced_{model_id}",
                    value=False
                )

                if selected:
                    # Map to standard identifier
                    if provider == "openai" and "gpt5" not in selected_models:
                        selected_models.append("gpt5")
                        model_configs["gpt5"] = model_info
                    elif provider == "google" and "gemini" not in selected_models:
                        selected_models.append("gemini")
                        model_configs["gemini"] = model_info
                    elif provider == "anthropic" and "claude" not in selected_models:
                        selected_models.append("claude")
                        model_configs["claude"] = model_info
                    elif provider == "meta-llama" and "llama" not in selected_models:
                        selected_models.append("llama")
                        model_configs["llama"] = model_info

    # Validation
    if not selected_models:
        st.warning("⚠️ Please select at least one visual LLM model to proceed.")
        return {"selected_models": [], "model_configs": {}, "total_estimated_cost": 0.0}

    # Cost estimation
    st.markdown("---")
    st.markdown("### 💰 Cost Estimation")

    # Estimate cost per image (assuming 1000 tokens prompt + 500 tokens completion)
    total_cost = 0.0
    cost_breakdown = []

    for model_id in selected_models:
        model_info = model_configs.get(model_id, {})
        pricing = model_info.get("pricing", {})

        prompt_cost = pricing.get("prompt", 0) * 1000  # 1000 tokens
        completion_cost = pricing.get("completion", 0) * 500  # 500 tokens
        image_cost = pricing.get("image", 0)  # Per image

        model_total = prompt_cost + completion_cost + image_cost
        total_cost += model_total

        cost_breakdown.append({
            "Model": model_info.get("name", model_id),
            "Prompt": f"${prompt_cost:.6f}",
            "Completion": f"${completion_cost:.6f}",
            "Image": f"${image_cost:.6f}",
            "Total": f"${model_total:.6f}"
        })

    df_cost = pd.DataFrame(cost_breakdown)
    st.dataframe(df_cost, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Selected Models", len(selected_models))
    with col2:
        st.metric("Cost per Image", f"${total_cost:.6f}")
    with col3:
        st.metric("Cost per 100 Images", f"${total_cost * 100:.4f}")

    return {
        "selected_models": selected_models,
        "model_configs": model_configs,
        "total_estimated_cost": total_cost
    }


def render_test6_tab(tab) -> None:
    """Render Test 6: Visual LLM Testing tab."""
    if not _CONFIGURED:
        st.error("Test 6 module not configured. Call configure() first.")
        return

    with tab:
        st.markdown("## 🎨 Test 6: Visual LLM Model Comparison & Artifact Detection")

        st.markdown("""
        This test compares multiple visual LLM models for image analysis tasks.

        **Two Modes Available:**
        - **Mode A**: VR Avatar Validation - Analyze VR avatar recordings for artifacts
        - **Mode B**: General Visual Comparison - Compare visual LLMs on any image dataset
        """)

        # Add main documentation popover
        with st.popover("ℹ️ How Test 6 Works", help="Click to see test orchestration details"):
            st.markdown("**Test 6: Visual LLM Model Comparison & Artifact Detection**")
            st.markdown("This test evaluates multiple visual LLM models on image analysis tasks with two distinct modes.")

            st.markdown("**Overall Orchestration:**")
            st.code("""
# 1. Select visual LLM models
selected_models = ["gpt5", "gemini", "claude", "llama"]

# 2. Choose mode (A or B)
if mode == "Mode A":
    # VR Avatar Validation
    df = load_vr_avatar_csv()
    for avatar in df:
        results = await analyze_image_multi_model(
            image_path=avatar.screenshot_path,
            prompt=build_vr_avatar_analysis_prompt(),
            selected_models=selected_models
        )
        compare_with_human_ratings(results, avatar.human_ratings)

elif mode == "Mode B":
    # General Visual Comparison
    images = collect_images_from_web_search(query)
    for image in images:
        results = await analyze_image_multi_model(
            image_path=image.path,
            prompt=build_general_visual_analysis_prompt(),
            selected_models=selected_models
        )

# 3. Advanced analysis (both modes)
computational_analysis = await plan_computational_analysis(results)
evaluation = await evaluate_visual_llm_performance(results)
            """, language="python")

            st.markdown("**See mode-specific popovers below for detailed workflows.**")

        st.divider()

        # Mode selector
        mode = st.radio(
            "Select Test Mode",
            options=["Mode A: VR Avatar Validation", "Mode B: General Visual Comparison"],
            horizontal=True,
            key="test6_mode"
        )

        st.divider()

        # Debug mode toggle
        with st.expander("⚙️ Advanced Settings", expanded=False):
            debug_mode = st.checkbox(
                "Enable Debug Mode (show parsing details)",
                value=False,
                key="test6_debug_mode",
                help="Shows detailed information about response parsing and any errors"
            )
            st.session_state['debug_mode'] = debug_mode

        # Model selection with cost comparison (common to both modes)
        model_selection = render_model_selection_ui()
        selected_models = model_selection["selected_models"]
        model_configs = model_selection["model_configs"]
        total_estimated_cost = model_selection["total_estimated_cost"]

        if not selected_models:
            return

        st.divider()

        # Render mode-specific UI
        if "Mode A" in mode:
            render_mode_a_vr_avatar(selected_models)
        else:
            render_mode_b_general_visual(selected_models)


def render_mode_a_vr_avatar(selected_models: List[str]) -> None:
    """Render Mode A: VR Avatar Validation workflow."""
    st.subheader("📊 Mode A: VR Avatar Validation Workflow")

    st.markdown("""
    **Workflow:**
    1. Upload CSV file with VR avatar test data
    2. Visual LLMs analyze videos/screenshots for artifacts
    3. Compare LLM ratings with human ratings
    4. Generate visualizations and bug prioritization report
    """)

    # Add Mode A specific documentation popover
    with st.popover("📖 Mode A: Detailed Orchestration", help="Click for Mode A workflow details"):
        st.markdown("**Mode A: VR Avatar Validation Workflow**")
        st.markdown("Compares visual LLM artifact detection with human ratings for VR avatars.")

        st.markdown("**Step-by-Step Orchestration:**")
        st.code("""
# 1. Load VR avatar test data from CSV
df = pd.read_csv(uploaded_file)
# Expected columns:
# - avatar_id, video_path, screenshot_path
# - human_movement_rating, human_visual_rating, human_comfort_rating
# - bug_description

# 2. Configure artifact detection
artifact_types = [
    "red lines in eyes",
    "finger movement issues",
    "feet not moving",
    "avatar distortions",
    "clothing distortions during movement"
]

# 3. Build specialized prompt
prompt = build_vr_avatar_analysis_prompt(artifact_types)

# 4. Analyze each avatar with all selected models
for avatar in df:
    model_results = await analyze_image_multi_model(
        image_path=avatar.screenshot_path,
        prompt=prompt,
        selected_models=["gpt5", "gemini", "claude"]
    )

    # Each model returns:
    # - detected_artifacts: List[str]
    # - severity_ratings: Dict[str, float]
    # - confidence_score: float
    # - rationale: str

# 5. Compare with human ratings
for result in results:
    correlation = compare_ratings(
        llm_ratings=result.severity_ratings,
        human_ratings=avatar.human_ratings
    )

# 6. Generate visualizations
- Rating comparison scatter plots
- Model agreement heatmaps
- Bug prioritization report
        """, language="python")

        st.markdown("**Key Functions:**")
        st.code("""
# Specialized VR avatar prompt builder
def build_vr_avatar_analysis_prompt(artifact_types):
    return f'''
    Analyze this VR avatar recording for the following artifacts:
    {', '.join(artifact_types)}

    For each artifact:
    1. Detect presence (yes/no)
    2. Rate severity (0-5 scale)
    3. Provide confidence score
    4. Explain your reasoning
    '''

# Multi-model analysis
async def analyze_image_multi_model(
    image_path, prompt, selected_models
):
    tasks = [
        analyze_with_model(image_path, prompt, model)
        for model in selected_models
    ]
    return await asyncio.gather(*tasks)
        """, language="python")

        st.markdown("**Expected CSV Format:**")
        st.code("""
avatar_id,screenshot_path,human_movement_rating,human_visual_rating,human_comfort_rating,bug_description
avatar_001,/path/img1.png,4.5,4.0,4.2,"Minor finger clipping"
avatar_002,/path/img2.png,3.0,2.5,3.5,"Red lines in eyes, feet not moving"
        """, language="text")

        st.markdown("---")
        st.markdown("**Example Input:**")
        st.code("""
# CSV Row:
avatar_id: "avatar_003"
screenshot_path: "screenshots/avatar_003_walk.png"
human_movement_rating: 3.5
human_visual_rating: 2.0
human_comfort_rating: 3.0
bug_description: "Eyes have red lines, arms clip through body during walk"
        """, language="text")

        st.markdown("**Example Output:**")
        st.code("""
# Multi-model analysis results:
{
  "avatar_id": "avatar_003",
  "image_path": "screenshots/avatar_003_walk.png",
  "human_ratings": {
    "movement": 3.5,
    "visual": 2.0,
    "comfort": 3.0
  },
  "model_results": {
    "gpt-5-nano": {
      "detected_artifacts": [
        "Red lines visible in eye area",
        "Arm geometry clips through torso during animation"
      ],
      "movement_rating": 3.0,
      "visual_rating": 2.5,
      "comfort_rating": 3.0,
      "confidence": 0.85,
      "rationale": "Clear visual artifacts in eyes reduce visual
        quality. Arm clipping is noticeable but doesn't severely
        impact movement fluidity."
    },
    "gemini-2.5-flash": {
      "detected_artifacts": [
        "Eye rendering artifacts (red lines)",
        "Mesh penetration in upper body"
      ],
      "movement_rating": 3.5,
      "visual_rating": 2.0,
      "comfort_rating": 2.5,
      "confidence": 0.92,
      "rationale": "Eye artifacts are prominent and distracting.
        Arm clipping may cause discomfort in VR."
    }
  }
}
        """, language="json")

        st.markdown("**Calculation Steps:**")
        st.markdown("**Step 1: Artifact Detection Agreement**")
        st.code("""
# Compare detected artifacts across models:
GPT-5-nano artifacts:
  - "Red lines visible in eye area"
  - "Arm geometry clips through torso during animation"

Gemini-2.5-flash artifacts:
  - "Eye rendering artifacts (red lines)"
  - "Mesh penetration in upper body"

Semantic matching:
  "Red lines in eye area" ≈ "Eye rendering artifacts" → MATCH
  "Arm clips through torso" ≈ "Mesh penetration" → MATCH

Agreement Score: 2/2 = 100% (both models detected same issues)
        """, language="text")

        st.markdown("**Step 2: Rating Correlation**")
        st.code("""
# Compare LLM ratings with human ratings:

Movement Rating:
  Human:  3.5
  GPT-5:  3.0  (diff: -0.5)
  Gemini: 3.5  (diff:  0.0) ✓ Perfect match

Visual Rating:
  Human:  2.0
  GPT-5:  2.5  (diff: +0.5)
  Gemini: 2.0  (diff:  0.0) ✓ Perfect match

Comfort Rating:
  Human:  3.0
  GPT-5:  3.0  (diff:  0.0) ✓ Perfect match
  Gemini: 2.5  (diff: -0.5)

# Calculate correlation (Pearson's r):
Human ratings:  [3.5, 2.0, 3.0]
GPT-5 ratings:  [3.0, 2.5, 3.0]
Gemini ratings: [3.5, 2.0, 2.5]

Correlation (GPT-5 vs Human):   r = 0.87
Correlation (Gemini vs Human):  r = 0.95  ← Better correlation
        """, language="text")

        st.markdown("**Step 3: Bug Prioritization**")
        st.code("""
# Prioritize bugs by severity and model confidence:

Bug 1: "Red lines in eyes"
  - Detected by: 2/2 models (100% agreement)
  - Avg confidence: (0.85 + 0.92) / 2 = 0.885
  - Impact on visual rating: 2.0 (LOW)
  - Priority Score: 0.885 × (5.0 - 2.0) = 2.66  ← HIGH PRIORITY

Bug 2: "Arm clipping through body"
  - Detected by: 2/2 models (100% agreement)
  - Avg confidence: (0.85 + 0.92) / 2 = 0.885
  - Impact on movement rating: 3.5 (MEDIUM)
  - Priority Score: 0.885 × (5.0 - 3.5) = 1.33  ← MEDIUM PRIORITY

Prioritized Bug List:
  1. Red lines in eyes (Priority: 2.66)
  2. Arm clipping (Priority: 1.33)
        """, language="text")

        st.markdown("---")
        st.markdown("**Expected Outputs:**")
        st.markdown("- LLM vs. Human rating correlation analysis")
        st.markdown("- Model agreement on artifact detection")
        st.markdown("- Bug prioritization based on severity and confidence")
        st.markdown("- Downloadable results with all ratings")

    # CSV upload
    st.markdown("### 📁 Upload Test Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file with VR avatar test results",
        type=["csv"],
        key="test6_mode_a_csv",
        help="CSV should contain: avatar_id, video_path, screenshot_path, human_movement_rating, human_visual_rating, human_comfort_rating, bug_description"
    )

    if uploaded_file is not None:
        # Load CSV
        df = pd.read_csv(uploaded_file)

        st.success(f"✅ Loaded {len(df)} avatar test records")

        # Preview data
        with st.expander("📋 Preview Test Data", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)

        # Artifact type configuration
        st.markdown("### 🔍 Artifact Detection Configuration")

        default_artifacts = [
            "red lines in eyes",
            "finger movement issues",
            "feet not moving",
            "avatar distortions",
            "clothing distortions during movement"
        ]

        artifact_types = st.multiselect(
            "Select artifact types to detect",
            options=default_artifacts + ["custom"],
            default=default_artifacts,
            key="test6_artifact_types"
        )

        if "custom" in artifact_types:
            custom_artifact = st.text_input(
                "Enter custom artifact type",
                key="test6_custom_artifact"
            )
            if custom_artifact:
                artifact_types.remove("custom")
                artifact_types.append(custom_artifact)

        # Run analysis button
        st.divider()

        if st.button("▶️ Run VR Avatar Analysis", type="primary", use_container_width=True):
            run_mode_a_analysis(df, selected_models, artifact_types)
    else:
        st.info("👆 Upload a CSV file to begin VR avatar validation")

        # Show example CSV format
        with st.expander("📄 Example CSV Format", expanded=False):
            example_df = pd.DataFrame({
                'avatar_id': ['avatar_001', 'avatar_002', 'avatar_003'],
                'video_path': ['/path/to/video1.mp4', '/path/to/video2.mp4', '/path/to/video3.mp4'],
                'screenshot_path': ['/path/to/img1.png', '/path/to/img2.png', '/path/to/img3.png'],
                'human_movement_rating': [4.5, 3.0, 5.0],
                'human_visual_rating': [4.0, 2.5, 4.5],
                'human_comfort_rating': [4.2, 3.5, 4.8],
                'bug_description': ['Minor finger glitch', 'Red lines in eyes, feet static', 'No issues']
            })
            st.dataframe(example_df, use_container_width=True)


def run_preset_analysis(
    test_images: List[str],
    selected_models: List[str],
    task_description: str,
    preset_name: str,
    preset_config: Optional[Dict[str, Any]] = None
) -> None:
    """Run preset analysis on test images with selected models."""

    # Get API keys from config
    openai_key = _CONFIG.get('OPENAI_API_KEY')
    gemini_key = _CONFIG.get('GEMINI_API_KEY')
    openrouter_key = _CONFIG.get('OPENROUTER_API_KEY')

    # Check API keys
    missing_keys = []
    if "gpt5" in selected_models and not openai_key:
        missing_keys.append("OPENAI_API_KEY")
    if "gemini" in selected_models and not gemini_key:
        missing_keys.append("GEMINI_API_KEY")
    if ("claude" in selected_models or "llama" in selected_models) and not openrouter_key:
        missing_keys.append("OPENROUTER_API_KEY")

    if missing_keys:
        st.error(f"❌ Missing API keys: {', '.join(missing_keys)}")
        st.info("💡 Please add API keys in the sidebar to run analysis.")
        return

    # === IMMEDIATE DISPLAY: Show images being used ===
    st.markdown("### 📸 Images Selected for Analysis")

    with st.expander(f"🖼️ View {len(test_images)} Selected Images", expanded=True):
        # Display images in a grid
        cols_per_row = 3
        for i in range(0, len(test_images), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(test_images):
                    with col:
                        try:
                            from PIL import Image
                            img = Image.open(test_images[idx])
                            st.image(img, caption=f"Image {idx + 1}: {os.path.basename(test_images[idx])}", use_container_width=True)
                        except Exception as e:
                            st.error(f"Error loading image {idx + 1}: {str(e)}")

    st.markdown("---")

    # Create progress tracking
    st.markdown("### 🔄 Running Visual LLM Analysis...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Create a container for progressive results display
    results_container = st.container()

    # Initialize results storage
    all_results = []

    # Analyze each image
    total_images = len(test_images)

    async def analyze_all_images():
        results = []
        image_metadata = []
        import time

        def _dominant_colors(image_path: str, max_colors: int = 3) -> List[str]:
            try:
                from PIL import Image
                img = Image.open(image_path).convert("RGB")
                img = img.resize((64, 64))
                colors = img.getcolors(maxcolors=64*64) or []  # list of (count, (r,g,b))
                colors.sort(reverse=True, key=lambda x: x[0])
                top = colors[:max_colors]
                def to_hex(rgb):
                    r, g, b = rgb
                    return f"#{r:02x}{g:02x}{b:02x}"
                return [to_hex(rgb) for _cnt, rgb in top]
            except Exception:
                return []


        def _compute_metadata(model_results: Dict[str, Any], image_name: str) -> Dict[str, Any]:
            # Aggregate structured metadata for later QA relevance
            rationales: List[str] = []
            confidences: List[float] = []
            has_artifacts = False
            try:
                for analysis in model_results.values():
                    text = None
                    if hasattr(analysis, 'rationale') and analysis.rationale:
                        text = str(analysis.rationale)
                    elif isinstance(analysis, dict) and analysis.get('rationale'):
                        text = str(analysis.get('rationale'))
                    if text:
                        rationales.append(text)
                        if 'artifact' in text.lower():
                            has_artifacts = True
                    if hasattr(analysis, 'confidence') and isinstance(analysis.confidence, (int, float)):
                        confidences.append(float(analysis.confidence))
                avg_conf = sum(confidences) / len(confidences) if confidences else None
            except Exception:
                avg_conf = None

            # Build descriptor from first rationale
            descriptor = (rationales[0].strip().replace("\n", " ")[:240]) if rationales else image_name

            # Derive keywords, OCR hints, categories, time-of-day
            from collections import Counter
            words: List[str] = []
            for r in rationales:
                for w in r.lower().split():
                    w = ''.join(ch for ch in w if ch.isalpha())
                    if len(w) > 3:
                        words.append(w)
            top_keywords = [w for w, _ in Counter(words).most_common(8)] if words else []

            # OCR presence heuristic
            ocr_terms = {"text", "label", "caption", "subtitle", "sign", "words", "number", "digits"}
            ocr_text_present = any(any(term in r.lower() for term in ocr_terms) for r in rationales)

            # Category labels via simple keyword mapping
            category_map = {
                "urban": {"street", "city", "urban", "building", "traffic"},
                "nature": {"forest", "tree", "mountain", "river", "nature", "landscape"},
                "portrait": {"face", "person", "portrait", "selfie", "human"},
                "indoor": {"indoor", "room", "kitchen", "office", "bedroom"},
                "outdoor": {"outdoor", "park", "street", "beach"},
                "medical": {"xray", "ct", "mri", "medical", "scan"},
                "product": {"product", "packaging", "bottle", "box", "label"},
                "chart": {"chart", "graph", "plot", "diagram"},
                "art": {"painting", "art", "illustration", "drawing"},
            }
            categories_detected: List[str] = []
            rationale_lower = " ".join(rationales).lower()
            for cat, terms in category_map.items():
                if any(t in rationale_lower for t in terms):
                    categories_detected.append(cat)

            # Time-of-day hint
            tod = None
            if any(t in rationale_lower for t in ["night", "evening", "dusk", "dark"]):
                tod = "night"
            elif any(t in rationale_lower for t in ["sunset", "golden hour", "twilight"]):
                tod = "sunset"
            elif any(t in rationale_lower for t in ["sunrise", "dawn", "morning"]):
                tod = "morning"
            elif any(t in rationale_lower for t in ["noon", "midday"]):
                tod = "noon"
            elif any(t in rationale_lower for t in ["day", "daytime", "bright"]):
                tod = "day"

            return {
                "descriptor": descriptor,
                "avg_confidence": avg_conf,
                "has_artifacts": has_artifacts,
                "keywords": top_keywords,
                "models_present": list(model_results.keys()),
                "ocr_text_present": ocr_text_present,
                "category_labels": categories_detected,
                "time_of_day_hint": tod,
            }

        for idx, image_path in enumerate(test_images):
            status_text.text(f"🔍 Analyzing image {idx + 1}/{total_images}: {os.path.basename(image_path)}")

            try:
                # Track cost/latency per image
                prev_len = None
                try:
                    prev_len = len(st.session_state.cost_tracker.by_call)  # type: ignore[attr-defined]
                except Exception:
                    prev_len = None
                start_ts = time.perf_counter()

                # Run multi-model analysis
                model_results = await analyze_image_multi_model(
                    image_path=image_path,
                    prompt=task_description,
                    selected_models=selected_models,
                    openai_api_key=openai_key,
                    gemini_api_key=gemini_key,
                    openrouter_api_key=openrouter_key
                )

                latency_ms = (time.perf_counter() - start_ts) * 1000.0

                # Aggregate cost/tokens from new calls
                total_cost = 0.0
                total_tokens = 0
                per_model_costs: List[Dict[str, Any]] = []
                if prev_len is not None:
                    try:
                        calls = st.session_state.cost_tracker.by_call[prev_len:]  # type: ignore[index]
                        per_model: Dict[str, Dict[str, Any]] = {}
                        for c in calls:
                            model = c.get('model') or c.get('model_name') or 'unknown'
                            cost = float(c.get('cost_usd', 0) or 0.0)
                            tokens = int(c.get('total_tokens', 0) or 0)
                            if model not in per_model:
                                per_model[model] = {"model": model, "cost_usd": 0.0, "total_tokens": 0}
                            per_model[model]["cost_usd"] += cost
                            per_model[model]["total_tokens"] += tokens
                        per_model_costs = list(per_model.values())
                        total_cost = sum(x["cost_usd"] for x in per_model_costs)
                        total_tokens = sum(x["total_tokens"] for x in per_model_costs)
                    except Exception:
                        pass

                image_name = os.path.basename(image_path)
                # Compute dominant colors
                dominant = _dominant_colors(image_path)
                # Base metadata
                metadata = _compute_metadata(model_results, image_name)
                # Enrich with perf/cost/color info
                metadata.update({
                    "dominant_colors": dominant,
                    "latency_ms": latency_ms,
                    "total_cost_usd": total_cost,
                    "total_tokens": total_tokens,
                    "per_model_costs": per_model_costs,
                })

                result = {
                    "image_path": image_path,
                    "image_name": image_name,
                    "model_results": model_results,
                    "metadata": metadata,
                }
                results.append(result)
                image_metadata.append({"image_name": image_name, **metadata})

                # === PROGRESSIVE DISPLAY: Show result immediately ===
                with results_container:
                    with st.expander(f"✅ Image {idx + 1}/{total_images}: {image_name}", expanded=(idx == 0)):
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            try:
                                from PIL import Image
                                img = Image.open(image_path)
                                st.image(img, use_container_width=True)
                            except:
                                st.info("Image preview unavailable")

                        with col2:
                            st.markdown("**Model Responses:**")
                            for model_name, analysis in model_results.items():
                                if hasattr(analysis, 'confidence') and hasattr(analysis, 'rationale'):
                                    st.markdown(f"**{model_name}:** {analysis.rationale[:100]}... (Confidence: {analysis.confidence:.0%})")
                                elif hasattr(analysis, 'confidence'):
                                    st.markdown(f"**{model_name}:** (Confidence: {analysis.confidence:.0%})")
                                else:
                                    st.markdown(f"**{model_name}:** Analysis complete")

            except Exception as e:
                st.warning(f"⚠️ Error analyzing {os.path.basename(image_path)}: {str(e)}")

            # Update progress
            progress_bar.progress((idx + 1) / total_images)

        return results, image_metadata

    # Run async analysis
    all_results, image_metadata = asyncio.run(analyze_all_images())
    # Store structured metadata for later QA/relevancy
    st.session_state.test6_image_metadata = image_metadata

    status_text.text("✅ All images analyzed!")
    progress_bar.progress(1.0)

    # Brief pause to show completion
    import time
    time.sleep(1)

    status_text.empty()
    progress_bar.empty()

    if not all_results:
        st.error("❌ No results generated. Please check API keys and try again.")
        return

    # Save results to session state
    st.session_state.test6_analysis_results = all_results
    st.session_state.test6_selected_models = selected_models
    st.session_state.test6_preset_name = preset_name
    st.session_state.test6_task_description = task_description

    # Display results
    st.success(f"✅ Analysis complete! Analyzed {len(all_results)} images with {len(selected_models)} models.")

    # Show unified results display
    from ui.test6_advanced_results import display_advanced_results

    # Get task description from preset_config or use default
    task_desc = task_description if task_description else "Analyze images"
    if preset_config:
        task_desc = preset_config.get('task_description', task_desc)

    # Save analysis to history
    history_manager = AnalysisHistoryManager()

    analysis_id = history_manager.save_analysis(
        results=all_results,
        preset_name=preset_name,
        task_description=task_desc,
        selected_models=selected_models,
        ground_truths=st.session_state.get('test6_ground_truths'),
        curation_report=st.session_state.get('test6_curation_report'),
        computational_results=st.session_state.get('test6_computational_results'),
        evaluation_results=st.session_state.get('test6_evaluation_results'),
        qa_history=st.session_state.get('test6_qa_history')
    )

    st.success(f"💾 Analysis saved to history (ID: {analysis_id})")

    display_advanced_results(
        results=all_results,
        selected_models=selected_models,
        preset_name=preset_name,
        task_description=task_desc,
        _CONFIG=_CONFIG
    )


def display_preset_results(
    results: List[Dict[str, Any]],
    selected_models: List[str],
    preset_name: str
) -> None:
    """
    DEPRECATED: Use display_advanced_results instead.
    This function is kept for backward compatibility but redirects to the unified display.
    """
    from ui.test6_advanced_results import display_advanced_results

    # Redirect to unified display
    display_advanced_results(
        results=results,
        selected_models=selected_models,
        preset_name=preset_name,
        task_description="Analyze images",
        _CONFIG=_CONFIG
    )
    return

    # OLD CODE BELOW - NOT EXECUTED
    """Display preset analysis results with visualizations."""

    st.markdown("### 📊 Analysis Results")

    # Create tabs for different views
    result_tabs = st.tabs(["📋 Summary", "🖼️ Image Results", "📈 Visualizations", "💾 Export"])

    # Tab 1: Summary
    with result_tabs[0]:
        st.markdown("#### 📊 Analysis Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Images Analyzed", len(results))

        with col2:
            st.metric("Models Used", len(selected_models))

        with col3:
            total_analyses = len(results) * len(selected_models)
            st.metric("Total Analyses", total_analyses)

        # Model comparison table
        st.markdown("#### 🤖 Model Performance")

        model_data = []

        # Extract actual model names from results
        actual_model_names = {}
        if results:
            first_result = results[0]
            model_results = first_result.get("model_results", {})
            for model_name in model_results.keys():
                # Map model_id to actual model name
                for model_id in selected_models:
                    if model_id in model_name.lower() or model_id.replace("gpt5", "gpt") in model_name.lower():
                        actual_model_names[model_id] = model_name
                        break

        for model_id in selected_models:
            success_count = 0
            actual_name = actual_model_names.get(model_id, model_id.upper())

            for result in results:
                model_results = result.get("model_results", {})
                for model_name, analysis in model_results.items():
                    if model_id in model_name.lower() or model_id.replace("gpt5", "gpt") in model_name.lower():
                        success_count += 1

            # Get cost data from session state if available
            total_cost = 0.0
            total_tokens = 0
            if 'cost_tracker' in st.session_state:
                cost_tracker = st.session_state.cost_tracker
                # Sum up costs for this model
                for call in cost_tracker.by_call:
                    # Match against actual model name (e.g., "gpt-5-nano", "gemini-2.5-flash-lite")
                    call_model = call.get("model", "")
                    if model_id in call_model.lower() or model_id.replace("gpt5", "gpt") in call_model.lower():
                        total_cost += call.get("cost_usd", 0)
                        total_tokens += call.get("total_tokens", 0)

            model_data.append({
                "Model": actual_name,  # Use actual model name instead of generic ID
                "Successful": success_count,
                "Total Tokens": f"{total_tokens:,}" if total_tokens > 0 else "N/A",
                "Total Cost": f"${total_cost:.4f}" if total_cost > 0 else "N/A"
            })

        st.dataframe(pd.DataFrame(model_data), use_container_width=True)

    # Tab 2: Image Results
    with result_tabs[1]:
        st.markdown("#### 🖼️ Per-Image Analysis")

        for idx, result in enumerate(results):
            with st.expander(f"📸 {result['image_name']}", expanded=(idx == 0)):
                # Show image
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(result['image_path'], caption=result['image_name'], use_container_width=True)

                with col2:
                    # Show model responses
                    for model_name, analysis in result.get("model_results", {}).items():
                        st.markdown(f"**{model_name}:**")

                        # Check if we have structured data
                        has_ratings = (
                            hasattr(analysis, 'movement_rating') and analysis.movement_rating is not None
                        ) or (
                            hasattr(analysis, 'visual_quality_rating') and analysis.visual_quality_rating is not None
                        ) or (
                            hasattr(analysis, 'artifact_presence_rating') and analysis.artifact_presence_rating is not None
                        )

                        if has_ratings:
                            # Display structured ratings
                            rating_cols = st.columns(3)

                            with rating_cols[0]:
                                if hasattr(analysis, 'movement_rating') and analysis.movement_rating:
                                    st.metric("Movement", f"{analysis.movement_rating:.1f}/5")

                            with rating_cols[1]:
                                if hasattr(analysis, 'visual_quality_rating') and analysis.visual_quality_rating:
                                    st.metric("Visual Quality", f"{analysis.visual_quality_rating:.1f}/5")

                            with rating_cols[2]:
                                if hasattr(analysis, 'artifact_presence_rating') and analysis.artifact_presence_rating:
                                    st.metric("Artifact Score", f"{analysis.artifact_presence_rating:.1f}/5")

                            # Show detected artifacts if any
                            if hasattr(analysis, 'detected_artifacts') and analysis.detected_artifacts:
                                st.markdown("**Detected Artifacts:**")
                                for artifact in analysis.detected_artifacts:
                                    st.markdown(f"- {artifact}")

                        # Show rationale/response
                        if hasattr(analysis, 'rationale') and analysis.rationale:
                            with st.expander("📝 Analysis Details", expanded=False):
                                st.info(analysis.rationale)
                        elif hasattr(analysis, 'raw_response') and analysis.raw_response:
                            with st.expander("📝 Raw Response", expanded=False):
                                st.info(analysis.raw_response)

                        # Show confidence
                        if hasattr(analysis, 'confidence'):
                            confidence_pct = analysis.confidence * 100
                            # Color code confidence
                            if confidence_pct >= 80:
                                st.success(f"✅ Confidence: {confidence_pct:.1f}%")
                            elif confidence_pct >= 50:
                                st.warning(f"⚠️ Confidence: {confidence_pct:.1f}%")
                            else:
                                st.error(f"❌ Confidence: {confidence_pct:.1f}%")

                        st.divider()

    # Tab 3: Visualizations
    with result_tabs[2]:
        st.markdown("#### 📈 Analysis Visualizations")
        st.info("📊 Visualization integration coming soon!")
        # TODO: Add visualizations from core/vision_visualizations.py

    # Tab 4: Export
    with result_tabs[3]:
        st.markdown("#### 💾 Export Results")

        # Prepare export data - convert Pydantic models to dicts
        export_results = []
        for result in results:
            export_result = {
                "image_path": result['image_path'],
                "image_name": result['image_name'],
                "model_results": {}
            }

            for model_name, analysis in result.get("model_results", {}).items():
                # Convert Pydantic model to dict
                if hasattr(analysis, 'model_dump'):
                    export_result["model_results"][model_name] = analysis.model_dump()
                elif hasattr(analysis, 'dict'):
                    export_result["model_results"][model_name] = analysis.dict()
                else:
                    export_result["model_results"][model_name] = str(analysis)

            export_results.append(export_result)

        export_data = {
            "preset_name": preset_name,
            "timestamp": datetime.now().isoformat(),
            "models": selected_models,
            "results": export_results
        }

        # JSON export
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"test6_preset_{preset_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

        # CSV export (flattened)
        csv_data = []
        for result in results:
            for model_name, analysis in result.get("model_results", {}).items():
                # Access Pydantic model attributes
                response_text = ""
                if hasattr(analysis, 'rationale'):
                    response_text = analysis.rationale
                elif hasattr(analysis, 'raw_response'):
                    response_text = analysis.raw_response

                csv_data.append({
                    "Image": result['image_name'],
                    "Model": model_name,
                    "Response": response_text,
                    "Confidence": getattr(analysis, 'confidence', 0.0)
                })

        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_str = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_str,
                file_name=f"test6_preset_{preset_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


def render_mode_b_general_visual(selected_models: List[str]) -> None:
    """Render Mode B: General Visual LLM Comparison workflow."""
    st.subheader("🌐 Mode B: General Visual LLM Comparison Framework")

    st.markdown("""
    **Workflow:**
    1. Collect images via web search or manual upload
    2. Run multiple visual LLMs in parallel
    3. Compare outputs and generate visualizations
    4. Use LLM judge for meta-analysis
    5. Interactive Q&A on results
    """)

    # Add Mode B specific documentation popover
    with st.popover("📖 Mode B: Detailed Orchestration", help="Click for Mode B workflow details"):
        st.markdown("**Mode B: General Visual LLM Comparison Framework**")
        st.markdown("Compare visual LLMs on any image dataset with flexible analysis tasks.")

        st.markdown("**Step-by-Step Orchestration:**")
        st.code("""
# 1. Image Collection (two options)

# Option A: Web search collection
images = await search_and_download_images(
    query="red pandas in snow",
    num_images=20,
    cache_preset="red_pandas"
)

# Option B: Upload custom dataset
images = load_uploaded_images(uploaded_files)

# 2. Define analysis task
task_description = '''
Analyze each image and provide:
1. Object detection (what objects are present)
2. Scene classification (indoor/outdoor, setting)
3. Anomaly detection (anything unusual)
4. Dominant colors and composition
'''

# 3. Build general analysis prompt
prompt = build_general_visual_analysis_prompt(task_description)

# 4. Analyze all images with all models
results = []
for image in images:
    model_results = await analyze_image_multi_model(
        image_path=image.path,
        prompt=prompt,
        selected_models=selected_models
    )
    results.append({
        "image_name": image.name,
        "image_path": image.path,
        "model_results": model_results
    })

# 5. Advanced Analysis Pipeline

# 5a. Computational Analysis (optional)
computational_plan = await plan_computational_analysis(
    visual_llm_outputs=results,
    task_description=task_description
)
code = generate_analysis_code(computational_plan)
computational_results = execute_code(code)

# 5b. Model Evaluation
evaluation = await evaluate_visual_llm_performance(
    visual_llm_outputs=results,
    task_description=task_description,
    computational_results=computational_results,
    judge_model="gpt-5-nano"
)

# 6. Interactive Q&A
while user_has_questions:
    answer = await answer_followup_question(
        question=user_question,
        visual_llm_outputs=results,
        computational_results=computational_results,
        evaluation_results=evaluation,
        conversation_history=history
    )
    # Automatically displays relevant images
        """, language="python")

        st.markdown("**Key Functions:**")
        st.code("""
# Image collection with caching
async def search_and_download_images(query, num_images, cache_preset):
    # Uses Linkup API for web search
    # Caches results for reuse
    # Returns list of image paths

# Computational analysis planning
async def plan_computational_analysis(visual_llm_outputs, task_description):
    # LLM generates Python code to analyze results
    # Examples: color analysis, object frequency, model agreement

# Model evaluation with LLM judge
async def evaluate_visual_llm_performance(
    visual_llm_outputs, task_description, judge_model
):
    # Judge evaluates each model's performance
    # Returns rankings, scores, and rationale

# Interactive Q&A with image context
async def answer_followup_question(question, visual_llm_outputs, ...):
    # LLM selects relevant images for the question
    # Provides answer with image references
    # UI automatically displays referenced images
        """, language="python")

        st.markdown("---")
        st.markdown("**Example Input:**")
        st.code("""
# Image Collection:
images = [
  "urban_scene_001.jpg",  # City street with cars and pedestrians
  "nature_002.jpg",       # Forest landscape
  "product_003.jpg"       # Product photo on white background
]

# Analysis Task:
"Detect all objects, classify the scene type, and identify
 the dominant colors in each image"

# Selected Models:
["gpt-5-nano", "gemini-2.5-flash-lite", "claude-3.5-sonnet"]
        """, language="text")

        st.markdown("**Example Output:**")
        st.code("""
# Multi-model analysis for "urban_scene_001.jpg":
{
  "image_name": "urban_scene_001.jpg",
  "model_results": {
    "gpt-5-nano": {
      "detected_objects": ["car", "pedestrian", "traffic_light",
                          "building", "street_sign"],
      "scene_classification": "urban_street",
      "dominant_colors": ["#4A4A4A", "#FFFFFF", "#FF0000"],
      "confidence": 0.92,
      "rationale": "Clear urban environment with multiple vehicles
        and pedestrians. Traffic infrastructure visible."
    },
    "gemini-2.5-flash-lite": {
      "detected_objects": ["vehicle", "person", "traffic_signal",
                          "building", "road"],
      "scene_classification": "city_street",
      "dominant_colors": ["#3C3C3C", "#F5F5F5", "#E63946"],
      "confidence": 0.88,
      "rationale": "Typical city street scene with traffic and
        pedestrians. Urban architecture in background."
    },
    "claude-3.5-sonnet": {
      "detected_objects": ["car", "person", "traffic_light",
                          "building", "sidewalk"],
      "scene_classification": "urban_street",
      "dominant_colors": ["#424242", "#FAFAFA", "#D32F2F"],
      "confidence": 0.90,
      "rationale": "Urban street environment with active traffic
        and pedestrian activity."
    }
  }
}
        """, language="json")

        st.markdown("**Calculation Steps:**")
        st.markdown("**Step 1: Object Detection Agreement**")
        st.code("""
# Normalize and compare detected objects:
GPT-5:    ["car", "pedestrian", "traffic_light", "building", "street_sign"]
Gemini:   ["vehicle", "person", "traffic_signal", "building", "road"]
Claude:   ["car", "person", "traffic_light", "building", "sidewalk"]

# Semantic matching:
"car" ≈ "vehicle" ≈ "car" → 3/3 models agree
"pedestrian" ≈ "person" ≈ "person" → 3/3 models agree
"traffic_light" ≈ "traffic_signal" ≈ "traffic_light" → 3/3 models agree
"building" = "building" = "building" → 3/3 models agree
"street_sign" vs "road" vs "sidewalk" → 0/3 agree (different objects)

Object Agreement Score: 4/5 = 80%
        """, language="text")

        st.markdown("**Step 2: Computational Analysis**")
        st.code("""
# LLM generates Python code to analyze all results:
import pandas as pd
from collections import Counter

# Extract all detected objects across all images
all_objects = []
for result in visual_llm_outputs:
    for model_result in result['model_results'].values():
        all_objects.extend(model_result['detected_objects'])

# Frequency analysis
object_freq = Counter(all_objects)
top_10 = object_freq.most_common(10)

# Results:
┌─────────────────┬───────────┐
│ Object          │ Frequency │
├─────────────────┼───────────┤
│ building        │ 45        │
│ person/pedestrian│ 38        │
│ car/vehicle     │ 35        │
│ tree            │ 28        │
│ traffic_light   │ 22        │
│ road            │ 20        │
│ sky             │ 18        │
│ sidewalk        │ 15        │
│ sign            │ 12        │
│ window          │ 10        │
└─────────────────┴───────────┘

# Color analysis
avg_dominant_colors = calculate_color_distribution(all_results)
# Most common: Grays (#4A4A4A), Whites (#FFFFFF), Blues (#1E88E5)
        """, language="python")

        st.markdown("**Step 3: Model Evaluation**")
        st.code("""
# LLM judge evaluates each model:
{
  "best_model": "gpt-5-nano",
  "rankings": [
    {
      "model": "gpt-5-nano",
      "score": 92,
      "strengths": [
        "Highest object detection accuracy",
        "Most detailed rationales",
        "Consistent confidence calibration"
      ]
    },
    {
      "model": "claude-3.5-sonnet",
      "score": 88,
      "strengths": [
        "Good scene classification",
        "Balanced confidence scores",
        "Clear explanations"
      ]
    },
    {
      "model": "gemini-2.5-flash-lite",
      "score": 85,
      "strengths": [
        "Fast inference",
        "Good color detection",
        "Reasonable accuracy"
      ]
    }
  ],
  "evaluation_criteria": {
    "accuracy": "Object detection correctness",
    "consistency": "Agreement with other models",
    "confidence_calibration": "Confidence matches accuracy",
    "detail": "Rationale completeness"
  }
}
        """, language="json")

        st.markdown("**Step 4: Interactive Q&A Example**")
        st.code("""
# User Question: "Which images contain traffic lights?"

# LLM selects relevant images:
relevant_images = [
  "urban_scene_001.jpg",  # All 3 models detected traffic_light
  "urban_scene_005.jpg"   # 2/3 models detected traffic_signal
]

# Answer:
"Based on the analysis, 2 images contain traffic lights:

1. **urban_scene_001.jpg**: All three models (GPT-5, Gemini,
   Claude) detected traffic lights with high confidence (avg 0.90).

2. **urban_scene_005.jpg**: Two models (GPT-5 and Claude) detected
   traffic signals with moderate confidence (avg 0.75).

The images are displayed below for your reference."

# UI automatically shows these images in expanders
        """, language="text")

        st.markdown("---")
        st.markdown("**Expected Outputs:**")
        st.markdown("- Multi-model analysis results for all images")
        st.markdown("- Computational analysis (color distributions, object frequencies, etc.)")
        st.markdown("- Model performance evaluation and rankings")
        st.markdown("- Interactive Q&A with automatic image display")
        st.markdown("- Downloadable results in JSON/CSV format")

    if 'test6_custom_curated_images' not in st.session_state:
        st.session_state.test6_custom_curated_images = []
    if 'test6_custom_uploaded_images' not in st.session_state:
        st.session_state.test6_custom_uploaded_images = []
    if 'test6_uploaded_files' not in st.session_state:
        st.session_state.test6_uploaded_files = {}
    if 'test6_curated_label' not in st.session_state:
        st.session_state.test6_curated_label = None

    # Initialize history manager
    history_manager = AnalysisHistoryManager()

    # History viewer section
    st.markdown("---")
    st.markdown("### 📚 Analysis History")

    col1, col2 = st.columns([3, 1])

    with col1:
        # Get all saved analyses
        all_analyses = history_manager.get_all_analyses()

        if all_analyses:
            # Create options for selectbox
            history_options = ["➕ New Analysis"] + [
                f"{a.get('preset_name', 'Unknown')} - {datetime.fromisoformat(a.get('timestamp', '')).strftime('%Y-%m-%d %H:%M:%S')} ({a.get('num_images', 0)} images, {a.get('num_models', 0)} models)"
                for a in all_analyses
            ]

            selected_history = st.selectbox(
                "View previous analysis or start new",
                options=history_options,
                key="test6_history_selector"
            )

            # If user selected a previous analysis
            if selected_history != "➕ New Analysis":
                selected_index = history_options.index(selected_history) - 1
                analysis_id = all_analyses[selected_index].get("id")

                # Load the analysis
                loaded_data = history_manager.load_analysis(analysis_id)

                if loaded_data:
                    st.success(f"✅ Loaded analysis from {datetime.fromisoformat(loaded_data.get('timestamp', '')).strftime('%Y-%m-%d %H:%M:%S')}")

                    # Display the loaded results
                    from ui.test6_advanced_results import display_advanced_results

                    display_advanced_results(
                        results=loaded_data.get("results", []),
                        selected_models=loaded_data.get("selected_models", []),
                        preset_name=loaded_data.get("preset_name", "Unknown"),
                        task_description=loaded_data.get("task_description", ""),
                        _CONFIG=_CONFIG
                    )

                    # Stop here - don't show the new analysis UI
                    return
                else:
                    st.error("❌ Failed to load analysis")
        else:
            st.info("💡 No previous analyses found. Run your first analysis below!")

    with col2:
        if all_analyses:
            if st.button("🗑️ Clear History", use_container_width=True):
                # Delete all analyses
                for analysis in all_analyses:
                    history_manager.delete_analysis(analysis.get("id"))
                st.success("✅ History cleared!")
                st.rerun()

    st.markdown("---")

    # Preset examples
    PRESET_EXAMPLES = {
        "🎮 VR Avatar Quality Check": {
            "search_query": "VR avatar screenshots virtual reality",
            "task": "Analyze the VR avatar for visual artifacts including: red lines in eyes, finger/feet movement issues, avatar distortions, clothing distortions. Rate the overall visual quality on a scale of 1-5."
        },
        "🏥 Medical Image Analysis": {
            "search_query": "medical imaging X-ray CT scan",
            "task": "Identify anatomical structures, detect any abnormalities or anomalies, assess image quality, and provide diagnostic observations."
        },
        "🏭 Product Defect Detection": {
            "search_query": "product defects manufacturing quality control",
            "task": "Detect any visible defects, scratches, dents, discoloration, or manufacturing flaws. Rate the product quality on a scale of 1-5."
        },
        "🌆 Scene Understanding": {
            "search_query": "urban street scenes city photography",
            "task": "Identify objects, people, vehicles, and landmarks. Describe the scene composition, lighting conditions, and overall atmosphere."
        },
        "📊 Chart & Diagram Analysis": {
            "search_query": "business charts graphs data visualization",
            "task": "Extract data from charts, identify trends, describe the visualization type, and summarize key insights."
        },
        "🎨 Art & Style Analysis": {
            "search_query": "artwork paintings artistic styles",
            "task": "Identify the artistic style, color palette, composition techniques, and emotional tone. Describe the subject matter and artistic elements."
        }
    }

    # Preset selector
    st.markdown("### 🎯 Quick Start Presets")

    preset_choice = st.selectbox(
        "Choose a preset example or create custom",
        options=["Custom"] + list(PRESET_EXAMPLES.keys()),
        key="test6_preset_choice"
    )

    # Show image source selection if preset is selected and has cache
    image_source = None
    if preset_choice != "Custom":
        cached_images = get_cached_images_for_preset(preset_choice)

        if cached_images:
            # Show cache info and let user choose
            cache_info = get_cache_info(preset_choice)

            st.markdown("### 📸 Image Source")

            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.info(f"✅ Found {cache_info['num_images']} cached images from previous search")
                st.caption(f"Last updated: {cache_info.get('last_modified', 'Unknown')}")

            with col2:
                image_source = st.radio(
                    "Choose image source",
                    options=["Use Cached Images", "Search New Images"],
                    key="test6_image_source",
                    horizontal=True
                )

            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️ Clear Cache", key="test6_clear_cache", help="Delete cached images"):
                    clear_preset_cache(preset_choice)
                    st.success("✅ Cache cleared!")
                    st.rerun()

    # Run button
    st.markdown("<br>", unsafe_allow_html=True)
    use_preset = st.button(
        "🚀 Run Preset",
        type="primary",
        disabled=(preset_choice == "Custom"),
        key="test6_run_preset",
        help="Run analysis with preset configuration using test images"
    )

    if use_preset and preset_choice != "Custom":
        preset = PRESET_EXAMPLES[preset_choice]
        st.success(f"✅ Running preset: **{preset_choice}**")

        # Store preset choice in session state to trigger analysis
        st.session_state.test6_pending_preset = preset_choice
        st.session_state.test6_pending_task = preset['task']

    # Check if we have a pending preset to run (either from button click or previous run)
    if 'test6_pending_preset' in st.session_state and st.session_state.test6_pending_preset == preset_choice:
        preset = PRESET_EXAMPLES[preset_choice]

        # Check for cached images first
        cached_images = get_cached_images_for_preset(preset_choice)
        test_images = None

        if cached_images:
            # Check if user selected to use cached images
            if image_source == "Use Cached Images":
                st.success(f"📸 Using {len(cached_images)} cached images")
                test_images = cached_images
            else:
                # User wants to search new images - ALWAYS use Master LLM curation
                # Clear cache first to avoid conflicts
                clear_preset_cache(preset_choice)

                # Get API keys
                linkup_key = None
                if hasattr(st, 'secrets') and 'LINKUP_API_KEY' in st.secrets:
                    linkup_key = st.secrets['LINKUP_API_KEY']
                elif 'LINKUP_API_KEY' in _CONFIG:
                    linkup_key = _CONFIG['LINKUP_API_KEY']

                openai_key = _CONFIG.get('OPENAI_API_KEY')

                # Use Master LLM curation (always)
                if not openai_key:
                    st.error("❌ OpenAI API key required for Master LLM curation")
                    return

                if not linkup_key:
                    st.error("❌ Linkup API key required for image search")
                    return

                st.info("🧠 Master LLM curating high-quality images...")

                from core.master_llm_curator import curate_image_dataset

                with st.spinner("Master LLM generating optimized search queries..."):
                    curation_result = asyncio.run(curate_image_dataset(
                        task_description=preset['task'],
                        preset_name=preset_choice,
                        num_images_needed=10,
                        master_model="gpt-5-mini",
                        openai_api_key=openai_key,
                        linkup_api_key=linkup_key,
                        relevance_threshold=70.0
                    ))

                test_images = curation_result.get("selected_images", [])

                # Store ground truth in session state
                st.session_state.test6_ground_truths = curation_result.get("ground_truths", {})
                st.session_state.test6_curation_report = curation_result.get("curation_report", {})

                # Display curation report
                report = curation_result.get("curation_report", {})

                st.success(f"✅ Master LLM curated {len(test_images)} high-quality images")

                with st.expander("📋 Curation Report", expanded=True):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Images Evaluated", report.get("images_evaluated", 0))
                    with col2:
                        st.metric("Images Selected", report.get("images_selected", 0))
                    with col3:
                        st.metric("Images Rejected", report.get("images_rejected", 0))

                    st.markdown("**Generated Search Queries:**")
                    for query_info in report.get("queries_generated", []):
                        st.markdown(f"- **{query_info.get('query')}**")
                        st.caption(f"  Rationale: {query_info.get('rationale')}")

                if not test_images:
                    st.error("❌ Master LLM could not find suitable images. Try adjusting the task description.")
                    return

        else:
            # No cached images - must search with Master LLM curation
            st.info(f"🔍 No cached images found. Master LLM will curate high-quality images...")

            # Get API keys
            linkup_key = None
            if hasattr(st, 'secrets') and 'LINKUP_API_KEY' in st.secrets:
                linkup_key = st.secrets['LINKUP_API_KEY']
            elif 'LINKUP_API_KEY' in _CONFIG:
                linkup_key = _CONFIG['LINKUP_API_KEY']

            openai_key = _CONFIG.get('OPENAI_API_KEY')

            # Use Master LLM curation (always)
            if not openai_key:
                st.error("❌ OpenAI API key required for Master LLM curation")
                return

            if not linkup_key:
                st.error("❌ Linkup API key required for image search")
                return

            from core.master_llm_curator import curate_image_dataset

            with st.spinner("🧠 Master LLM generating optimized search queries..."):
                curation_result = asyncio.run(curate_image_dataset(
                    task_description=preset['task'],
                    preset_name=preset_choice,
                    num_images_needed=10,
                    master_model="gpt-5-mini",
                    openai_api_key=openai_key,
                    linkup_api_key=linkup_key,
                    relevance_threshold=70.0
                ))

            test_images = curation_result.get("selected_images", [])

            # Store ground truth in session state
            st.session_state.test6_ground_truths = curation_result.get("ground_truths", {})
            st.session_state.test6_curation_report = curation_result.get("curation_report", {})

            # Display curation report
            report = curation_result.get("curation_report", {})

            st.success(f"✅ Master LLM curated {len(test_images)} high-quality images")

            with st.expander("📋 Curation Report", expanded=True):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Images Evaluated", report.get("images_evaluated", 0))
                with col2:
                    st.metric("Images Selected", report.get("images_selected", 0))
                with col3:
                    st.metric("Images Rejected", report.get("images_rejected", 0))

                st.markdown("**Generated Search Queries:**")
                for query_info in report.get("queries_generated", []):
                    st.markdown(f"- **{query_info.get('query')}**")
                    st.caption(f"  Rationale: {query_info.get('rationale')}")

            if not test_images:
                st.error("❌ Master LLM could not find suitable images. Try adjusting the task description.")
                return

        # Verify we have images before continuing
        if not test_images:
            st.error("❌ No images available for analysis.")
            return

        # Get model names for display
        from core.vision_model_discovery import get_recommended_vision_models
        recommended = get_recommended_vision_models()

        model_display_names = []
        for model_id in selected_models:
            if model_id == "gpt5":
                model_name = recommended.get("openai", "gpt-5-nano").split('/')[-1]
                model_display_names.append(f"GPT-5 ({model_name})")
            elif model_id == "gemini":
                model_name = recommended.get("google", "gemini-2.5-flash-lite").split('/')[-1]
                model_display_names.append(f"Gemini 2.5 ({model_name})")
            elif model_id == "claude":
                model_name = recommended.get("anthropic", "claude-sonnet-4.5").split('/')[-1]
                model_display_names.append(f"Claude 4.5 ({model_name})")
            elif model_id == "llama":
                model_name = recommended.get("meta-llama", "llama-3.2-90b-vision-instruct").split('/')[-1]
                model_display_names.append(f"Llama 3.2 ({model_name})")

        st.info(f"📸 Using {len(test_images)} test images from `test_dataset/visual_llm_images/`")

        # Show configuration
        with st.expander("📋 Preset Configuration", expanded=True):
            st.markdown(f"**Search Query:** `{preset['search_query']}`")
            st.markdown(f"**Analysis Task:**")
            st.info(preset['task'])
            st.markdown(f"**Models:** {', '.join(model_display_names)}")
            st.markdown(f"**Images:** {len(test_images)} test images")

        # Run analysis
        run_preset_analysis(
            test_images=test_images,
            selected_models=selected_models,
            task_description=preset['task'],
            preset_name=preset_choice,
            preset_config=preset
        )

        # Clear pending preset after analysis
        if 'test6_pending_preset' in st.session_state:
            del st.session_state.test6_pending_preset
        if 'test6_pending_task' in st.session_state:
            del st.session_state.test6_pending_task

        return  # Exit early after running preset

    st.divider()

    # Image collection method
    st.markdown("### 📸 Image Collection")

    collection_method = st.radio(
        "Choose image collection method",
        options=["Web Search (Linkup API)", "Manual Upload"],
        horizontal=True,
        key="test6_collection_method"
    )

    if collection_method == "Web Search (Linkup API)":
        search_query = st.text_input(
            "Enter search query for images",
            placeholder="e.g., 'VR avatar screenshots', 'product defects', 'medical imaging'",
            key="test6_search_query"
        )

        num_images = st.slider(
            "Number of images to download",
            min_value=5,
            max_value=50,
            value=20,
            key="test6_num_images"
        )

        if st.button("🔍 Search and Download Images", key="test6_search_btn"):
            if search_query:
                st.info(f"🔍 Searching for '{search_query}' and downloading {num_images} images...")
                # TODO: Implement Linkup API image search
                st.warning("⚠️ Linkup API integration coming soon!")
            else:
                st.error("Please enter a search query")

    else:
        # Manual upload
        uploaded_images = st.file_uploader(
            "Upload images for analysis",
            type=["jpg", "jpeg", "png", "gif", "webp"],
            accept_multiple_files=True,
            key="test6_uploaded_images"
        )

        if uploaded_images:
            st.success(f"✅ Uploaded {len(uploaded_images)} images")

            # Preview images
            with st.expander("🖼️ Preview Uploaded Images", expanded=False):
                cols = st.columns(min(len(uploaded_images), 4))
                for idx, img in enumerate(uploaded_images[:8]):  # Show max 8 previews
                    with cols[idx % 4]:
                        st.image(img, caption=img.name, use_container_width=True)

    # Analysis task description
    st.markdown("### 📝 Analysis Task")

    task_description = st.text_area(
        "Describe what the visual LLMs should analyze",
        value="Detect objects, classify image category, and identify any anomalies or artifacts",
        key="test6_task_description",
        height=100
    )

    # Run analysis button
    st.divider()

    if st.button("▶️ Run Multi-Model Visual Analysis", type="primary", use_container_width=True):
        st.info("🚀 Mode B analysis coming soon!")
        # TODO: Implement Mode B analysis


def run_mode_a_analysis(
    df: pd.DataFrame,
    selected_models: List[str],
    artifact_types: List[str]
) -> None:
    """Run Mode A: VR Avatar analysis."""
    st.markdown("### 🔄 Running Analysis...")

    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Initialize results storage
    if 'test6_results' not in st.session_state:
        st.session_state.test6_results = {}

    # Build analysis prompt
    prompt = build_vr_avatar_analysis_prompt(artifact_types)

    # Get API keys from config
    openai_key = _CONFIG.get('OPENAI_API_KEY')
    gemini_key = _CONFIG.get('GEMINI_API_KEY')
    openrouter_key = _CONFIG.get('OPENROUTER_API_KEY')

    # Analyze each avatar
    total_avatars = len(df)

    async def analyze_all_avatars():
        results = []

        for idx, row in df.iterrows():
            status_text.text(f"Analyzing avatar {idx + 1}/{total_avatars}: {row.get('avatar_id', 'Unknown')}")

            # Use screenshot if available, otherwise video (for now, just screenshot)
            image_path = row.get('screenshot_path')

            if image_path and os.path.exists(image_path):
                # Analyze with selected models
                model_results = await analyze_image_multi_model(
                    image_path=image_path,
                    prompt=prompt,
                    selected_models=selected_models,
                    openai_api_key=openai_key,
                    gemini_api_key=gemini_key,
                    openrouter_api_key=openrouter_key
                )

                results.append({
                    'avatar_id': row.get('avatar_id'),
                    'model_results': model_results,
                    'human_ratings': {
                        'movement': row.get('human_movement_rating'),
                        'visual': row.get('human_visual_rating'),
                        'comfort': row.get('human_comfort_rating')
                    },
                    'bug_description': row.get('bug_description')
                })
            else:
                st.warning(f"⚠️ Image not found for avatar {row.get('avatar_id')}: {image_path}")

            progress_bar.progress((idx + 1) / total_avatars)

        return results

    # Run async analysis
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(analyze_all_avatars())

    # Store results
    st.session_state.test6_results = {
        'mode': 'A',
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'selected_models': selected_models
    }

    status_text.text("✅ Analysis complete!")
    progress_bar.progress(1.0)

    # Display results
    display_mode_a_results(results, selected_models)


def display_mode_a_results(results: List[Dict], selected_models: List[str]) -> None:
    """Display Mode A analysis results with visualizations."""
    st.markdown("### 📊 Analysis Results")

    st.success(f"✅ Analyzed {len(results)} avatars with {len(selected_models)} visual LLM models")

    # TODO: Add visualizations
    # - Human vs LLM ratings comparison
    # - Artifact detection frequency
    # - Model agreement analysis
    # - Bug prioritization

    st.info("📈 Visualizations coming soon!")

    # Show raw results for now
    with st.expander("🔍 View Raw Results", expanded=False):
        st.json(results)

