"""
Interactive Q&A Interface for Visual LLM Analysis

Allows users to ask follow-up questions about analysis results.
"""

import json
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime


# Simple in-memory cache for LLM selection results per session
_SELECTION_CACHE: Dict[str, List[str]] = {}

async def answer_followup_question(
    question: str,
    visual_llm_outputs: List[Dict[str, Any]],
    computational_results: Optional[Dict[str, Any]],
    evaluation_results: Optional[Dict[str, Any]],
    conversation_history: List[Dict[str, str]],
    qa_model: str = "gpt-5-nano",
    openai_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Answer a follow-up question about the analysis.

    Args:
        question: User's question
        visual_llm_outputs: All visual LLM outputs
        computational_results: Computational analysis results
        evaluation_results: Model evaluation results
        conversation_history: Previous Q&A exchanges
        qa_model: Model to use for Q&A
        openai_api_key: OpenAI API key

    Returns:
        Dict containing:
        - answer: Answer to the question
        - relevant_data: Relevant data excerpts
        - suggested_actions: Suggested follow-up actions
    """
    from openai import AsyncOpenAI

    # Build context
    context = _build_qa_context(
        visual_llm_outputs,
        computational_results,
        evaluation_results
    )

    # Build conversation messages
    messages = [
        {
            "role": "system",
            "content": f"""You are an expert AI assistant helping analyze visual LLM outputs.

**Available Context:**
{context}

**Capabilities:**
- Answer questions about analysis results and model performance
- Provide insights on specific images when asked
- Compare model outputs and explain differences
- Suggest follow-up actions and improvements

**When asked about specific images:**
- Reference the image by name
- Summarize key findings from all models
- Highlight agreements and disagreements
- Note: The UI will automatically display the image alongside your answer

Answer questions clearly and concisely. Reference specific data when relevant.
Suggest follow-up actions when appropriate."""
        }
    ]

    # Add conversation history
    for exchange in conversation_history[-5:]:  # Last 5 exchanges
        messages.append({"role": "user", "content": exchange["question"]})
        messages.append({"role": "assistant", "content": exchange["answer"]})

    # Add current question
    messages.append({"role": "user", "content": question})

    # Call LLM
    client = AsyncOpenAI(api_key=openai_api_key)

    response = await client.chat.completions.create(
        model=qa_model,
        messages=messages,
        max_completion_tokens=16000  # Increased for detailed Q&A responses
    )

    answer = response.choices[0].message.content

    # Extract relevant data
    relevant_data = _extract_relevant_data(
        question,
        visual_llm_outputs,
        computational_results
    )

    # Extract relevant images by explicit mention (fast path)
    explicit_images = _extract_relevant_images(
        question,
        visual_llm_outputs
    )

    # Always provide a compact list of images to the LLM to select relevant ones
    image_descriptors = _build_image_descriptors(visual_llm_outputs)
    try:
        llm_selected_names = await _llm_select_relevant_images(
            question=question,
            image_descriptors=image_descriptors,
            qa_model=qa_model,
            openai_api_key=openai_api_key
        )
    except Exception:
        llm_selected_names = []

    # Map selected names to full records
    name_to_output = {o.get("image_name", ""): o for o in visual_llm_outputs}
    llm_selected_images = []
    for name in llm_selected_names:
        o = name_to_output.get(name)
        if o:
            llm_selected_images.append({
                "image_name": o.get("image_name", ""),
                "image_path": o.get("image_path", ""),
                "model_results": o.get("model_results", {})
            })

    # Merge and dedupe (prefer explicit first), cap at 5
    def _key(img):
        return img.get("image_name")
    merged = []
    for img in explicit_images + llm_selected_images:
        if all(_key(img) != _key(x) for x in merged):
            merged.append(img)
    relevant_images = merged[:5]

    # Generate suggested actions
    suggested_actions = _generate_suggested_actions(question, answer)

    return {
        "answer": answer,
        "relevant_data": relevant_data,
        "relevant_images": relevant_images,
        "suggested_actions": suggested_actions,
        "timestamp": datetime.now().isoformat()
    }


def _build_qa_context(
    visual_llm_outputs: List[Dict[str, Any]],
    computational_results: Optional[Dict[str, Any]],
    evaluation_results: Optional[Dict[str, Any]]
) -> str:
    """Build context string for Q&A."""
    context_parts = []

    # Summary statistics
    num_images = len(visual_llm_outputs)
    models = set()
    for output in visual_llm_outputs:
        models.update(output.get("model_results", {}).keys())

    context_parts.append(f"- Analyzed {num_images} images")
    context_parts.append(f"- Models used: {', '.join(models)}")

    # Computational results summary
    if computational_results and computational_results.get("success"):
        context_parts.append(f"\n**Computational Analysis:**\n{computational_results.get('results', 'N/A')[:500]}")

    # Evaluation summary
    if evaluation_results:
        best_model = evaluation_results.get("best_model", "N/A")
        context_parts.append(f"\n**Best Model:** {best_model}")

        rankings = evaluation_results.get("model_rankings", [])
        if rankings:
            context_parts.append("\n**Model Rankings:**")
            for rank in rankings[:3]:
                context_parts.append(f"  - {rank.get('model')}: {rank.get('score')}/100")

    return "\n".join(context_parts)


def _extract_relevant_data(
    question: str,
    visual_llm_outputs: List[Dict[str, Any]],
    computational_results: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Extract data relevant to the question."""
    relevant = []

    # Simple keyword matching (can be enhanced with embeddings)
    question_lower = question.lower()

    # Check for specific image references
    for output in visual_llm_outputs:
        image_name = output.get("image_name", "")
        if image_name.lower() in question_lower:
            relevant.append({
                "type": "image_output",
                "image_name": image_name,
                "data": output
            })

    # Check for model-specific questions
    for output in visual_llm_outputs[:5]:  # Limit to first 5
        model_results = output.get("model_results", {})
        for model_name in model_results.keys():
            if model_name.lower() in question_lower:
                relevant.append({
                    "type": "model_output",
                    "model": model_name,
                    "image": output.get("image_name"),
                    "data": model_results[model_name]
                })

    return relevant[:10]  # Limit to 10 items


def _extract_relevant_images(
    question: str,
    visual_llm_outputs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract images relevant to the question by explicit references and simple heuristics.

    Returns a list of dicts with:
    - image_name: Name of the image
    - image_path: Path to the image file
    - model_results: Analysis results from all models
    """
    relevant_images: List[Dict[str, Any]] = []
    question_lower = question.lower()

    # Keywords that indicate image-specific questions
    image_keywords = [
        "show me", "display", "look at", "view", "see",
        "image", "picture", "photo", ".jpg", ".png", ".jpeg"
    ]

    # If it's not even an image-oriented question, return empty list
    is_image_question = any(keyword in question_lower for keyword in image_keywords)
    if not is_image_question:
        return []

    # 1) Explicit mentions by image name
    for output in visual_llm_outputs:
        image_name = output.get("image_name", "")
        image_path = output.get("image_path", "")
        if image_name and image_name.lower() in question_lower:
            relevant_images.append({
                "image_name": image_name,
                "image_path": image_path,
                "model_results": output.get("model_results", {})
            })

    # 2) If user asks to show/display/view and mentions a model, include first few where that model appears
    if not relevant_images and any(word in question_lower for word in ["show", "display", "view"]):
        for output in visual_llm_outputs[:5]:  # Limit to first 5
            model_results = output.get("model_results", {})
            for model_name in model_results.keys():
                if model_name.lower() in question_lower:
                    relevant_images.append({
                        "image_name": output.get("image_name", ""),
                        "image_path": output.get("image_path", ""),
                        "model_results": model_results
                    })
                    break

    # Limit to 5 images to avoid overwhelming the UI
    return relevant_images[:5]


def _build_image_descriptors(
    visual_llm_outputs: List[Dict[str, Any]],
    max_desc_chars: int = 160
) -> List[Dict[str, str]]:
    """Build a compact descriptor list for images for LLM selection.
    Each item: {"image_name": str, "descriptor": str}
    """
    descriptors: List[Dict[str, str]] = []
    for output in visual_llm_outputs:
        image_name = output.get("image_name", "")
        if not image_name:
            continue
        # Prefer precomputed metadata descriptor if available
        meta = output.get("metadata", {}) if isinstance(output, dict) else {}
        desc = meta.get("descriptor", "") if isinstance(meta, dict) else ""

        model_results = output.get("model_results", {})
        # Fallback: use the first model's rationale/summary
        if not desc:
            for analysis in model_results.values():
                try:
                    if hasattr(analysis, "rationale") and analysis.rationale:
                        desc = str(analysis.rationale)
                        break
                    if isinstance(analysis, dict) and analysis.get("rationale"):
                        desc = str(analysis.get("rationale"))
                        break
                except Exception:
                    continue
        if not desc:
            # Fallback to image name only
            desc = image_name
        desc = desc.strip().replace("\n", " ")[:max_desc_chars]
        descriptors.append({"image_name": image_name, "descriptor": desc})
    return descriptors


async def _llm_select_relevant_images(
    question: str,
    image_descriptors: List[Dict[str, str]],
    qa_model: str,
    openai_api_key: Optional[str]
) -> List[str]:
    """Ask the LLM to select up to 5 relevant image names from descriptors.
    Returns a list of image names.
    """
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=openai_api_key)

    # Build cache key
    desc_json = json.dumps(image_descriptors[:50], ensure_ascii=False, sort_keys=True)
    desc_hash = hashlib.md5(desc_json.encode("utf-8")).hexdigest()
    q_norm = question.strip().lower()
    cache_key = f"{qa_model}:{desc_hash}:{q_norm}"
    if cache_key in _SELECTION_CACHE:
        return _SELECTION_CACHE[cache_key]

    # Keep the payload small
    compact_payload = {
        "question": question,
        "images": image_descriptors[:50],  # cap context
    }

    system_prompt = (
        "You are assisting in selecting relevant images for a user's question. "
        "You will receive a list of images with short descriptors. "
        "Return ONLY a JSON object with the following shape: "
        "{\"image_names\": [\"name1.jpg\", \"name2.png\"]}. "
        "Select at most 5 images that are most relevant to the user's question."
    )

    user_prompt = (
        "User Question and Available Images (JSON):\n" + json.dumps(compact_payload, ensure_ascii=False)
    )

    resp = await client.chat.completions.create(
        model=qa_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=512,
    )

    content = resp.choices[0].message.content or ""

    # Try to parse a JSON object from the response
    def _extract_json(s: str) -> Optional[Dict[str, Any]]:
        try:
            # Remove code fences if present
            if s.strip().startswith("```) "):
                s = s.strip().strip("`")
            # Find first '{' and last '}'
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start:end+1])
        except Exception:
            return None
        return None

    data = _extract_json(content) or {}
    names = data.get("image_names") or []

    # Ensure names are strings and unique order-preserving
    cleaned: List[str] = []
    for n in names:
        if isinstance(n, str) and n and n not in cleaned:
            cleaned.append(n)
    selected = cleaned[:5]
    try:
        _SELECTION_CACHE[cache_key] = selected
    except Exception:
        pass
    return selected




def _generate_suggested_actions(question: str, answer: str) -> List[str]:
    """Generate suggested follow-up actions."""
    suggestions = []

    question_lower = question.lower()

    # Pattern-based suggestions
    if "compare" in question_lower or "difference" in question_lower:
        suggestions.append("View detailed comparison matrix")
        suggestions.append("Generate comparison visualization")

    if "image" in question_lower and any(word in question_lower for word in ["specific", "particular", "this", "show"]):
        suggestions.append("View full analysis for this image")
        suggestions.append("Compare with similar images")
        suggestions.append("Re-analyze with enhanced prompt")

    if "model" in question_lower and "better" in question_lower:
        suggestions.append("View model evaluation report")
        suggestions.append("Test with different models")

    if "why" in question_lower or "how" in question_lower:
        suggestions.append("View computational analysis details")
        suggestions.append("Generate deeper analysis")

    # Image-specific suggestions
    if any(word in question_lower for word in ["show", "display", "view", ".jpg", ".png"]):
        suggestions.append("Ask about other images")
        suggestions.append("Compare this image with others")

    # Default suggestions
    if not suggestions:
        suggestions.extend([
            "Ask about specific images (e.g., 'Show me image_003.jpg')",
            "Compare model outputs",
            "Request visualization"
        ])

    return suggestions[:5]  # Limit to 5 suggestions


async def refine_and_reanalyze(
    image_paths: List[str],
    enhanced_prompts: Dict[str, str],
    selected_models: List[str],
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Re-analyze images with enhanced prompts.

    Args:
        image_paths: Paths to images to re-analyze
        enhanced_prompts: Dict mapping model IDs to enhanced prompts
        selected_models: Models to use
        API keys

    Returns:
        List of analysis results
    """
    from core.visual_llm_clients import analyze_image_multi_model

    results = []

    for image_path in image_paths:
        # Get enhanced prompt for each model
        # For now, use the first enhanced prompt (can be model-specific)
        enhanced_prompt = list(enhanced_prompts.values())[0] if enhanced_prompts else "Analyze this image in detail."

        # Run analysis
        model_results = await analyze_image_multi_model(
            image_path=image_path,
            prompt=enhanced_prompt,
            selected_models=selected_models,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            openrouter_api_key=openrouter_api_key
        )

        results.append({
            "image_path": image_path,
            "image_name": image_path.split("/")[-1],
            "model_results": model_results,
            "enhanced_prompt_used": enhanced_prompt
        })

    return results

