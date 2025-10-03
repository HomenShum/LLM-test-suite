"""
Master LLM Image Curator

Uses a master LLM to:
1. Generate optimized search queries for image collection
2. Evaluate downloaded images for task relevance
3. Select the best images for analysis
4. Create ground truth expectations for testing
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import base64
from datetime import datetime
from pydantic import BaseModel


# Pydantic models for structured outputs
class SearchQuery(BaseModel):
    query: str
    rationale: str
    expected_results: str
    priority: int

class SearchQueriesResponse(BaseModel):
    queries: List[SearchQuery]

class ImageEvaluation(BaseModel):
    relevance_score: int
    is_relevant: bool
    rationale: str
    quality_score: int
    issues: List[str]
    expected_content: str

class GroundTruth(BaseModel):
    expected_analysis: str
    key_findings: List[str]
    expected_rating: Optional[int] = None
    confidence_range: str
    difficulty_level: str
    common_mistakes: List[str]
    critical_details: List[str]


async def generate_optimized_search_queries(
    task_description: str,
    preset_name: str,
    num_queries: int = 3,
    master_model: str = "gpt-5-mini",
    openai_api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use master LLM to generate optimized search queries.

    Args:
        task_description: The analysis task description
        preset_name: Name of the preset
        num_queries: Number of search queries to generate
        master_model: Master LLM model to use
        openai_api_key: OpenAI API key

    Returns:
        List of search query dicts with query, rationale, expected_results
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=openai_api_key)

    prompt = f"""You are an expert at generating search queries for image collection.

**Task:** {task_description}
**Preset:** {preset_name}

**Your Goal:**
Generate {num_queries} optimized search queries that will find the MOST RELEVANT images for this task.

**Requirements:**
1. Queries should be specific and targeted
2. Use technical terms when appropriate
3. Include quality indicators (e.g., "high resolution", "clear", "professional")
4. Avoid generic terms that return irrelevant results
5. Consider different aspects of the task

Generate {num_queries} search queries with rationale, expected results, and priority (1=highest, 3=lowest)."""

    response = await client.beta.chat.completions.parse(
        model=master_model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=8000,  # Increased for detailed query generation
        response_format=SearchQueriesResponse
    )

    queries_response = response.choices[0].message.parsed

    # Convert to dict list and sort by priority
    queries = [q.model_dump() for q in queries_response.queries]
    queries.sort(key=lambda x: x.get("priority", 3))

    return queries


async def evaluate_image_relevance(
    image_path: str,
    task_description: str,
    preset_name: str,
    master_model: str = "gpt-5-mini",
    openai_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Use master LLM to evaluate if an image is relevant for the task.

    Args:
        image_path: Path to image file
        task_description: The analysis task
        preset_name: Name of preset
        master_model: Master LLM model
        openai_api_key: OpenAI API key

    Returns:
        Dict with relevance_score (0-100), is_relevant (bool), rationale, issues
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=openai_api_key)

    # Encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Determine image format
    ext = Path(image_path).suffix.lower()
    mime_type = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }.get(ext, 'image/jpeg')

    prompt = f"""Evaluate if this image is relevant for the following task:

**Task:** {task_description}
**Preset:** {preset_name}

**Evaluation Criteria:**
1. Does the image clearly show content relevant to the task?
2. Is the image quality sufficient for analysis?
3. Does the image contain the expected subject matter?
4. Are there any issues (watermarks, text overlays, poor quality)?

Provide relevance score (0-100), quality score (0-100), whether it's relevant, rationale, any issues, and expected content."""

    response = await client.beta.chat.completions.parse(
        model=master_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}"
                        }
                    }
                ]
            }
        ],
        max_completion_tokens=8000,  # Increased for detailed image evaluation
        response_format=ImageEvaluation
    )

    evaluation = response.choices[0].message.parsed

    return evaluation.model_dump()


async def create_ground_truth_expectations(
    image_path: str,
    task_description: str,
    preset_name: str,
    master_model: str = "gpt-5-mini",
    openai_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Use master LLM to create ground truth expectations for an image.

    This creates the "correct answer" that other models will be tested against.

    Args:
        image_path: Path to image
        task_description: Analysis task
        preset_name: Preset name
        master_model: Master LLM
        openai_api_key: API key

    Returns:
        Dict with expected_analysis, key_findings, confidence_range, difficulty_level
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=openai_api_key)

    # Encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(image_path).suffix.lower()
    mime_type = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }.get(ext, 'image/jpeg')

    prompt = f"""You are the MASTER evaluator creating ground truth for testing other visual LLMs.

**Task:** {task_description}
**Preset:** {preset_name}

**Your Goal:**
Provide the DEFINITIVE, CORRECT analysis of this image that will be used as ground truth.

Provide:
- Detailed expected analysis
- Key findings (list)
- Expected rating (1-5 if applicable, or null)
- Confidence range (e.g., "0.8-0.95")
- Difficulty level (easy/medium/hard)
- Common mistakes other models might make (list)
- Critical details that must not be missed (list)

Be thorough and precise. This is the gold standard."""

    response = await client.beta.chat.completions.parse(
        model=master_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}"
                        }
                    }
                ]
            }
        ],
        max_completion_tokens=16000,  # Increased for comprehensive ground truth generation
        response_format=GroundTruth
    )

    ground_truth = response.choices[0].message.parsed

    return ground_truth.model_dump()


async def curate_image_dataset(
    task_description: str,
    preset_name: str,
    num_images_needed: int = 20,
    master_model: str = "gpt-5-mini",
    openai_api_key: Optional[str] = None,
    linkup_api_key: Optional[str] = None,
    relevance_threshold: float = 70.0
) -> Dict[str, Any]:
    """
    Complete image curation workflow:
    1. Generate optimized search queries
    2. Search and download images
    3. Evaluate each image for relevance
    4. Select best images
    5. Create ground truth for selected images
    
    Args:
        task_description: Analysis task
        preset_name: Preset name
        num_images_needed: Target number of images
        master_model: Master LLM
        openai_api_key: OpenAI API key
        linkup_api_key: Linkup API key
        relevance_threshold: Minimum relevance score (0-100)
    
    Returns:
        Dict with selected_images, ground_truths, curation_report
    """
    from core.image_collector import search_and_download_images
    
    curation_report = {
        "timestamp": datetime.now().isoformat(),
        "task": task_description,
        "preset": preset_name,
        "master_model": master_model,
        "queries_generated": [],
        "images_evaluated": 0,
        "images_selected": 0,
        "images_rejected": 0,
        "rejection_reasons": []
    }
    
    # Step 1: Generate optimized search queries
    print("🧠 Master LLM generating optimized search queries...")
    queries = await generate_optimized_search_queries(
        task_description=task_description,
        preset_name=preset_name,
        num_queries=3,
        master_model=master_model,
        openai_api_key=openai_api_key
    )
    
    curation_report["queries_generated"] = queries
    
    # Step 2: Search and download images for each query
    all_downloaded_images = []

    # Calculate how many images to download per query
    # Aim for 2x the needed amount to have good selection after filtering
    target_candidates = num_images_needed * 2
    images_per_query = max(3, target_candidates // len(queries))

    for query_info in queries:
        query = query_info.get("query", "")
        print(f"🔍 Searching with query: {query}")
        print(f"   Rationale: {query_info.get('rationale', 'N/A')}")

        # Download images for this query (limited per query)
        images = await search_and_download_images(
            search_query=query,
            num_images=images_per_query,
            preset_name=preset_name,
            linkup_api_key=linkup_api_key
        )

        all_downloaded_images.extend(images)

        # Stop if we have enough candidates
        if len(all_downloaded_images) >= target_candidates:
            break
    
    print(f"📥 Downloaded {len(all_downloaded_images)} candidate images")
    
    # Step 3: Evaluate each image for relevance
    print("🔍 Master LLM evaluating image relevance...")
    
    evaluated_images = []
    
    for image_path in all_downloaded_images:
        evaluation = await evaluate_image_relevance(
            image_path=image_path,
            task_description=task_description,
            preset_name=preset_name,
            master_model=master_model,
            openai_api_key=openai_api_key
        )
        
        curation_report["images_evaluated"] += 1
        
        evaluated_images.append({
            "path": image_path,
            "evaluation": evaluation
        })
        
        if not evaluation.get("is_relevant") or evaluation.get("relevance_score", 0) < relevance_threshold:
            curation_report["images_rejected"] += 1
            curation_report["rejection_reasons"].append({
                "image": Path(image_path).name,
                "reason": evaluation.get("rationale", "Unknown")
            })
    
    # Step 4: Select best images
    # Sort by relevance score
    evaluated_images.sort(key=lambda x: x["evaluation"].get("relevance_score", 0), reverse=True)
    
    # Select top N relevant images
    selected_images = []
    for img_data in evaluated_images:
        if len(selected_images) >= num_images_needed:
            break
        
        if img_data["evaluation"].get("is_relevant") and \
           img_data["evaluation"].get("relevance_score", 0) >= relevance_threshold:
            selected_images.append(img_data["path"])
    
    curation_report["images_selected"] = len(selected_images)
    
    print(f"✅ Selected {len(selected_images)} high-quality images")
    
    # Step 5: Create ground truth for selected images
    print("📋 Master LLM creating ground truth expectations...")
    
    ground_truths = {}
    
    for image_path in selected_images:
        ground_truth = await create_ground_truth_expectations(
            image_path=image_path,
            task_description=task_description,
            preset_name=preset_name,
            master_model=master_model,
            openai_api_key=openai_api_key
        )
        
        ground_truths[image_path] = ground_truth
    
    print("✅ Ground truth created for all selected images")
    
    return {
        "selected_images": selected_images,
        "ground_truths": ground_truths,
        "curation_report": curation_report,
        "all_evaluations": evaluated_images
    }

