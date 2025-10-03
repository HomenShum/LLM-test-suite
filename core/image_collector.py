"""
Image collection module for Test 6: Visual LLM Testing.

Handles:
- Web search via Linkup API
- Image downloading and local caching
- Preset-specific image management
"""

import os
import hashlib
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import streamlit as st
from PIL import Image


# Image storage directory
_IMAGE_CACHE_DIR = Path("test_dataset/visual_llm_images")
_IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_preset_cache_dir(preset_name: str) -> Path:
    """Get cache directory for a specific preset."""
    # Sanitize preset name for filesystem
    safe_name = preset_name.replace(" ", "_").replace(":", "").replace("/", "_")
    cache_dir = _IMAGE_CACHE_DIR / safe_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_image_hash(url: str) -> str:
    """Generate hash for image URL to use as filename."""
    return hashlib.md5(url.encode()).hexdigest()


async def search_and_download_images(
    search_query: str,
    num_images: int,
    preset_name: str,
    linkup_api_key: Optional[str] = None
) -> List[str]:
    """
    Search for images using Linkup API and download them locally.
    
    Args:
        search_query: Search query for images
        num_images: Number of images to download
        preset_name: Name of preset (for caching)
        linkup_api_key: Linkup API key
    
    Returns:
        List of local image paths
    """
    # Get preset-specific cache directory
    cache_dir = _get_preset_cache_dir(preset_name)
    
    # Check if we already have cached images for this preset
    existing_images = list(cache_dir.glob("*.png")) + list(cache_dir.glob("*.jpg")) + list(cache_dir.glob("*.jpeg"))
    
    if len(existing_images) >= num_images:
        st.info(f"✅ Using {len(existing_images)} cached images for preset: {preset_name}")
        return [str(img) for img in existing_images[:num_images]]
    
    # Need to download new images
    if not linkup_api_key:
        st.warning("⚠️ Linkup API key not found. Using existing test images instead.")
        # Fall back to general test images
        general_images = list(_IMAGE_CACHE_DIR.glob("*.png"))
        return [str(img) for img in general_images[:num_images]]
    
    try:
        # Search for images using Linkup API
        st.info(f"🔍 Searching for images: '{search_query}'...")
        
        image_urls = await _search_images_linkup(search_query, num_images, linkup_api_key)
        
        if not image_urls:
            st.warning("⚠️ No images found. Using existing test images instead.")
            general_images = list(_IMAGE_CACHE_DIR.glob("*.png"))
            return [str(img) for img in general_images[:num_images]]
        
        # Download images
        st.info(f"📥 Downloading {len(image_urls)} images...")
        downloaded_paths = await _download_images(image_urls, cache_dir)
        
        st.success(f"✅ Downloaded {len(downloaded_paths)} images to {cache_dir}")
        
        return downloaded_paths
        
    except Exception as e:
        st.error(f"❌ Error during image search/download: {str(e)}")
        # Fall back to general test images
        general_images = list(_IMAGE_CACHE_DIR.glob("*.png"))
        return [str(img) for img in general_images[:num_images]]


async def _search_images_linkup(
    query: str,
    num_results: int,
    api_key: str
) -> List[str]:
    """
    Search for images using Linkup API.

    Args:
        query: Search query
        num_results: Number of results to return
        api_key: Linkup API key

    Returns:
        List of image URLs
    """
    # Linkup API endpoint for image search
    url = "https://api.linkup.so/v1/search"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Correct payload format based on Linkup API documentation
    payload = {
        "q": query,
        "depth": "standard",
        "outputType": "searchResults",
        "includeImages": True,  # ✅ Correct parameter for image search
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()

        # Extract image URLs from response
        # Filter for results with type="image" and extract URL
        results = data.get("results", [])
        image_urls = []

        for result in results:
            # Only include results that are images
            if result.get("type") == "image":
                img_url = result.get("url")
                if img_url:
                    image_urls.append(img_url)

        st.info(f"🔍 Found {len(image_urls)} image URLs from Linkup API")

        return image_urls[:num_results]


async def _download_images(
    image_urls: List[str],
    cache_dir: Path
) -> List[str]:
    """
    Download images from URLs to local cache directory.

    Args:
        image_urls: List of image URLs
        cache_dir: Directory to save images

    Returns:
        List of local file paths
    """
    downloaded_paths = []

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        for idx, url in enumerate(image_urls):
            try:
                # Download image with proper headers
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
                }

                response = await client.get(url, headers=headers)
                response.raise_for_status()

                # Validate content type is actually an image
                content_type = response.headers.get("content-type", "").lower()

                if not any(img_type in content_type for img_type in ["image/", "jpeg", "jpg", "png", "webp", "gif"]):
                    st.warning(f"⚠️ Skipping non-image URL {idx + 1}: {content_type}")
                    continue

                # Validate content is not HTML
                content_preview = response.content[:100].lower()
                if b'<!doctype' in content_preview or b'<html' in content_preview:
                    st.warning(f"⚠️ Skipping HTML page {idx + 1}")
                    continue

                # Validate minimum file size (avoid tiny/broken images)
                if len(response.content) < 1024:  # Less than 1KB
                    st.warning(f"⚠️ Skipping tiny file {idx + 1} ({len(response.content)} bytes)")
                    continue

                # Determine file extension from content type
                if "jpeg" in content_type or "jpg" in content_type:
                    ext = "jpg"
                elif "png" in content_type:
                    ext = "png"
                elif "webp" in content_type:
                    ext = "webp"
                elif "gif" in content_type:
                    ext = "gif"
                else:
                    ext = "jpg"  # Default

                # Generate filename
                filename = f"image_{len(downloaded_paths) + 1:03d}.{ext}"
                filepath = cache_dir / filename

                # Save image
                filepath.write_bytes(response.content)

                # Validate saved image can be opened
                try:
                    from PIL import Image
                    with Image.open(filepath) as img:
                        # Verify it's a valid image
                        img.verify()
                    downloaded_paths.append(str(filepath))
                except Exception as img_error:
                    st.warning(f"⚠️ Invalid image file {idx + 1}: {str(img_error)}")
                    filepath.unlink()  # Delete invalid file
                    continue

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    st.warning(f"⚠️ Access denied for image {idx + 1} (403 Forbidden)")
                else:
                    st.warning(f"⚠️ HTTP error {e.response.status_code} for image {idx + 1}")
                continue
            except Exception as e:
                st.warning(f"⚠️ Failed to download image {idx + 1}: {str(e)}")
                continue

    return downloaded_paths


def get_cached_images_for_preset(preset_name: str) -> List[str]:
    """
    Get cached images for a specific preset.
    
    Args:
        preset_name: Name of preset
    
    Returns:
        List of cached image paths
    """
    cache_dir = _get_preset_cache_dir(preset_name)
    
    images = (
        list(cache_dir.glob("*.png")) +
        list(cache_dir.glob("*.jpg")) +
        list(cache_dir.glob("*.jpeg")) +
        list(cache_dir.glob("*.webp"))
    )
    
    return [str(img) for img in images]


def clear_preset_cache(preset_name: str) -> None:
    """
    Clear cached images for a specific preset.
    
    Args:
        preset_name: Name of preset
    """
    cache_dir = _get_preset_cache_dir(preset_name)
    
    for img in cache_dir.glob("*"):
        if img.is_file():
            img.unlink()
    
    st.success(f"✅ Cleared cache for preset: {preset_name}")


def get_cache_info(preset_name: str) -> Dict[str, Any]:
    """
    Get information about cached images for a preset.
    
    Args:
        preset_name: Name of preset
    
    Returns:
        Dictionary with cache info
    """
    cache_dir = _get_preset_cache_dir(preset_name)
    
    images = get_cached_images_for_preset(preset_name)
    
    total_size = sum(Path(img).stat().st_size for img in images)
    
    return {
        "preset_name": preset_name,
        "cache_dir": str(cache_dir),
        "num_images": len(images),
        "total_size_mb": total_size / (1024 * 1024),
        "images": images
    }

