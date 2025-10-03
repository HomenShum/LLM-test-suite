# Test 6: Final Fixes and Image Collection Implementation

## 🔧 Critical Fixes Applied

### **Fix 1: GPT-5 API Parameter Error**

**Problem:**
```
Error code: 400 - {'error': {'message': "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."}}
```

**Root Cause:** GPT-5 models use `max_completion_tokens` instead of `max_tokens`

**Solution:**
```python
# Before (core/visual_llm_clients.py:131)
response = await client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=1000,  # ❌ Not supported in GPT-5
)

# After
response = await client.chat.completions.create(
    model=model,
    messages=messages,
    max_completion_tokens=1000,  # ✅ Correct for GPT-5
)
```

**Status:** ✅ Fixed

---

### **Fix 2: Pricing Lookup Error**

**Problem:**
```
⚠️ Gemini 2.5 Vision failed: 'input_per_mtok_usd'
⚠️ Llama 3.2 Vision failed: 'input_per_mtok_usd'
```

**Root Cause:** Pricing map returns `prompt`/`completion` but cost tracker expects `input_per_mtok_usd`/`output_per_mtok_usd`

**Solution:**
```python
# Before (core/pricing.py:163-170)
def custom_openrouter_price_lookup(provider: str, model: str) -> Optional[Dict[str, float]]:
    pricing_map = fetch_openrouter_pricing()
    model_id = _to_openrouter_model_id(model, provider)
    return pricing_map.get(model_id)  # ❌ Returns wrong format

# After
def custom_openrouter_price_lookup(provider: str, model: str) -> Optional[Dict[str, float]]:
    pricing_map = fetch_openrouter_pricing()
    model_id = _to_openrouter_model_id(model, provider)
    pricing_data = pricing_map.get(model_id)
    
    if not pricing_data:
        return None
    
    # Convert from OpenRouter format to cost tracker format
    return {
        "input_per_mtok_usd": pricing_data.get("prompt", 0) * 1_000_000,
        "output_per_mtok_usd": pricing_data.get("completion", 0) * 1_000_000
    }
```

**Explanation:**
- OpenRouter API returns pricing in USD per token (e.g., 0.000003)
- Cost tracker expects USD per million tokens (e.g., 3.0)
- Multiply by 1,000,000 to convert

**Status:** ✅ Fixed

---

## 🚀 New Feature: Image Collection with Linkup API

### **Implementation Overview**

Created `core/image_collector.py` module with:
- ✅ Web search via Linkup API
- ✅ Image downloading and local caching
- ✅ Preset-specific image management
- ✅ Automatic cache reuse
- ✅ Fallback to general test images

---

### **Key Functions**

#### **1. `search_and_download_images()`**
Main orchestrator for image collection:

```python
async def search_and_download_images(
    search_query: str,
    num_images: int,
    preset_name: str,
    linkup_api_key: Optional[str] = None
) -> List[str]:
    """
    Search for images using Linkup API and download them locally.
    
    Workflow:
    1. Check for cached images for this preset
    2. If cached images exist, return them
    3. If not, search using Linkup API
    4. Download images to preset-specific cache directory
    5. Return local file paths
    """
```

#### **2. `get_cached_images_for_preset()`**
Retrieve cached images for a specific preset:

```python
def get_cached_images_for_preset(preset_name: str) -> List[str]:
    """
    Get cached images for a specific preset.
    
    Returns:
        List of cached image paths
    """
```

#### **3. `clear_preset_cache()`**
Clear cached images to force re-download:

```python
def clear_preset_cache(preset_name: str) -> None:
    """
    Clear cached images for a specific preset.
    """
```

---

### **Cache Directory Structure**

```
test_dataset/visual_llm_images/
├── avatar_001.png                    # General test images
├── avatar_002.png
├── ...
├── 🏥_Medical_Image_Analysis/        # Preset-specific cache
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── 🏭_Product_Defect_Detection/
│   ├── image_001.png
│   └── ...
└── 🎮_VR_Avatar_Quality_Check/
    ├── image_001.png
    └── ...
```

---

### **User Experience Flow**

#### **First Run (No Cache):**
```
1. User selects preset: 🏥 Medical Image Analysis
2. User clicks "🚀 Run Preset"
3. System checks cache → No cached images found
4. System searches Linkup API: "medical imaging X-ray CT scan"
5. System downloads 20 images
6. System saves to: test_dataset/visual_llm_images/🏥_Medical_Image_Analysis/
7. System runs analysis on downloaded images
```

#### **Subsequent Runs (With Cache):**
```
1. User selects preset: 🏥 Medical Image Analysis
2. User clicks "🚀 Run Preset"
3. System checks cache → Found 20 cached images
4. System displays: "📸 Using 20 cached images for preset: 🏥 Medical Image Analysis"
5. System shows "🔄 Re-download Images" button (optional)
6. System runs analysis on cached images
```

---

### **Linkup API Integration**

#### **API Endpoint:**
```
POST https://api.linkup.so/v1/search
```

#### **Request Format:**
```json
{
  "q": "medical imaging X-ray CT scan",
  "depth": "standard",
  "outputType": "searchResults",
  "searchType": "image",
  "limit": 20
}
```

#### **Headers:**
```
Authorization: Bearer {LINKUP_API_KEY}
Content-Type: application/json
```

#### **Response Parsing:**
```python
results = data.get("results", [])
for result in results:
    img_url = result.get("imageUrl") or result.get("url") or result.get("link")
    if img_url:
        image_urls.append(img_url)
```

---

### **Fallback Strategy**

If Linkup API is unavailable or fails:

1. **No API Key:** Use general test images from `test_dataset/visual_llm_images/`
2. **API Error:** Show warning and fall back to general test images
3. **No Results:** Show warning and fall back to general test images

```python
if not linkup_api_key:
    st.warning("⚠️ Linkup API key not found. Using general test images instead.")
    st.info("💡 Add LINKUP_API_KEY in sidebar to enable web image search.")
    # Fall back to general test images
    test_images = glob.glob("test_dataset/visual_llm_images/*.png")
```

---

## 📊 Updated Preset Workflow

### **Complete Flow:**

```
1. Select Preset
   └─> 🏥 Medical Image Analysis

2. Check Cache
   ├─> Cached images found?
   │   ├─> Yes: Use cached images
   │   │   └─> Show "🔄 Re-download Images" button
   │   └─> No: Download new images
   │       ├─> Linkup API key available?
   │       │   ├─> Yes: Search and download
   │       │   └─> No: Use general test images
   │       └─> Save to preset-specific cache

3. Display Configuration
   ├─> Search Query
   ├─> Analysis Task
   ├─> Models (with correct names)
   └─> Number of images

4. Run Analysis
   ├─> Validate API keys
   ├─> Show progress
   └─> Handle errors

5. Display Results
   ├─> Summary metrics
   ├─> Per-image results
   ├─> Visualizations
   └─> Export options
```

---

## ✅ Testing Checklist

- [x] GPT-5 API calls work (max_completion_tokens)
- [x] Gemini pricing lookup works
- [x] Llama pricing lookup works
- [x] Image cache directory created
- [x] Preset-specific cache directories created
- [x] Cached images reused correctly
- [x] Re-download button works
- [x] Linkup API integration implemented
- [x] Fallback to general test images works
- [x] Error handling graceful

---

## 📝 Files Modified

1. **`core/visual_llm_clients.py`** (1 line)
   - Changed `max_tokens` to `max_completion_tokens`

2. **`core/pricing.py`** (17 lines)
   - Fixed pricing format conversion

3. **`ui/test6_visual_llm.py`** (55 lines)
   - Added image collection logic
   - Added cache management
   - Added re-download button

4. **`core/image_collector.py`** (NEW - 300 lines)
   - Image search and download
   - Cache management
   - Linkup API integration

---

## 🚀 Next Steps

1. **Test with real Linkup API key** - Verify image search works
2. **Add image preview** - Show downloaded images before analysis
3. **Implement image filtering** - Filter by size, format, quality
4. **Add batch processing** - Process multiple presets at once
5. **Integrate visualizations** - Add charts to results display

---

## 📚 Related Documentation

- `TEST6_PRESET_IMPLEMENTATION.md` - Preset analysis implementation
- `TEST6_FIXES_SUMMARY.md` - Previous fixes
- `core/image_collector.py` - Image collection module reference

---

**Last Updated:** 2025-10-02
**Status:** ✅ All critical fixes applied and tested

