# Test 6: Image Validation and Download Fixes

## 🔧 Critical Fixes Applied

### **Fix 1: Invalid Image Format Errors**

**Problem:**
```
⚠️ GPT-5 Vision failed: Error code: 400 - "You uploaded an unsupported image. Please make sure your image has of one the following formats: ['png', 'jpeg', 'gif', 'webp']."

⚠️ Gemini 2.5 Vision failed: 400 INVALID_ARGUMENT. Unable to process input image.
```

**Root Cause:** 
Downloaded files were not actual images - they were HTML pages, redirects, or corrupted files.

**Solution:**
Added comprehensive image validation in `core/image_collector.py`:

```python
async def _download_images(image_urls: List[str], cache_dir: Path) -> List[str]:
    """Download and validate images."""
    
    for idx, url in enumerate(image_urls):
        # 1. Validate content type
        content_type = response.headers.get("content-type", "").lower()
        if not any(img_type in content_type for img_type in ["image/", "jpeg", "jpg", "png", "webp", "gif"]):
            st.warning(f"⚠️ Skipping non-image URL {idx + 1}: {content_type}")
            continue
        
        # 2. Validate content is not HTML
        content_preview = response.content[:100].lower()
        if b'<!doctype' in content_preview or b'<html' in content_preview:
            st.warning(f"⚠️ Skipping HTML page {idx + 1}")
            continue
        
        # 3. Validate minimum file size
        if len(response.content) < 1024:  # Less than 1KB
            st.warning(f"⚠️ Skipping tiny file {idx + 1} ({len(response.content)} bytes)")
            continue
        
        # 4. Validate with PIL after saving
        try:
            from PIL import Image
            with Image.open(filepath) as img:
                img.verify()  # Verify it's a valid image
            downloaded_paths.append(str(filepath))
        except Exception as img_error:
            st.warning(f"⚠️ Invalid image file {idx + 1}: {str(img_error)}")
            filepath.unlink()  # Delete invalid file
            continue
```

**Validation Steps:**
1. ✅ Check HTTP Content-Type header
2. ✅ Detect HTML content (not images)
3. ✅ Validate minimum file size (1KB)
4. ✅ Use PIL to verify image integrity
5. ✅ Delete invalid files automatically

---

### **Fix 2: 403 Forbidden Errors**

**Problem:**
```
⚠️ Failed to download image 15: Client error '403 Forbidden'
⚠️ Failed to download image 16: Client error '403 Forbidden'
```

**Root Cause:** 
Some websites block requests without proper User-Agent headers.

**Solution:**
Added proper HTTP headers to mimic browser requests:

```python
# Download with proper headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
}

response = await client.get(url, headers=headers)
```

**Also added:**
- ✅ `follow_redirects=True` to handle redirects
- ✅ Better error messages for different HTTP status codes
- ✅ Continue on error (don't fail entire batch)

---

### **Fix 3: Llama 3.2 Vision 'choices' Error**

**Problem:**
```
⚠️ Llama 3.2 Vision failed: 'choices'
```

**Root Cause:** 
API response didn't contain expected 'choices' field, but code didn't validate response structure.

**Solution:**
Added response validation in `core/visual_llm_clients.py`:

```python
# Call OpenRouter API
response = await client.post(...)
data = response.json()

# Validate response structure
if 'choices' not in data or not data['choices']:
    raise ValueError(f"Invalid API response: {data}")

if 'message' not in data['choices'][0] or 'content' not in data['choices'][0]['message']:
    raise ValueError(f"Invalid response structure: {data}")

# Now safe to access
content = data['choices'][0]['message']['content']
```

**Validation Steps:**
1. ✅ Check 'choices' exists and is not empty
2. ✅ Check 'message' exists in first choice
3. ✅ Check 'content' exists in message
4. ✅ Raise descriptive error with full response

---

## 📊 Improved Download Statistics

### **Before:**
```
📥 Downloading 20 images...
⚠️ Failed to download image 15: Client error '403 Forbidden'
⚠️ Failed to download image 16: Client error '403 Forbidden'
⚠️ Failed to download image 19: Client error '403 Forbidden'
✅ Downloaded 17 images
```

### **After:**
```
📥 Downloading 20 images...
⚠️ Access denied for image 15 (403 Forbidden)
⚠️ Access denied for image 16 (403 Forbidden)
⚠️ Skipping HTML page 17
⚠️ Invalid image file 18: cannot identify image file
⚠️ Access denied for image 19 (403 Forbidden)
⚠️ Skipping tiny file 20 (512 bytes)
✅ Downloaded 14 valid images to test_dataset\visual_llm_images\🏥_Medical_Image_Analysis
```

**Improvements:**
- ✅ More descriptive error messages
- ✅ Categorized errors (403, HTML, invalid, tiny)
- ✅ Only counts valid images
- ✅ Shows final destination path

---

## 🔍 Image Validation Flow

```
Download Image URL
    ↓
Check HTTP Status
    ├─> 403 Forbidden → Skip with warning
    ├─> 404 Not Found → Skip with warning
    └─> 200 OK → Continue
        ↓
Check Content-Type Header
    ├─> Not image/* → Skip (HTML/JSON/etc)
    └─> image/* → Continue
        ↓
Check Content Preview
    ├─> Contains HTML tags → Skip
    └─> Binary data → Continue
        ↓
Check File Size
    ├─> < 1KB → Skip (too small)
    └─> >= 1KB → Continue
        ↓
Save to Disk
    ↓
Validate with PIL
    ├─> Invalid → Delete file, skip
    └─> Valid → Add to results
        ↓
Return Valid Image Path
```

---

## 🛡️ Error Handling Strategy

### **Graceful Degradation:**

1. **Individual Image Failures:**
   - ✅ Log warning
   - ✅ Continue with next image
   - ✅ Don't fail entire batch

2. **All Images Failed:**
   - ✅ Fall back to general test images
   - ✅ Show helpful error message
   - ✅ Allow analysis to continue

3. **API Response Errors:**
   - ✅ Validate response structure
   - ✅ Show full error details
   - ✅ Skip failed model, continue with others

### **Example:**
```python
try:
    # Download and validate image
    downloaded_paths.append(validate_and_save(url))
except httpx.HTTPStatusError as e:
    if e.response.status_code == 403:
        st.warning(f"⚠️ Access denied for image {idx + 1} (403 Forbidden)")
    else:
        st.warning(f"⚠️ HTTP error {e.response.status_code} for image {idx + 1}")
    continue  # Don't fail entire batch
except Exception as e:
    st.warning(f"⚠️ Failed to download image {idx + 1}: {str(e)}")
    continue
```

---

## 📝 Files Modified

| File | Changes | Description |
|------|---------|-------------|
| `core/image_collector.py` | ~90 lines | Added image validation |
| `core/visual_llm_clients.py` | 7 lines | Added response validation |

---

## ✅ Testing Checklist

- [x] Download valid images (PNG, JPEG, GIF, WebP)
- [x] Skip HTML pages
- [x] Skip tiny files (< 1KB)
- [x] Handle 403 Forbidden errors
- [x] Handle 404 Not Found errors
- [x] Validate images with PIL
- [x] Delete invalid files
- [x] Continue on individual failures
- [x] Fall back to general test images
- [x] Validate API responses
- [x] Handle missing 'choices' field

---

## 🚀 Expected Behavior

### **Successful Download:**
```
🔍 Searching for images: 'medical imaging X-ray CT scan'...
📥 Downloading 20 images...
⚠️ Access denied for image 3 (403 Forbidden)
⚠️ Skipping HTML page 7
⚠️ Invalid image file 12: cannot identify image file
✅ Downloaded 17 valid images to test_dataset\visual_llm_images\🏥_Medical_Image_Analysis

🔄 Running Analysis...
Analyzing image 1/17: image_001.jpg
✅ GPT-5 Vision: Success
✅ Gemini 2.5 Vision: Success
✅ Llama 3.2 Vision: Success

Analyzing image 2/17: image_002.png
✅ GPT-5 Vision: Success
✅ Gemini 2.5 Vision: Success
✅ Llama 3.2 Vision: Success
...
```

### **All Downloads Failed:**
```
🔍 Searching for images: 'medical imaging X-ray CT scan'...
📥 Downloading 20 images...
⚠️ Access denied for image 1 (403 Forbidden)
⚠️ Access denied for image 2 (403 Forbidden)
...
⚠️ No valid images downloaded. Using general test images instead.
📸 Using 9 test images from test_dataset/visual_llm_images/

🔄 Running Analysis...
```

---

## 📚 Related Documentation

- `TEST6_PYDANTIC_AND_API_FIXES.md` - Pydantic and API fixes
- `TEST6_FINAL_FIXES.md` - Initial API and pricing fixes
- `core/image_collector.py` - Image collection module

---

**Last Updated:** 2025-10-02
**Status:** ✅ All image validation fixes applied and tested

