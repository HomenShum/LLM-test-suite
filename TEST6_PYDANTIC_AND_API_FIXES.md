# Test 6: Pydantic Model and API Fixes

## 🔧 Critical Fixes Applied

### **Fix 1: Pydantic Model Attribute Access Error**

**Problem:**
```
AttributeError: 'VisualLLMAnalysis' object has no attribute 'get'
```

**Root Cause:** 
The code was treating Pydantic `VisualLLMAnalysis` objects as dictionaries using `.get()` method, but Pydantic models use attribute access.

**Solution:**

#### **Before (Incorrect):**
```python
# Trying to use .get() on Pydantic model
total_tokens += analysis.get("tokens_used", 0)  # ❌ AttributeError
total_cost += analysis.get("cost", 0)  # ❌ AttributeError
st.info(analysis.get("response_text", "No response"))  # ❌ AttributeError
```

#### **After (Correct):**
```python
# Use attribute access for Pydantic models
if hasattr(analysis, 'rationale'):
    st.info(analysis.rationale)  # ✅ Correct
elif hasattr(analysis, 'raw_response'):
    st.info(analysis.raw_response)  # ✅ Correct

# Use getattr with default
confidence = getattr(analysis, 'confidence', 0.0)  # ✅ Correct
```

**Files Modified:**
- `ui/test6_visual_llm.py` - Lines 509-540 (Model Performance table)
- `ui/test6_visual_llm.py` - Lines 542-569 (Image Results display)
- `ui/test6_visual_llm.py` - Lines 577-643 (Export functionality)

---

### **Fix 2: GPT-5 Temperature Parameter Error**

**Problem:**
```
Error code: 400 - {'error': {'message': "Unsupported value: 'temperature' does not support 0.3 with this model. Only the default (1) value is supported."}}
```

**Root Cause:** 
GPT-5 models only support `temperature=1` (default) and don't allow custom temperature values.

**Solution:**

#### **Before:**
```python
response = await client.chat.completions.create(
    model=model,
    messages=messages,
    max_completion_tokens=1000,
    temperature=0.3  # ❌ Not supported in GPT-5
)
```

#### **After:**
```python
response = await client.chat.completions.create(
    model=model,
    messages=messages,
    max_completion_tokens=1000
    # temperature not supported in GPT-5, uses default of 1
)
```

**File Modified:**
- `core/visual_llm_clients.py` - Line 127-135

---

### **Fix 3: Linkup API Key Not Found**

**Problem:**
```
⚠️ Linkup API key not found. Using general test images instead.
```

**Root Cause:** 
The code was only checking `_CONFIG` dict, but the API key was stored in `st.secrets`.

**Solution:**

#### **Before:**
```python
# Only checking _CONFIG
linkup_key = _CONFIG.get('LINKUP_API_KEY')
```

#### **After:**
```python
# Check multiple sources
linkup_key = None

# Try st.secrets first
if hasattr(st, 'secrets') and 'LINKUP_API_KEY' in st.secrets:
    linkup_key = st.secrets['LINKUP_API_KEY']
# Fall back to _CONFIG
elif 'LINKUP_API_KEY' in _CONFIG:
    linkup_key = _CONFIG['LINKUP_API_KEY']
```

**File Modified:**
- `ui/test6_visual_llm.py` - Lines 687-714

---

## 📊 Updated Model Performance Display

### **New Implementation:**

```python
# Model comparison table
model_data = []
for model_id in selected_models:
    success_count = 0
    
    for result in results:
        model_results = result.get("model_results", {})
        for model_name, analysis in model_results.items():
            if model_id in model_name.lower():
                success_count += 1
    
    # Get cost data from session state if available
    total_cost = 0.0
    total_tokens = 0
    if 'cost_tracker' in st.session_state:
        cost_tracker = st.session_state.cost_tracker
        for record in cost_tracker.records:
            if model_id.upper() in record.get("model", "").upper():
                total_cost += record.get("cost_usd", 0)
                total_tokens += record.get("total_tokens", 0)
    
    model_data.append({
        "Model": model_id.upper(),
        "Successful": success_count,
        "Total Tokens": f"{total_tokens:,}" if total_tokens > 0 else "N/A",
        "Total Cost": f"${total_cost:.4f}" if total_cost > 0 else "N/A"
    })
```

**Key Changes:**
- ✅ Removed `.get()` calls on Pydantic models
- ✅ Get cost/token data from `st.session_state.cost_tracker` instead
- ✅ Show "N/A" when data not available
- ✅ Proper error handling with `hasattr()` and `getattr()`

---

## 📝 Updated Image Results Display

### **New Implementation:**

```python
# Show model responses
for model_name, analysis in result.get("model_results", {}).items():
    st.markdown(f"**{model_name}:**")
    
    # Access Pydantic model attributes correctly
    if hasattr(analysis, 'rationale'):
        st.info(analysis.rationale)
    elif hasattr(analysis, 'raw_response'):
        st.info(analysis.raw_response)
    else:
        st.info("No response available")
    
    # Show confidence if available
    if hasattr(analysis, 'confidence'):
        st.caption(f"Confidence: {analysis.confidence:.2%}")
```

**Key Changes:**
- ✅ Use `hasattr()` to check for attributes
- ✅ Access attributes directly (not via `.get()`)
- ✅ Graceful fallback when attributes missing
- ✅ Display confidence as percentage

---

## 💾 Updated Export Functionality

### **JSON Export:**

```python
# Convert Pydantic models to dicts for JSON serialization
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
```

**Key Changes:**
- ✅ Use `model_dump()` (Pydantic v2) or `dict()` (Pydantic v1)
- ✅ Proper JSON serialization of Pydantic models
- ✅ Fallback to string representation if needed

### **CSV Export:**

```python
# Access Pydantic model attributes for CSV
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
```

**Key Changes:**
- ✅ Use `getattr()` with defaults
- ✅ Proper attribute access for CSV columns
- ✅ Graceful handling of missing attributes

---

## 🔑 API Key Priority Order

The system now checks for API keys in this order:

1. **`st.secrets`** (Streamlit secrets.toml) - **Highest priority**
2. **`_CONFIG`** (Session state config) - Fallback
3. **Environment variables** - Not currently checked (could be added)

```python
# Priority order for Linkup API key
linkup_key = None

# 1. Check st.secrets first
if hasattr(st, 'secrets') and 'LINKUP_API_KEY' in st.secrets:
    linkup_key = st.secrets['LINKUP_API_KEY']

# 2. Fall back to _CONFIG
elif 'LINKUP_API_KEY' in _CONFIG:
    linkup_key = _CONFIG['LINKUP_API_KEY']

# 3. Could add os.environ check here if needed
```

---

## ✅ Testing Checklist

- [x] Pydantic model attribute access works
- [x] Model performance table displays correctly
- [x] Image results display correctly
- [x] Export to JSON works
- [x] Export to CSV works
- [x] GPT-5 API calls work (no temperature error)
- [x] Linkup API key found from st.secrets
- [x] Cost tracking displays correctly
- [x] Confidence scores display correctly

---

## 📝 Files Modified

| File | Lines Changed | Description |
|------|---------------|-------------|
| `core/visual_llm_clients.py` | 1 line | Removed temperature parameter |
| `ui/test6_visual_llm.py` | ~100 lines | Fixed Pydantic access, API key lookup |

---

## 🚀 Next Steps

1. **Test with real Linkup API** - Verify image search works
2. **Add more robust error handling** - Handle API failures gracefully
3. **Implement caching** - Cache analysis results to avoid re-running
4. **Add visualizations** - Integrate charts from `core/vision_visualizations.py`
5. **Improve cost tracking** - Better integration with cost tracker

---

## 📚 Related Documentation

- `TEST6_FINAL_FIXES.md` - Previous API and pricing fixes
- `TEST6_PRESET_IMPLEMENTATION.md` - Preset analysis implementation
- `core/models.py` - Pydantic model definitions

---

**Last Updated:** 2025-10-02
**Status:** ✅ All critical fixes applied and tested

