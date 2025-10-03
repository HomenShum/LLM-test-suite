# Cost Tracking Debug Report

## Issue
User reports that cost tracking is not working and not showing on sidebar in Test 6 (Visual LLM Testing).

---

## Investigation

### 1. Cost Tracker Initialization ✅

**Location:** `streamlit_test_v5.py` line 318-320

```python
# --- Initialize cost tracker ---
if "cost_tracker" not in st.session_state:
    st.session_state.cost_tracker = CostTracker()
    register_all_extractors()
```

**Status:** ✅ Cost tracker is properly initialized in session state

---

### 2. Sidebar Rendering ✅

**Location:** `streamlit_test_v5.py` line 351-352

```python
sidebar.configure(globals())
sidebar.render_api_sidebar()
```

**Location:** `ui/sidebar.py` line 243-300

```python
# --- COST TRACKING UI (AT BOTTOM) ---
with st.sidebar:
    st.divider()
    st.subheader("💰 Cost Tracking")

    ct = st.session_state.cost_tracker

    # Compact display - just the essentials
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Cost", f"${ct.totals['total_cost_usd']:.4f}")
    with col2:
        st.metric("Total Tokens", f"{ct.totals['total_tokens']:,}")
```

**Status:** ✅ Sidebar is properly rendering cost tracker

---

### 3. Cost Tracking Calls in Visual LLM Clients ✅

**Location:** `core/visual_llm_clients.py`

**OpenAI (line 138-146):**
```python
# Track cost
if st.session_state.get('cost_tracker'):
    from core.pricing import custom_openai_price_lookup
    st.session_state.cost_tracker.update(
        provider="OpenAI",
        model=model,
        api="chat.completions",
        raw_response_obj=response,
        pricing_resolver=custom_openai_price_lookup
    )
```

**Gemini (line 217-225):**
```python
# Track cost
if st.session_state.get('cost_tracker'):
    from core.pricing import custom_gemini_price_lookup
    st.session_state.cost_tracker.update(
        provider="Google",
        model=model,
        api="generate_content",
        raw_response_obj=response,
        pricing_resolver=custom_gemini_price_lookup
    )
```

**OpenRouter (line 312-320):**
```python
# Track cost
if st.session_state.get('cost_tracker'):
    from core.pricing import custom_openrouter_price_lookup
    st.session_state.cost_tracker.update(
        provider="OpenRouter",
        model=model,
        api="chat.completions",
        raw_response_obj=data,
        pricing_resolver=custom_openrouter_price_lookup
    )
```

**Status:** ✅ All visual LLM clients are calling cost tracker

---

### 4. Cost Tracking in Meta-Analysis ✅ (FIXED)

**Location:** `core/visual_meta_analysis.py`

**Before (WRONG):**
```python
from utils.cost_tracker import custom_gemini_price_lookup  # ❌ Wrong import
```

**After (FIXED):**
```python
from core.pricing import custom_gemini_price_lookup  # ✅ Correct import
```

**Status:** ✅ Fixed import path in lines 111 and 183

---

### 5. Pricing Lookup Functions ✅

**Location:** `core/pricing.py`

**Test:**
```bash
$ python -c "from core.pricing import custom_gemini_price_lookup; result = custom_gemini_price_lookup('Google', 'gemini-2.5-flash'); print(f'Result: {result}')"

Result: {'input_per_mtok_usd': 0.3, 'output_per_mtok_usd': 2.5}
```

**Status:** ✅ Pricing lookup is working correctly

---

### 6. Pricing Cache ✅

**Location:** `pricing_cache/openrouter_pricing.json`

**Relevant entries:**
```json
{
  "openai/gpt-5-mini": {
    "prompt": 2.5e-07,
    "completion": 2e-06
  },
  "google/gemini-2.5-flash": {
    "prompt": 3e-07,
    "completion": 2.5e-06
  }
}
```

**Status:** ✅ Pricing cache has correct data

---

### 7. Usage Extractors ✅

**Location:** `cost_tracker/extractors.py`

**Gemini Extractor (line 96-117):**
```python
def gemini_extractor(provider, model, raw_obj, raw_json):
    """
    Extract usage from Google Gemini API response.
    Gemini objects: usage_metadata.promptTokenCount / candidatesTokenCount / totalTokenCount
    """
    um = getattr(raw_obj, "usage_metadata", None)
    if um is not None:
        pt = int(getattr(um, "promptTokenCount", 0) or getattr(um, "prompt_token_count", 0) or 0)
        ct = int(getattr(um, "candidatesTokenCount", 0) or getattr(um, "candidates_token_count", 0) or 0)
        tt = int(getattr(um, "totalTokenCount", 0) or getattr(um, "total_token_count", 0) or (pt + ct))
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}
    
    # Try JSON format
    if isinstance(raw_json, dict):
        um = raw_json.get("usage_metadata") or raw_json.get("usageMetadata") or {}
        pt = int(um.get("promptTokenCount", 0) or um.get("prompt_token_count", 0) or 0)
        ct = int(um.get("candidatesTokenCount", 0) or um.get("candidates_token_count", 0) or 0)
        tt = int(um.get("totalTokenCount", 0) or um.get("total_token_count", 0) or (pt + ct))
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}
    
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
```

**Registration (line 175-189):**
```python
def register_all_extractors():
    """Register all built-in extractors."""
    register_usage_extractor("Anthropic", anthropic_extractor)
    register_usage_extractor("Cohere", cohere_extractor)
    register_usage_extractor("Groq", groq_extractor)
    register_usage_extractor("Mistral", mistral_extractor)
    register_usage_extractor("OpenAI", openai_extractor)
    register_usage_extractor("Google", gemini_extractor)  # ✅
    register_usage_extractor("Gemini", gemini_extractor)  # ✅
    register_usage_extractor("OpenRouter", openrouter_extractor)
    register_usage_extractor("Together", together_extractor)
    register_usage_extractor("Perplexity", perplexity_extractor)
```

**Status:** ✅ Gemini extractor is registered for both "Google" and "Gemini" providers

---

## Potential Issues

### Issue 1: Streamlit Rerun Behavior

**Problem:** Streamlit reruns the entire script on every interaction, which could cause the cost tracker to appear empty if:
1. The cost tracker is being reset
2. The API calls haven't been made yet in the current run
3. The sidebar is rendered before the API calls are made

**Solution:** The cost tracker is stored in `st.session_state`, which persists across reruns. This should work correctly.

---

### Issue 2: Async API Calls

**Problem:** The visual LLM API calls are async, and the sidebar might be rendered before the async calls complete.

**Current Flow:**
```
1. Sidebar renders (shows $0.00)
2. User clicks "Run Preset"
3. Async API calls execute
4. Cost tracker updates
5. Results display
6. Sidebar still shows old values (needs rerun)
```

**Solution:** After API calls complete, the app needs to trigger a rerun to update the sidebar.

---

### Issue 3: Missing Rerun After Analysis

**Problem:** The cost tracker updates during analysis, but the sidebar doesn't refresh until the next manual interaction.

**Check:** Look for `st.rerun()` calls after analysis completes.

**Location to check:** `ui/test6_visual_llm.py` after analysis completes

---

## Recommended Fixes

### Fix 1: Add Rerun After Analysis ✅

**Location:** `ui/test6_visual_llm.py` (after analysis completes)

**Add:**
```python
# After saving to history
st.success("✅ Analysis complete!")
st.rerun()  # Force rerun to update sidebar
```

---

### Fix 2: Show Cost in Results Tab

**Instead of relying on sidebar, show cost directly in results:**

```python
# In results display
if st.session_state.get('cost_tracker'):
    ct = st.session_state.cost_tracker
    st.metric("Analysis Cost", f"${ct.totals['total_cost_usd']:.4f}")
```

---

### Fix 3: Debug Display

**Add debug info to see if cost tracker is being updated:**

```python
# In sidebar or results
if st.session_state.get('cost_tracker'):
    ct = st.session_state.cost_tracker
    st.write(f"Debug: {len(ct.by_call)} API calls tracked")
    st.write(f"Debug: Total cost: ${ct.totals['total_cost_usd']:.4f}")
```

---

## Files Modified

1. ✅ `core/visual_meta_analysis.py` - Fixed import paths (line 111, 183)
   - Changed `from utils.cost_tracker import custom_gemini_price_lookup`
   - To `from core.pricing import custom_gemini_price_lookup`

---

## Next Steps

1. ✅ Test if cost tracking works after fixing import paths
2. ⏳ Add `st.rerun()` after analysis completes to update sidebar
3. ⏳ Add cost display in results tab for immediate feedback
4. ⏳ Add debug display to verify cost tracker is being updated

---

## Testing

**To test:**
1. Run Test 6 with a preset
2. Check sidebar for cost updates
3. Check browser console for errors
4. Check if `st.session_state.cost_tracker.by_call` has entries
5. Check if `st.session_state.cost_tracker.totals` has non-zero values

**Expected behavior:**
- Sidebar shows $0.00 before analysis
- During analysis, cost tracker updates
- After analysis, sidebar shows total cost (may need manual refresh or rerun)
- Results tab shows cost immediately

---

**Status:** Import paths fixed. Need to test and potentially add rerun trigger.

