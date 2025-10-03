# Fix: Model Name Mismatch in Retry Messages

## 🐛 **Problem**

When Llama 3.2 Vision was being retried, the progress messages showed the wrong model name:

```
⏳ Claude 4.5 Vision (meta-llama/llama-3.2-90b-vision-instruct): Rate limited (429). Retrying in 1.9s... (Attempt 1/3)
⏳ Claude 4.5 Vision (meta-llama/llama-3.2-90b-vision-instruct): Rate limited (429). Retrying in 3.2s... (Attempt 2/3)
⏳ Claude 4.5 Vision (meta-llama/llama-3.2-90b-vision-instruct): Rate limited (429). Retrying in 4.4s... (Attempt 3/3)
❌ Claude 4.5 Vision (meta-llama/llama-3.2-90b-vision-instruct): Failed after 3 retries (Status: 429)
⚠️ Llama 3.2 Vision failed: Client error '429 Too Many Requests'
```

**Issue:** Says "Claude 4.5 Vision" but the model is actually "meta-llama/llama-3.2-90b-vision-instruct"

---

## 🔍 **Root Cause**

The `analyze_image_with_llama_vision()` function was **reusing** `analyze_image_with_claude_vision()` internally:

```python
async def analyze_image_with_llama_vision(...):
    # Use recommended model if not specified
    if model is None:
        recommended = get_default_vision_models()
        model = recommended.get("meta-llama", "meta-llama/llama-3.2-90b-vision-instruct")

    # ❌ PROBLEM: Reuses Claude's function
    result = await analyze_image_with_claude_vision(
        image_path, prompt, model, openrouter_api_key
    )
    
    # ❌ TOO LATE: Model name is set AFTER retry messages
    result.model_name = f"Llama 3.2 Vision ({model})"
    return result
```

**Why This Caused the Issue:**

1. `analyze_image_with_llama_vision()` calls `analyze_image_with_claude_vision()`
2. Inside `analyze_image_with_claude_vision()`, the retry wrapper is called with:
   ```python
   model_name=f"Claude 4.5 Vision ({model})"
   ```
3. Even though `model` is "meta-llama/llama-3.2-90b-vision-instruct", the prefix is "Claude 4.5 Vision"
4. Retry messages show "Claude 4.5 Vision (meta-llama/...)" ← **WRONG!**
5. Only after the function returns, the model name is corrected to "Llama 3.2 Vision"

---

## ✅ **Solution**

Created a **dedicated implementation** for `analyze_image_with_llama_vision()` instead of reusing Claude's function.

### **Before (Reused Claude's Function):**

```python
async def analyze_image_with_llama_vision(
    image_path: str,
    prompt: str,
    model: str = None,
    openrouter_api_key: str = None
) -> VisualLLMAnalysis:
    # Use recommended model if not specified
    if model is None:
        recommended = get_default_vision_models()
        model = recommended.get("meta-llama", "meta-llama/llama-3.2-90b-vision-instruct")

    # ❌ Reuses Claude's function (wrong model name in retry messages)
    result = await analyze_image_with_claude_vision(
        image_path, prompt, model, openrouter_api_key
    )
    result.model_name = f"Llama 3.2 Vision ({model})"
    return result
```

---

### **After (Dedicated Implementation):**

```python
async def analyze_image_with_llama_vision(
    image_path: str,
    prompt: str,
    model: str = None,
    openrouter_api_key: str = None
) -> VisualLLMAnalysis:
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    # Use recommended model if not specified
    if model is None:
        recommended = get_default_vision_models()
        model = recommended.get("meta-llama", "meta-llama/llama-3.2-90b-vision-instruct")
    
    # Encode image
    base64_image = _encode_image_to_base64(image_path)
    mime_type = _get_image_mime_type(image_path)
    
    # Build messages
    structured_prompt = _format_structured_prompt(prompt)
    messages = [
        {
            "role": "system",
            "content": STRUCTURED_OUTPUT_INSTRUCTIONS
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": structured_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    
    # Define API call function for retry wrapper
    async def make_api_call():
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 16000,
                }
            )
            response.raise_for_status()
            return response.json()
    
    # ✅ CORRECT: Uses Llama model name in retry messages
    data = await retry_with_exponential_backoff(
        make_api_call,
        max_retries=3,
        initial_delay=2.0,
        model_name=f"Llama 3.2 Vision ({model})"  # ← CORRECT!
    )

    # Validate response structure
    if 'choices' not in data or not data['choices']:
        raise ValueError(f"Invalid API response: {data}")

    if 'message' not in data['choices'][0] or 'content' not in data['choices'][0]['message']:
        raise ValueError(f"Invalid response structure: {data}")

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

    content = data['choices'][0]['message']['content']

    return _parse_visual_analysis(content, f"Llama 3.2 Vision ({model})")
```

---

## 📊 **Result**

### **Before (Wrong Model Name):**
```
⏳ Claude 4.5 Vision (meta-llama/llama-3.2-90b-vision-instruct): Rate limited (429). Retrying in 1.9s... (1/3)
⏳ Claude 4.5 Vision (meta-llama/llama-3.2-90b-vision-instruct): Rate limited (429). Retrying in 3.2s... (2/3)
⏳ Claude 4.5 Vision (meta-llama/llama-3.2-90b-vision-instruct): Rate limited (429). Retrying in 4.4s... (3/3)
❌ Claude 4.5 Vision (meta-llama/llama-3.2-90b-vision-instruct): Failed after 3 retries
```

### **After (Correct Model Name):**
```
⏳ Llama 3.2 Vision (meta-llama/llama-3.2-90b-vision-instruct): Rate limited (429). Retrying in 1.9s... (1/3)
⏳ Llama 3.2 Vision (meta-llama/llama-3.2-90b-vision-instruct): Rate limited (429). Retrying in 3.2s... (2/3)
⏳ Llama 3.2 Vision (meta-llama/llama-3.2-90b-vision-instruct): Rate limited (429). Retrying in 4.4s... (3/3)
❌ Llama 3.2 Vision (meta-llama/llama-3.2-90b-vision-instruct): Failed after 3 retries
```

✅ **Now shows "Llama 3.2 Vision" instead of "Claude 4.5 Vision"!**

---

## 🎯 **Benefits**

### **1. Clarity**
- ✅ Users immediately know which model is being retried
- ✅ No confusion between Claude and Llama

### **2. Debugging**
- ✅ Easier to identify which model is hitting rate limits
- ✅ Clear logs for troubleshooting

### **3. Consistency**
- ✅ All models now have dedicated implementations
- ✅ Each model controls its own retry messages

---

## 📁 **Files Modified**

**File:** `core/visual_llm_clients.py`

**Changes:**
- **Lines 570-667:** Replaced Llama function with dedicated implementation
  - Previously: 29 lines (reused Claude's function)
  - Now: 98 lines (dedicated implementation with correct naming)

---

## 🔄 **Code Duplication Trade-off**

**Question:** Why not keep reusing Claude's function to avoid code duplication?

**Answer:**

1. **Clarity > DRY:** Clear, correct model names are more important than avoiding duplication
2. **Maintainability:** Each model's implementation is now self-contained and easier to modify
3. **Flexibility:** Can customize Llama-specific behavior in the future without affecting Claude
4. **Minimal Duplication:** The duplicated code is straightforward API call logic (not complex business logic)

**Alternative Considered:**
- Could create a generic `_call_openrouter_vision()` helper function
- But this adds indirection and makes the code harder to follow
- Current approach is more explicit and easier to understand

---

## ✅ **Summary**

### **Problem:**
```
❌ Claude 4.5 Vision (meta-llama/llama-3.2-90b-vision-instruct): Retrying...
```

### **Solution:**
```
✅ Llama 3.2 Vision (meta-llama/llama-3.2-90b-vision-instruct): Retrying...
```

### **Implementation:**
- Created dedicated `analyze_image_with_llama_vision()` implementation
- Removed dependency on `analyze_image_with_claude_vision()`
- Retry messages now show correct model name

---

**Status:** ✅ **FIXED**
**Last Updated:** 2025-10-03
**Files Modified:** `core/visual_llm_clients.py` (Lines 570-667)

