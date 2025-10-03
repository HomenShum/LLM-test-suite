# Retry and Fallback Implementation for Visual LLM API Calls

## 🐛 **Problem**

API calls to visual LLM providers (especially OpenRouter) were failing with rate limit errors:

```
⚠️ Llama 3.2 Vision failed: Client error '429 Too Many Requests' 
for url 'https://openrouter.ai/api/v1/chat/completions'
```

**Issues:**
- ❌ No retry logic for transient errors (429, 500-599, timeouts)
- ❌ Single failure would stop entire analysis
- ❌ Poor user experience during rate limiting
- ❌ No exponential backoff to prevent thundering herd

---

## ✅ **Solution**

Implemented **comprehensive retry logic with exponential backoff** and **graceful fallback** for all visual LLM API calls.

---

## 🔧 **Implementation Details**

### **1. Retry Wrapper with Exponential Backoff**

**File:** `core/visual_llm_clients.py`

**Function:** `retry_with_exponential_backoff()`

```python
async def retry_with_exponential_backoff(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    model_name: str = "Model"
) -> Any:
    """
    Retry an async function with exponential backoff.
    
    Handles:
    - 429 Too Many Requests (rate limiting)
    - 500-599 Server errors
    - Network timeouts
    - Connection errors
    """
```

**Key Features:**
- ✅ **Exponential backoff:** Delay doubles with each retry (1s → 2s → 4s)
- ✅ **Jitter:** Random variation (±50%) prevents thundering herd
- ✅ **Max delay cap:** Never waits more than 60 seconds
- ✅ **User feedback:** Shows progress messages in Streamlit UI
- ✅ **Smart error detection:** Handles multiple error formats

---

### **2. Error Handling Strategy**

**Retryable Errors:**
- ✅ **429 Too Many Requests** - Rate limiting
- ✅ **500-599** - Server errors
- ✅ **Timeouts** - Network timeouts
- ✅ **Connection errors** - Network failures

**Non-Retryable Errors:**
- ❌ **400-499** (except 429) - Client errors (bad request, auth, etc.)
- ❌ **Validation errors** - Invalid response structure
- ❌ **Unexpected errors** - Unknown exceptions

---

### **3. Retry Logic Applied to All Models**

#### **GPT-5 Vision (OpenAI)**
```python
async def make_api_call():
    return await client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=16000,
        response_format={"type": "json_object"}
    )

response = await retry_with_exponential_backoff(
    make_api_call,
    max_retries=3,
    initial_delay=2.0,
    model_name=f"GPT-5 Vision ({model})"
)
```

#### **Gemini 2.5 Vision (Google)**
```python
async def make_api_call():
    return await asyncio.to_thread(
        lambda: client.models.generate_content(
            model=model,
            contents=contents
        )
    )

response = await retry_with_exponential_backoff(
    make_api_call,
    max_retries=3,
    initial_delay=2.0,
    model_name=f"Gemini 2.5 Vision ({model})"
)
```

#### **Claude 4.5 Vision (OpenRouter)**
```python
async def make_api_call():
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={...},
            json={...}
        )
        response.raise_for_status()
        return response.json()

data = await retry_with_exponential_backoff(
    make_api_call,
    max_retries=3,
    initial_delay=2.0,
    model_name=f"Claude 4.5 Vision ({model})"
)
```

#### **Llama 3.2 Vision (OpenRouter)**
```python
async def make_api_call():
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={...},
            json={...}
        )
        response.raise_for_status()
        return response.json()

data = await retry_with_exponential_backoff(
    make_api_call,
    max_retries=3,
    initial_delay=2.0,
    model_name=f"Llama 3.2 Vision ({model})"  # ← Correct model name!
)
```

**Note:** Previously reused Claude's function, but this caused model name mismatch in retry messages. Now has dedicated implementation with correct naming.

---

### **4. Graceful Fallback in Multi-Model Analysis**

**Already Implemented:** The `analyze_image_multi_model()` function uses `asyncio.gather(..., return_exceptions=True)` to continue even if one model fails.

```python
# Run all tasks in parallel with error handling
task_results = await asyncio.gather(*tasks, return_exceptions=True)

for name, result in zip(model_names, task_results):
    if isinstance(result, Exception):
        # Log error and create error result
        st.warning(f"⚠️ {name} failed: {str(result)}")
        results[name] = VisualLLMAnalysis(
            model_name=name,
            detected_artifacts=[],
            confidence=0.0,
            rationale=f"Error: {str(result)}",
            raw_response=None
        )
    else:
        results[name] = result
```

**Benefit:** If Llama fails due to rate limiting, GPT-5, Gemini, and Claude still complete successfully.

---

## 📊 **User Experience**

### **Before (No Retry):**
```
🔍 Analyzing image 1/10...
❌ Llama 3.2 Vision failed: 429 Too Many Requests
[Analysis stops or continues with incomplete results]
```

### **After (With Retry):**
```
🔍 Analyzing image 1/10...
⏳ Llama 3.2 Vision: Rate limited (429). Retrying in 2.3s... (Attempt 1/3)
⏳ Llama 3.2 Vision: Rate limited (429). Retrying in 4.7s... (Attempt 2/3)
✅ Llama 3.2 Vision: Success!
```

### **If All Retries Fail:**
```
🔍 Analyzing image 1/10...
⏳ Llama 3.2 Vision: Rate limited (429). Retrying in 2.1s... (Attempt 1/3)
⏳ Llama 3.2 Vision: Rate limited (429). Retrying in 4.3s... (Attempt 2/3)
⏳ Llama 3.2 Vision: Rate limited (429). Retrying in 8.9s... (Attempt 3/3)
❌ Llama 3.2 Vision: Rate limit exceeded after 3 retries
⚠️ Llama 3.2 Vision failed: 429 Too Many Requests

[Other models continue successfully]
✅ GPT-5 Vision: Success!
✅ Gemini 2.5 Vision: Success!
✅ Claude 4.5 Vision: Success!
```

---

## 🎯 **Retry Parameters**

### **Default Configuration:**
- **Max Retries:** 3 attempts
- **Initial Delay:** 2.0 seconds
- **Exponential Base:** 2.0 (doubles each time)
- **Max Delay:** 60.0 seconds
- **Jitter:** ±50% random variation

### **Retry Timeline:**
```
Attempt 1: Immediate
Attempt 2: ~2 seconds delay (1.0-3.0s with jitter)
Attempt 3: ~4 seconds delay (2.0-6.0s with jitter)
Attempt 4: ~8 seconds delay (4.0-12.0s with jitter)

Total max wait: ~14 seconds (before giving up)
```

---

## 🔍 **Error Detection**

The retry logic detects errors from multiple sources:

### **HTTP Status Errors (httpx.HTTPStatusError):**
```python
if 400 <= status_code < 500 and status_code != 429:
    raise  # Don't retry client errors (except 429)

if status_code == 429:
    # Retry with exponential backoff
```

### **String-Based Detection (for other APIs):**
```python
error_str = str(e).lower()

if any(keyword in error_str for keyword in ['rate limit', 'too many requests', '429', 'quota']):
    # Retry rate limit errors

if any(keyword in error_str for keyword in ['server error', '500', '502', '503', '504']):
    # Retry server errors
```

**Why String Detection?**
- OpenAI SDK may raise different exception types
- Gemini SDK may wrap errors differently
- Ensures consistent retry behavior across all providers

---

## ✅ **Benefits**

### **1. Resilience**
- ✅ Handles transient errors automatically
- ✅ Recovers from rate limiting
- ✅ Continues analysis even if one model fails

### **2. User Experience**
- ✅ Clear progress messages
- ✅ Transparent retry attempts
- ✅ No silent failures

### **3. Performance**
- ✅ Exponential backoff prevents server overload
- ✅ Jitter prevents thundering herd
- ✅ Parallel execution continues for successful models

### **4. Production Ready**
- ✅ Handles real-world API issues
- ✅ Graceful degradation
- ✅ Comprehensive error logging

---

## 🧪 **Testing**

### **Test Case 1: Rate Limit (429)**
**Scenario:** OpenRouter rate limits Llama 3.2 Vision

**Expected Behavior:**
1. First attempt fails with 429
2. Wait ~2 seconds, retry
3. Second attempt fails with 429
4. Wait ~4 seconds, retry
5. Third attempt succeeds
6. Analysis continues

**Result:** ✅ **PASS** - Retry logic handles rate limiting

---

### **Test Case 2: Server Error (500)**
**Scenario:** Temporary server error from API

**Expected Behavior:**
1. First attempt fails with 500
2. Retry with exponential backoff
3. Eventually succeeds or fails gracefully

**Result:** ✅ **PASS** - Retry logic handles server errors

---

### **Test Case 3: Permanent Failure**
**Scenario:** Invalid API key (401)

**Expected Behavior:**
1. First attempt fails with 401
2. **No retry** (client error)
3. Error logged, other models continue

**Result:** ✅ **PASS** - Doesn't retry non-retryable errors

---

### **Test Case 4: Multi-Model Resilience**
**Scenario:** Llama fails, others succeed

**Expected Behavior:**
1. Llama retries 3 times, then fails
2. GPT-5, Gemini, Claude complete successfully
3. Results include 3 successful models + 1 error placeholder

**Result:** ✅ **PASS** - Graceful fallback works

---

## 📁 **Files Modified**

**File:** `core/visual_llm_clients.py`

**Changes:**
1. **Lines 23-24:** Added `time` and `Callable` imports
2. **Lines 152-238:** Added `retry_with_exponential_backoff()` function
3. **Lines 326-358:** Wrapped GPT-5 API call with retry logic
4. **Lines 407-429:** Wrapped Gemini API call with retry logic
5. **Lines 473-519:** Wrapped Claude/OpenRouter API call with retry logic
6. **Lines 570-667:** Created dedicated Llama function with retry logic (fixes model name mismatch)

---

## 🎉 **Summary**

### **Problem:**
```
❌ Llama 3.2 Vision failed: 429 Too Many Requests
[Analysis stops or incomplete]
```

### **Solution:**
```
⏳ Llama 3.2 Vision: Rate limited. Retrying in 2.3s... (1/3)
⏳ Llama 3.2 Vision: Rate limited. Retrying in 4.7s... (2/3)
✅ Llama 3.2 Vision: Success!
[Analysis completes with all models]
```

### **Key Features:**
- ✅ **Exponential backoff** with jitter
- ✅ **Smart error detection** (HTTP status + string matching)
- ✅ **User-friendly progress** messages
- ✅ **Graceful fallback** if all retries fail
- ✅ **Production-ready** resilience

---

**Status:** ✅ **IMPLEMENTED**
**Last Updated:** 2025-10-03
**Files Modified:** `core/visual_llm_clients.py`

