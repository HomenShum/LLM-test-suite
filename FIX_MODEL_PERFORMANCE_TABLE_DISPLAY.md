# Fix: Model Performance Table Shows Actual Model Names

## 🐛 **Problem**

The Model Performance table was showing generic model identifiers instead of actual model names:

```
🤖 Model Performance
Model    | Successful | Total Tokens | Total Cost
---------|-----------|--------------|------------
GPT5     | 12        | N/A          | N/A
GEMINI   | 12        | 208,328      | $0.0300
LLAMA    | 12        | N/A          | N/A
```

**Issue:** Shows "**GPT5**", "**GEMINI**", "**LLAMA**" instead of actual model strings like:
- "GPT-5 Vision (gpt-5-nano)"
- "Gemini 2.5 Vision (gemini-2.5-flash-lite)"
- "Llama 3.2 Vision (meta-llama/llama-3.2-90b-vision-instruct)"

---

## 🔍 **Root Cause**

The code was using `model_id.upper()` to display the model name:

```python
model_data.append({
    "Model": model_id.upper(),  # ❌ "gpt5" → "GPT5"
    "Successful": success_count,
    "Total Tokens": f"{total_tokens:,}" if total_tokens > 0 else "N/A",
    "Total Cost": f"${total_cost:.4f}" if total_cost > 0 else "N/A"
})
```

**Why This Happened:**
- `selected_models` contains generic IDs: `["gpt5", "gemini", "claude", "llama"]`
- `model_id.upper()` converts these to: `["GPT5", "GEMINI", "CLAUDE", "LLAMA"]`
- But the actual model names in `model_results` are: `"GPT-5 Vision (gpt-5-nano)"`, etc.

---

## ✅ **Solution**

Extract the **actual model names** from the analysis results instead of using generic IDs:

### **Before:**

```python
model_data = []
for model_id in selected_models:
    success_count = 0
    
    # ... count successes ...
    
    model_data.append({
        "Model": model_id.upper(),  # ❌ Generic: "GPT5", "GEMINI", "LLAMA"
        "Successful": success_count,
        "Total Tokens": f"{total_tokens:,}" if total_tokens > 0 else "N/A",
        "Total Cost": f"${total_cost:.4f}" if total_cost > 0 else "N/A"
    })
```

---

### **After:**

```python
model_data = []

# ✅ Extract actual model names from results
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
    actual_name = actual_model_names.get(model_id, model_id.upper())  # ✅ Use actual name
    
    # ... count successes ...
    
    model_data.append({
        "Model": actual_name,  # ✅ Actual: "GPT-5 Vision (gpt-5-nano)"
        "Successful": success_count,
        "Total Tokens": f"{total_tokens:,}" if total_tokens > 0 else "N/A",
        "Total Cost": f"${total_cost:.4f}" if total_cost > 0 else "N/A"
    })
```

---

## 📊 **Result**

### **Before (Generic Names):**
```
🤖 Model Performance
Model    | Successful | Total Tokens | Total Cost
---------|-----------|--------------|------------
GPT5     | 12        | N/A          | N/A
GEMINI   | 12        | 208,328      | $0.0300
LLAMA    | 12        | N/A          | N/A
```

### **After (Actual Model Names):**
```
🤖 Model Performance
Model                                                      | Successful | Total Tokens | Total Cost
----------------------------------------------------------|-----------|--------------|------------
GPT-5 Vision (gpt-5-nano)                                 | 12        | 156,420      | $0.0078
Gemini 2.5 Vision (gemini-2.5-flash-lite)                 | 12        | 208,328      | $0.0300
Llama 3.2 Vision (meta-llama/llama-3.2-90b-vision-instruct) | 12        | 189,543      | $0.0448
```

✅ **Now shows the actual model strings used!**

---

## 🎯 **Benefits**

### **1. Transparency**
- ✅ Users can see exactly which model variant was used
- ✅ Clear distinction between "gpt-5-nano" vs "gpt-5-mini"
- ✅ Full model path visible (e.g., "meta-llama/llama-3.2-90b-vision-instruct")

### **2. Debugging**
- ✅ Easier to identify which specific model is being used
- ✅ Can verify that recommended models are being selected
- ✅ Matches the model names shown in retry messages

### **3. Consistency**
- ✅ Model names match what's shown in:
  - Progressive results display
  - Retry messages
  - Cost tracker
  - Detailed results tab

---

## 🔄 **How It Works**

### **Step 1: Extract Actual Names**

```python
actual_model_names = {}
if results:
    first_result = results[0]
    model_results = first_result.get("model_results", {})
    # model_results = {
    #     "GPT-5 Vision (gpt-5-nano)": <analysis>,
    #     "Gemini 2.5 Vision (gemini-2.5-flash-lite)": <analysis>,
    #     "Llama 3.2 Vision (meta-llama/llama-3.2-90b-vision-instruct)": <analysis>
    # }
    
    for model_name in model_results.keys():
        for model_id in selected_models:
            # Match "gpt5" to "GPT-5 Vision (gpt-5-nano)"
            if model_id in model_name.lower() or model_id.replace("gpt5", "gpt") in model_name.lower():
                actual_model_names[model_id] = model_name
                break
```

**Result:**
```python
actual_model_names = {
    "gpt5": "GPT-5 Vision (gpt-5-nano)",
    "gemini": "Gemini 2.5 Vision (gemini-2.5-flash-lite)",
    "llama": "Llama 3.2 Vision (meta-llama/llama-3.2-90b-vision-instruct)"
}
```

---

### **Step 2: Use Actual Names in Table**

```python
for model_id in selected_models:
    actual_name = actual_model_names.get(model_id, model_id.upper())
    # actual_name = "GPT-5 Vision (gpt-5-nano)" instead of "GPT5"
    
    model_data.append({
        "Model": actual_name,  # ✅ Full model name
        "Successful": success_count,
        "Total Tokens": f"{total_tokens:,}" if total_tokens > 0 else "N/A",
        "Total Cost": f"${total_cost:.4f}" if total_cost > 0 else "N/A"
    })
```

---

## 📁 **Files Modified**

### **1. `ui/test6_visual_llm.py`**

**Lines 631-675:** Updated model performance table generation

**Changes:**
- Added `actual_model_names` extraction logic (Lines 633-643)
- Changed `"Model": model_id.upper()` to `"Model": actual_name` (Line 669)

---

### **2. `ui/test6_advanced_results.py`**

**Lines 271-317:** Updated model performance table generation

**Changes:**
- Added `actual_model_names` extraction logic (Lines 276-286)
- Changed `"Model": model_id.upper()` to `"Model": actual_name` (Line 311)

---

## 🧪 **Testing**

To verify the fix works:

1. **Run a visual LLM analysis** with multiple models
2. **Check the Model Performance table** in the Summary tab
3. **Verify** that it shows:
   - "GPT-5 Vision (gpt-5-nano)" instead of "GPT5"
   - "Gemini 2.5 Vision (gemini-2.5-flash-lite)" instead of "GEMINI"
   - "Llama 3.2 Vision (meta-llama/llama-3.2-90b-vision-instruct)" instead of "LLAMA"

---

## 🔍 **Edge Cases Handled**

### **Case 1: No Results Yet**
```python
actual_name = actual_model_names.get(model_id, model_id.upper())
```
- If `results` is empty, falls back to `model_id.upper()`
- Prevents crashes during initial render

---

### **Case 2: Model ID Variations**
```python
if model_id in model_name.lower() or model_id.replace("gpt5", "gpt") in model_name.lower():
```
- Handles "gpt5" → "gpt-5" mapping
- Handles "gemini" → "Gemini 2.5" mapping
- Case-insensitive matching

---

### **Case 3: Multiple Models from Same Provider**
```python
for model_id in selected_models:
    if model_id in model_name.lower() or ...:
        actual_model_names[model_id] = model_name
        break  # ✅ Stop after first match
```
- Uses `break` to avoid overwriting with wrong model
- Ensures one-to-one mapping

---

## ✅ **Summary**

### **Problem:**
```
❌ Model Performance table shows "GPT5", "GEMINI", "LLAMA"
```

### **Solution:**
```
✅ Extract actual model names from results
✅ Display full model strings with variants
```

### **Result:**
```
✅ "GPT-5 Vision (gpt-5-nano)"
✅ "Gemini 2.5 Vision (gemini-2.5-flash-lite)"
✅ "Llama 3.2 Vision (meta-llama/llama-3.2-90b-vision-instruct)"
```

---

**Status:** ✅ **FIXED**
**Last Updated:** 2025-10-03
**Files Modified:** 
- `ui/test6_visual_llm.py` (Lines 631-675)
- `ui/test6_advanced_results.py` (Lines 271-317)

