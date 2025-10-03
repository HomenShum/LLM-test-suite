# Computational Analysis JSON Parsing Fix

## ✅ Issue Resolved

Fixed `json.decoder.JSONDecodeError` when running computational analysis in Test 6.

---

## 🐛 Root Cause

**Error:**
```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Location:** `core/visual_meta_analysis.py` line 93

**Problem:**
```python
response = await client.chat.completions.create(
    model=planner_model,
    messages=[{"role": "user", "content": planning_prompt}],
    max_completion_tokens=2000,
    response_format={"type": "json_object"}  # ❌ Not supported by GPT-5
)

plan_json = json.loads(response.choices[0].message.content)  # ❌ Empty or invalid JSON
```

**Why it failed:**
1. GPT-5 models don't reliably support `response_format={"type": "json_object"}`
2. The model returned empty content or non-JSON text
3. `json.loads()` failed on empty/invalid string

---

## ✅ Solution

### **1. Use Pydantic Structured Outputs**

**Created Pydantic Model:**
```python
from pydantic import BaseModel

class ComputationalAnalysisPlan(BaseModel):
    """Structured plan for computational analysis."""
    analysis_plan: str
    python_code: str
    expected_outputs: str
    recommended_visualizations: List[str]
```

**Updated API Call:**
```python
# Use Pydantic structured outputs for GPT-5
response = await client.beta.chat.completions.parse(
    model=planner_model,
    messages=[{"role": "user", "content": planning_prompt}],
    max_completion_tokens=2000,
    response_format=ComputationalAnalysisPlan  # ✅ Pydantic model
)

# Extract parsed object
plan_obj = response.choices[0].message.parsed
if plan_obj:
    plan_json = plan_obj.model_dump()  # ✅ Guaranteed valid structure
```

---

### **2. Added Error Handling**

**GPT-5 Path:**
```python
try:
    response = await client.beta.chat.completions.parse(...)
    
    plan_obj = response.choices[0].message.parsed
    if plan_obj:
        plan_json = plan_obj.model_dump()
    else:
        # Fallback: try to parse content as JSON
        content = response.choices[0].message.content
        if content:
            plan_json = json.loads(content)
        else:
            raise ValueError("Empty response from model")
            
except Exception as e:
    # Return error plan
    return {
        "analysis_plan": f"Error generating plan: {str(e)}",
        "python_code": "# Error: Could not generate code",
        "expected_outputs": "N/A",
        "recommended_visualizations": []
    }
```

**Gemini Path:**
```python
try:
    response = await asyncio.to_thread(
        lambda: client.models.generate_content(
            model="gemini-2.5-flash",
            contents=planning_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
    )
    
    plan_json = json.loads(response.text)
    
except Exception as e:
    # Return error plan
    return {
        "analysis_plan": f"Error generating plan: {str(e)}",
        "python_code": "# Error: Could not generate code",
        "expected_outputs": "N/A",
        "recommended_visualizations": []
    }
```

---

### **3. Updated Prompt**

**Before:**
```
**Output Format (JSON):**
{
    "analysis_plan": "Description of recommended analyses",
    "python_code": "Complete Python code to perform analysis",
    "expected_outputs": "Description of expected results",
    "visualizations": ["List of recommended chart types"]
}

Provide ONLY the JSON output, no additional text.
```

**After:**
```
**Required Output:**
- analysis_plan: Description of recommended analyses (string)
- python_code: Complete Python code to perform analysis (string)
- expected_outputs: Description of expected results (string)
- recommended_visualizations: List of recommended chart types (array of strings)

Generate a comprehensive analysis plan with executable Python code.
```

**Why better:**
- Clearer field descriptions
- Explicit type hints
- No confusing "ONLY JSON" instruction (Pydantic handles structure)
- Works better with structured outputs

---

## 📝 Files Modified

### **`core/visual_meta_analysis.py`**

**Changes:**
1. ✅ Added Pydantic import (line 18)
2. ✅ Created `ComputationalAnalysisPlan` model (line 21-27)
3. ✅ Updated prompt to be clearer (line 54-78)
4. ✅ Changed to use `client.beta.chat.completions.parse()` (line 87-89)
5. ✅ Added error handling for GPT-5 path (line 85-122)
6. ✅ Added error handling for Gemini path (line 124-159)

---

## 🎯 Benefits

### **1. Guaranteed Valid Structure**

**Before:**
```python
plan_json = json.loads(response.choices[0].message.content)
# ❌ Could be empty, invalid JSON, or wrong structure
```

**After:**
```python
plan_obj = response.choices[0].message.parsed
plan_json = plan_obj.model_dump()
# ✅ Guaranteed to have all required fields with correct types
```

---

### **2. Better Error Messages**

**Before:**
```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**After:**
```
Error generating plan: Empty response from model
```
or
```
Error generating plan: API key not valid
```

---

### **3. Graceful Degradation**

**Before:** App crashes with traceback

**After:** Shows error message in UI, allows user to retry

```python
# In UI
if execution_results.get('success'):
    st.success("✅ Analysis executed successfully!")
else:
    st.error(f"❌ Execution failed: {execution_results.get('error')}")
```

---

## 🧪 Testing

### **Test Scenario:**

1. **Run Test 6 with preset:**
   - Select "🎨 Art & Style Analysis"
   - Choose 2 models
   - Click "🚀 Run Preset"
   - Wait for analysis to complete

2. **Go to Tab 4: Computational Analysis**
   - Click "🚀 Run Computational Analysis"

3. **Expected Results:**

**Success Case:**
```
🧠 Planning analysis with GPT-5...
✅ Analysis plan generated!

📋 Analysis Plan
Plan: Compare model agreement on art style identification...
Expected Outputs: Agreement percentages, confidence distributions...
[Python code displayed]

⚙️ Executing code with Gemini Code Execution...
✅ Analysis executed successfully with Gemini Code Execution!

📊 Analysis Results
[Execution output displayed]
```

**Error Case (if API key missing):**
```
🧠 Planning analysis with GPT-5...
❌ Error: Error generating plan: API key not valid

Plan: Error generating plan: API key not valid
Code: # Error: Could not generate code
```

---

## 🔄 Execution Flow

```
User clicks "🚀 Run Computational Analysis"
         │
         ▼
┌────────────────────────────────────────┐
│ Step 1: Planning (GPT-5)               │
│ - Use Pydantic structured outputs      │
│ - Parse response to ComputationalPlan  │
│ - Extract: plan, code, outputs, viz    │
│ - Track cost                           │
└────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ Step 2: Display Plan                   │
│ - Show analysis plan                   │
│ - Show Python code                     │
│ - Show expected outputs                │
└────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ Step 3: Execute Code (Gemini)          │
│ - Send code to Gemini Code Execution   │
│ - Extract execution output             │
│ - Track cost                           │
└────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ Step 4: Display Results                │
│ - Show execution method badge          │
│ - Show execution output                │
│ - Show generated code (expandable)     │
└────────────────────────────────────────┘
```

---

## 📊 Comparison

### **Before (Broken):**

```python
# Planning
response = await client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"}  # ❌ Not reliable
)
plan_json = json.loads(response.choices[0].message.content)  # ❌ Crashes
```

**Result:** `JSONDecodeError` crash

---

### **After (Fixed):**

```python
# Planning
response = await client.beta.chat.completions.parse(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": prompt}],
    response_format=ComputationalAnalysisPlan  # ✅ Pydantic model
)
plan_obj = response.choices[0].message.parsed
plan_json = plan_obj.model_dump()  # ✅ Guaranteed valid
```

**Result:** Valid structured output or graceful error

---

## ✅ Verification Checklist

- [x] Added Pydantic model for structured outputs
- [x] Updated GPT-5 to use `client.beta.chat.completions.parse()`
- [x] Added error handling for GPT-5 path
- [x] Added error handling for Gemini path
- [x] Updated prompt to be clearer
- [x] Added fallback for empty responses
- [x] Graceful error messages in UI

---

## 🎉 Result

**Computational Analysis now works reliably!**

Users can:
1. ✅ Generate analysis plans with GPT-5
2. ✅ Execute code with Gemini Code Execution
3. ✅ See clear error messages if something fails
4. ✅ Retry without app crashes

**All JSON parsing errors are handled gracefully with informative error messages.**

---

**Last Updated:** 2025-10-02  
**Status:** ✅ Fixed and tested

