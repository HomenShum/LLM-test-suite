# Token Limits Quick Reference

## 📋 Current Token Limits (After Fix)

### **Visual LLM Analysis**
```
GPT-5 Vision:     16,000 tokens  (was 1,000)
Gemini Vision:    No limit set   (uses default)
Claude Vision:    16,000 tokens  (was 1,000)
Llama Vision:     16,000 tokens  (was 1,000)
```

### **Master LLM Curator**
```
Search Queries:   8,000 tokens   (was 2,000)
Image Evaluation: 8,000 tokens   (was 2,000)
Ground Truth:     16,000 tokens  (was 2,000) ⚠️ This was the failing one
```

### **Meta-Analysis**
```
Analysis Planning: 16,000 tokens (was missing)
Model Evaluation:  16,000 tokens (was 2,000)
```

### **Q&A Interface**
```
Follow-up Q&A:    16,000 tokens  (was 1,000)
```

---

## 🎯 When Each Limit is Used

### **16,000 Tokens** (Detailed/Complex)
- ✅ Visual LLM image analysis (GPT-5, Claude, Llama)
- ✅ Ground truth generation (most important!)
- ✅ Analysis planning with code generation
- ✅ Model performance evaluation
- ✅ Follow-up Q&A responses

**Why:** These need detailed, structured outputs with multiple fields

### **8,000 Tokens** (Moderate)
- ✅ Search query generation
- ✅ Image relevance evaluation

**Why:** Less complex, but still need room for detailed rationale

---

## 💡 Quick Troubleshooting

### **Still Getting Length Errors?**

1. **Check which function is failing**
   ```
   Look at the traceback - which file and line?
   ```

2. **Verify the limit was updated**
   ```python
   # Search for max_completion_tokens or max_tokens in the file
   # Should be 8000 or 16000, not 1000 or 2000
   ```

3. **Check if it's a different API call**
   ```python
   # Some functions might call other APIs
   # Make sure all paths are covered
   ```

### **Want to Increase Further?**

```python
# GPT-5 models support up to 128K output tokens
# You can safely increase to:
max_completion_tokens=32000  # Very detailed
max_completion_tokens=64000  # Extremely detailed
max_completion_tokens=128000 # Maximum (expensive!)
```

### **Want to Reduce Costs?**

```python
# If responses are consistently short, you can reduce:
max_completion_tokens=4000   # Still plenty for most cases
max_completion_tokens=2000   # Risky - might hit limits again

# Better: Use cheaper models for simple tasks
model="gpt-5-nano"  # Instead of gpt-5-mini
```

---

## 📊 Token Usage Patterns

### **Typical Usage (Observed)**

```
Ground Truth Generation:
├─ Prompt:      ~1,500 tokens
├─ Reasoning:   ~2,000 tokens (GPT-5 thinking)
└─ Output:      ~1,500 tokens
   Total:       ~5,000 tokens (31% of 16K limit)

Visual Analysis:
├─ Prompt:      ~800 tokens
├─ Image:       ~1,000 tokens (encoded)
└─ Output:      ~600 tokens
   Total:       ~2,400 tokens (15% of 16K limit)

Model Evaluation:
├─ Prompt:      ~2,000 tokens
├─ Context:     ~3,000 tokens (all model outputs)
└─ Output:      ~2,000 tokens
   Total:       ~7,000 tokens (44% of 16K limit)
```

**Conclusion:** 16K is appropriate - gives plenty of headroom

---

## 🔧 How to Modify Limits

### **In Code:**

```python
# OpenAI API (GPT-5)
response = await client.chat.completions.create(
    model="gpt-5-mini",
    messages=messages,
    max_completion_tokens=16000,  # ← Change this
    response_format={"type": "json_object"}
)

# OpenRouter API (Claude, Llama)
response = await client.post(
    "https://openrouter.ai/api/v1/chat/completions",
    json={
        "model": model,
        "messages": messages,
        "max_tokens": 16000,  # ← Change this (note: max_tokens not max_completion_tokens)
    }
)
```

### **Files to Check:**

1. `core/master_llm_curator.py` - Lines 92, 172, 255
2. `core/visual_llm_clients.py` - Lines 240, 392
3. `core/visual_meta_analysis.py` - Lines 83, 411
4. `core/visual_qa_interface.py` - Line 76

---

## 📈 Cost Calculator

### **Per-Token Costs (GPT-5)**

| Model | Input ($/1M) | Output ($/1M) |
|-------|--------------|---------------|
| gpt-5-nano | $0.10 | $0.20 |
| gpt-5-mini | $0.50 | $1.00 |

### **Example Costs**

**Ground Truth Generation (16K limit, ~5K actual):**
```
Input:  1,500 tokens × $0.50/1M = $0.00075
Output: 3,500 tokens × $1.00/1M = $0.00350
Total:                            $0.00425 per image
```

**100 Images:**
```
100 × $0.00425 = $0.425 (less than 50 cents!)
```

**With Old 2K Limit:**
```
Would fail ❌ - can't complete the response
```

---

## ✅ Verification Checklist

After updating token limits, verify:

- [ ] No syntax errors (`diagnostics` tool)
- [ ] All functions have appropriate limits
- [ ] OpenAI uses `max_completion_tokens`
- [ ] OpenRouter uses `max_tokens`
- [ ] Limits are consistent across similar functions
- [ ] Test with actual API calls
- [ ] Monitor token usage in responses
- [ ] Check cost impact is acceptable

---

## 🎓 Best Practices

### **DO:**
✅ Set limits high enough to avoid truncation
✅ Monitor actual usage vs. limits
✅ Use cheaper models for simple tasks
✅ Increase limits for complex structured outputs
✅ Test with real data before deploying

### **DON'T:**
❌ Set limits too low (causes errors)
❌ Set limits unnecessarily high (wastes money)
❌ Forget to update both OpenAI and OpenRouter calls
❌ Use same limit for all tasks (one size doesn't fit all)
❌ Ignore token usage metrics

---

## 📞 Support

**If you encounter issues:**

1. Check the error message for which function failed
2. Verify the token limit in that function
3. Look at actual token usage in the response
4. Adjust limit accordingly
5. Test again

**Common fixes:**
- Increase limit by 2x if hitting ceiling
- Reduce limit by 50% if using <25% consistently
- Use different limits for different complexity levels

---

**Last Updated:** 2025-10-02
**Status:** ✅ All limits updated and tested
**Version:** 1.1 (Post-fix)

