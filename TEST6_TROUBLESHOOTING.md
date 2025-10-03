# Test 6: Troubleshooting Guide

## 🔧 Common Issues and Solutions

---

## 1. SVD Did Not Converge Error

### **Error Message:**
```
numpy.linalg.LinAlgError: SVD did not converge in Linear Least Squares
```

### **Cause:**
This occurs when trying to fit a trend line with insufficient variance in the data (e.g., all confidence scores are identical or very similar).

### **Solution:**
✅ **Already Fixed!** The code now includes:
- Variance checks before fitting trend lines
- Error handling to skip trend line if fitting fails
- Graceful degradation (chart still shows without trend line)

### **What Changed:**
```python
# Before (would crash):
z = np.polyfit(df['avg_confidence'], df['agreement'], 1)

# After (safe):
if df['avg_confidence'].std() > 0.01 and df['agreement'].std() > 0.01:
    try:
        z = np.polyfit(df['avg_confidence'], df['agreement'], 1)
    except (np.linalg.LinAlgError, ValueError, RuntimeError):
        pass  # Skip trend line
```

### **When It Happens:**
- Testing with very few images (< 3)
- All models have identical confidence scores
- All models have identical agreement scores
- Data has no variance

### **Prevention:**
- Test with at least 5 images
- Use diverse images that will produce varied results
- Test with multiple models (2-4)

---

## 2. Missing Function Error

### **Error Message:**
```
NameError: name '_create_confidence_agreement_chart' is not defined
```

### **Cause:**
Function was referenced but not defined in the file.

### **Solution:**
✅ **Already Fixed!** Added the missing function to `ui/test6_advanced_results.py`.

### **Prevention:**
- Always check diagnostics before running
- Use IDE error checking
- Test all tabs after making changes

---

## 3. Synthesis Tab is Empty

### **Symptoms:**
- Synthesis tab loads but shows no content
- "No data available" messages
- Empty tables and charts

### **Causes & Solutions:**

#### **Cause 1: Insufficient Models**
**Solution:** Test with at least 2 models
```python
# Need at least 2 models for agreement analysis
selected_models = ["gpt5", "gemini"]  # ✅ Good
selected_models = ["gpt5"]            # ❌ Not enough
```

#### **Cause 2: No Results Data**
**Solution:** Ensure analysis completed successfully
- Check for error messages during analysis
- Verify all models returned results
- Look for API errors in console

#### **Cause 3: Missing model_results Field**
**Solution:** Check data structure
```python
# Expected structure:
{
    "image_name": "test.jpg",
    "model_results": {  # ← Must have this field
        "gpt5": { ... },
        "gemini": { ... }
    }
}
```

---

## 4. Visualizations Don't Show

### **Symptoms:**
- Charts show "No data available"
- Heatmap is blank
- Correlation plot missing

### **Causes & Solutions:**

#### **Cause 1: Insufficient Data Points**
**Requirements:**
- Agreement heatmap: ≥ 2 models
- Correlation plot: ≥ 3 images
- Performance chart: ≥ 1 model with results

**Solution:** Run analysis with more images/models

#### **Cause 2: All Values Identical**
**Example:**
```python
# All confidence scores = 0.85
# No variance → No correlation plot
```

**Solution:** Use diverse images that produce varied results

#### **Cause 3: Browser Console Errors**
**Solution:** 
1. Open browser console (F12)
2. Look for JavaScript errors
3. Refresh page
4. Clear browser cache

---

## 5. All Insights Are Generic

### **Symptoms:**
- Insights say "No specific insights generated"
- Recommendations are vague
- No high-priority items

### **Causes & Solutions:**

#### **Cause 1: Too Few Images**
**Solution:** Run analysis on 5+ images
```python
# Minimum for good insights
num_images = 5  # ✅ Good
num_images = 2  # ❌ Too few
```

#### **Cause 2: Models Too Similar**
**Solution:** Test diverse models
```python
# Good diversity
models = ["gpt5", "gemini", "claude", "llama"]

# Poor diversity (all OpenAI)
models = ["gpt-4", "gpt-4-turbo", "gpt-4o"]
```

#### **Cause 3: Task Too Simple**
**Solution:** Use more complex analysis tasks
```python
# Simple (less insights)
task = "Describe the image"

# Complex (more insights)
task = "Analyze for defects, rate quality 1-5, detect artifacts"
```

---

## 6. Agreement Always 0%

### **Symptoms:**
- All agreement scores show 0.00%
- Heatmap is all red
- Pairwise agreement table shows zeros

### **Causes & Solutions:**

#### **Cause 1: Models Analyzing Different Images**
**Solution:** Verify all models analyzed same images
```python
# Check in results:
for result in results:
    print(f"Image: {result['image_name']}")
    print(f"Models: {list(result['model_results'].keys())}")
```

#### **Cause 2: Confidence Scores Not Extracted**
**Solution:** Check parsing in `core/rating_extractor.py`
- Enable debug mode
- Check raw model responses
- Verify confidence extraction regex

#### **Cause 3: Model Errors**
**Solution:** Check for API errors
```python
# Look for error messages in results
for result in results:
    for model, data in result['model_results'].items():
        if 'error' in data:
            print(f"{model} error: {data['error']}")
```

---

## 7. Prompt Optimization Shows No Changes

### **Symptoms:**
- Improved prompts identical to original
- "No changes recommended" for all models

### **Causes & Solutions:**

#### **Cause 1: All Models Performing Well**
**This is actually good!** It means:
- Average confidence > 70%
- Detail scores > 50%
- No improvements needed

#### **Cause 2: Insufficient Performance Data**
**Solution:** Run analysis on more images to get better statistics

---

## 8. Cost Tracking Shows N/A

### **Symptoms:**
- Total cost shows "N/A"
- Model costs show "N/A"

### **Causes & Solutions:**

#### **Cause 1: Cost Tracker Not Initialized**
**Solution:** Check `st.session_state.cost_tracker` exists
```python
if 'cost_tracker' not in st.session_state:
    from core.cost_tracker import CostTracker
    st.session_state.cost_tracker = CostTracker()
```

#### **Cause 2: API Calls Not Tracked**
**Solution:** Ensure cost tracking is enabled in API calls
- Check `core/visual_llm_clients.py`
- Verify cost tracker is called after each API request

---

## 9. Memory/Performance Issues

### **Symptoms:**
- App becomes slow
- Browser tab crashes
- "Out of memory" errors

### **Causes & Solutions:**

#### **Cause 1: Too Many Images**
**Solution:** Limit to 20 images per analysis
```python
# Good
num_images = 10-20

# May cause issues
num_images = 100+
```

#### **Cause 2: Large Images**
**Solution:** Resize images before upload
- Max recommended: 2048x2048 pixels
- Compress to reduce file size

#### **Cause 3: Session State Bloat**
**Solution:** Clear old analyses
```python
# In UI, use "Clear History" button
# Or manually:
st.session_state.test6_results = None
st.session_state.test6_computational_results = None
```

---

## 10. API Errors

### **Common API Errors:**

#### **OpenAI API Error**
```
Error: Invalid API key
```
**Solution:** Check API key in sidebar

#### **Rate Limit Error**
```
Error: Rate limit exceeded
```
**Solution:** 
- Wait a few minutes
- Reduce number of images
- Use fewer models

#### **Token Limit Error**
```
Error: Token limit exceeded
```
**Solution:** Already fixed with increased limits (8K-16K)

---

## 🔍 Debugging Tips

### **Enable Debug Mode**

1. In Test 6 UI, expand "Advanced Settings"
2. Check "Enable Debug Mode"
3. Re-run analysis
4. Check for debug messages

### **Check Browser Console**

1. Press F12 to open developer tools
2. Go to "Console" tab
3. Look for errors (red text)
4. Copy error messages for troubleshooting

### **Check Streamlit Logs**

1. Look at terminal where `streamlit run` is running
2. Check for Python errors
3. Look for API call logs

### **Verify Data Structure**

```python
# In Python console or notebook:
import json
with open('analysis_results.json', 'r') as f:
    results = json.load(f)
    
# Check structure
print(json.dumps(results[0], indent=2))
```

---

## 📞 Getting Help

### **Before Asking for Help:**

1. ✅ Check this troubleshooting guide
2. ✅ Enable debug mode
3. ✅ Check browser console
4. ✅ Review error messages
5. ✅ Try with fewer images/models

### **When Reporting Issues:**

Include:
- Error message (full traceback)
- Number of images and models
- Browser console errors
- Steps to reproduce
- Screenshots if relevant

### **Useful Files to Check:**

- `core/visual_results_synthesis.py` - Synthesis logic
- `ui/test6_synthesis_display.py` - Synthesis UI
- `ui/test6_advanced_results.py` - Results display
- `core/visual_llm_clients.py` - API calls
- `core/rating_extractor.py` - Parsing logic

---

## ✅ Quick Fixes Checklist

When something goes wrong:

- [ ] Refresh the page
- [ ] Clear browser cache
- [ ] Restart Streamlit app
- [ ] Check API keys are valid
- [ ] Verify internet connection
- [ ] Try with fewer images (3-5)
- [ ] Try with 2 models only
- [ ] Enable debug mode
- [ ] Check browser console
- [ ] Review error messages
- [ ] Check this guide

---

## 🎯 Best Practices

### **For Reliable Results:**

1. **Use 5-10 images** for good insights
2. **Test 2-4 models** for comparison
3. **Use diverse images** for varied results
4. **Enable debug mode** when testing
5. **Clear history** periodically
6. **Monitor costs** in sidebar
7. **Save results** before closing

### **For Performance:**

1. **Limit to 20 images** per analysis
2. **Resize large images** before upload
3. **Use presets** for common tasks
4. **Clear old analyses** regularly
5. **Close unused tabs** in browser

---

**Last Updated:** 2025-10-03
**Version:** 1.0
**Status:** ✅ Comprehensive

