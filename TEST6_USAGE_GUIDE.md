# Test 6: Visual LLM Testing - Usage Guide

## 🚀 Quick Start

1. **Launch the app**: `streamlit run app.py`
2. **Navigate to Test 6**: Click on the "Test 6: Visual LLM Testing" tab
3. **Choose mode**: Select Mode A (VR Avatar) or Mode B (General Visual)
4. **Select models**: Choose which visual LLMs to test
5. **Run analysis**: Upload images and click "Run Analysis"

---

## 🎯 Mode A: VR Avatar Validation

### Purpose
Analyze VR avatar recordings for visual artifacts and quality issues.

### Workflow

1. **Upload CSV File**
   - Must contain columns: `avatar_id`, `image_path`, `human_movement_rating`, etc.
   - See `data/vr_avatar_test_data.csv` for example format

2. **Select Models**
   - Quick Select: Use recommended models (GPT-5, Gemini, Claude, Llama)
   - Advanced: Choose specific models with custom pricing

3. **Configure Analysis**
   - Artifact types to detect (red lines, texture issues, etc.)
   - Rating criteria (movement, visual quality, artifacts)

4. **Run Analysis**
   - Models analyze each avatar image in parallel
   - Results compared against human ratings

5. **View Results**
   - **Summary Tab**: Overall statistics and agreement scores
   - **Detailed Results Tab**: Per-avatar analysis with ratings
   - **Visualizations Tab**: Scatter plots, correlation charts

### Expected Output

```
Avatar: avatar_001

┌─────────────────────────────────────────────────────┐
│ GPT-5 Vision                                        │
├─────────────┬─────────────┬─────────────────────────┤
│ Movement    │ Visual      │ Artifact                │
│ 3.5/5       │ 4.0/5       │ 2.5/5                   │
└─────────────┴─────────────┴─────────────────────────┘

Detected Artifacts:
• red lines in eyes
• texture glitches

Human Ratings:
• Movement: 3.0/5
• Visual: 4.5/5
• Comfort: 3.5/5

Agreement Score: 85%
✅ Confidence: 85.0%
```

---

## 🎨 Mode B: General Visual Comparison

### Purpose
Compare visual LLM performance on any image dataset.

### Workflow

1. **Collect Images**
   - **Option A**: Web Search (Linkup API) - Search and download images
   - **Option B**: Manual Upload - Upload your own images

2. **Define Task**
   - Describe what models should analyze
   - Examples:
     - "Detect objects and classify image category"
     - "Identify product defects and quality issues"
     - "Analyze medical imaging for anomalies"

3. **Select Models**
   - Same as Mode A

4. **Run Analysis**
   - Each model analyzes all images
   - Results cached for follow-up questions

5. **View Results**
   - **Raw Results Tab**: Model responses for each image
   - **Comparison Tab**: Side-by-side model comparison
   - **Visualizations Tab**: Performance metrics

### Expected Output

```
Image: product_001.jpg

┌─────────────────────────────────────────────────────┐
│ Model Comparison                                    │
├─────────────┬─────────────┬─────────────────────────┤
│ GPT-5       │ Gemini      │ Claude                  │
│ 4.5/5       │ 4.0/5       │ 4.8/5                   │
└─────────────┴─────────────┴─────────────────────────┘

Consensus: High quality product, no defects
Agreement: 92%
```

---

## ⚙️ Advanced Settings

### Debug Mode

**When to use:**
- Models returning unexpected results
- Confidence scores seem wrong
- Want to see raw API responses

**How to enable:**
1. Expand "⚙️ Advanced Settings"
2. Check "Enable Debug Mode"
3. Run analysis

**What you'll see:**
- Parsing warnings for failed JSON
- Raw response previews (first 500 chars)
- Detailed error messages

**Example debug output:**
```
⚠️ JSON parsing failed for GPT-5 Vision, falling back to text extraction

Debug: Raw response from GPT-5 Vision ▼
{
  "movement_rating": null,
  "visual_quality_rating": 5.0,
  ...
```

---

## 📊 Understanding Results

### Confidence Scores

| Color | Range | Meaning |
|-------|-------|---------|
| ✅ Green | 80-100% | High confidence - Model is very certain |
| ⚠️ Yellow | 50-79% | Medium confidence - Some uncertainty |
| ❌ Red | 0-49% | Low confidence - High uncertainty |

**How confidence is determined:**
1. **Explicit**: Model provides confidence in response
2. **Fallback**: Based on response quality and structure
   - Structured with ratings + artifacts: 85%
   - Very detailed (>300 chars): 80%
   - Detailed (>200 chars): 75%
   - Moderate (>100 chars): 70%
   - Brief: 60%

### Rating Scales

All ratings use a **1-5 scale**:
- **5**: Excellent / No issues
- **4**: Good / Minor issues
- **3**: Average / Some issues
- **2**: Poor / Significant issues
- **1**: Very poor / Major issues
- **N/A**: Not applicable / Cannot determine

### Artifact Detection

Models detect and list specific artifacts:
- Red lines in eyes
- Texture glitches
- Clipping issues
- Lighting problems
- Rendering artifacts
- etc.

---

## 💰 Cost Tracking

### Viewing Costs

1. Check "Cost Estimation" section before running
2. See per-image cost breakdown by model
3. Total cost displayed at bottom

### Example Cost Breakdown

```
Model                    | Prompt    | Completion | Image    | Total
-------------------------|-----------|------------|----------|----------
GPT-5 Vision (nano)      | $0.0001   | $0.0001    | $0.0000  | $0.0002
Gemini 2.5 Flash Lite    | $0.0000   | $0.0000    | $0.0000  | $0.0000
Claude 4.5 Sonnet        | $0.0030   | $0.0150    | $0.0000  | $0.0180
Llama 3.2 Vision         | $0.0001   | $0.0001    | $0.0000  | $0.0002

Total per image: $0.0184
```

### Cost Optimization Tips

1. **Use Quick Select**: Recommended models are cost-optimized
2. **Start small**: Test with 1-2 images first
3. **Choose wisely**: Claude is more expensive but often more accurate
4. **Batch processing**: Analyze multiple images in one run

---

## 🔧 Troubleshooting

### Issue: "Unable to parse structured response"

**Cause**: Model returned non-JSON format

**Solution**:
1. Enable Debug Mode to see raw response
2. Check if model supports structured output
3. Try a different model
4. Report issue if persistent

### Issue: Confidence always 0%

**Cause**: Model not providing confidence in response

**Solution**:
1. Check Debug Mode for raw response
2. Verify model is returning expected format
3. Fallback confidence should apply automatically
4. Report if fallback not working

### Issue: No ratings displayed

**Cause**: Model didn't provide ratings in expected format

**Solution**:
1. Check if task description asks for ratings
2. Enable Debug Mode to see what model returned
3. Try rephrasing the task description
4. Some models may not support all rating types

### Issue: Models taking too long

**Cause**: Large images or slow API response

**Solution**:
1. Reduce image size before upload
2. Use fewer models
3. Check internet connection
4. Try again during off-peak hours

---

## 📝 Best Practices

### Writing Good Task Descriptions

**Good:**
```
Analyze this VR avatar for visual quality. Rate the following on a 
1-5 scale: movement smoothness, visual quality, and artifact presence. 
List any detected artifacts such as red lines, texture glitches, or 
clipping issues.
```

**Bad:**
```
Look at this image.
```

### Selecting Models

**For accuracy**: Include Claude 4.5 Sonnet
**For speed**: Use GPT-5 Nano or Gemini Flash Lite
**For cost**: Gemini Flash Lite is often free
**For comparison**: Use all 4 recommended models

### Interpreting Results

1. **Look for consensus**: If all models agree, high confidence
2. **Check confidence scores**: Higher = more reliable
3. **Compare to human ratings**: Validate model accuracy
4. **Review detected artifacts**: Specific issues identified
5. **Consider context**: Some tasks harder than others

---

## 📚 Additional Resources

- **Implementation Details**: See `TEST6_IMPLEMENTATION_SUMMARY.md`
- **Parsing Fixes**: See `TEST6_PARSING_FIXES.md`
- **Before/After**: See `TEST6_BEFORE_AFTER_COMPARISON.md`
- **Quick Start**: See `TEST6_QUICK_START.md`

---

## 🆘 Getting Help

**If you encounter issues:**

1. Enable Debug Mode
2. Check the console for error messages
3. Review the documentation files
4. Check API key configuration
5. Verify image formats are supported (jpg, png, webp)

**Common fixes:**
- Restart the app
- Clear browser cache
- Check API keys are valid
- Verify internet connection
- Update dependencies

---

**Last Updated**: 2025-10-02
**Version**: 1.1 (with parsing fixes)
**Status**: ✅ Production Ready

