# Test 6: Visual LLM Testing - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### **Step 1: Verify API Keys**

Check your `.env` file has these keys:
```bash
OPENAI_API_KEY=sk-proj-...
GEMINI_API_KEY=AIz...
OPENROUTER_API_KEY=sk-or-...
LINKUP_API_KEY=2df...  # For Mode B web search
```

### **Step 2: Run the App**

```bash
cd "d:\VSCode Projects\LLM_test_suite"
streamlit run streamlit_test_v5.py
```

### **Step 3: Navigate to Test 6**

Click on the **"Test 6: Visual LLM Testing"** tab in the Streamlit app.

---

## 🎯 Mode A: VR Avatar Validation (Quick Test)

### **Option 1: Use Example CSV**

Create a test CSV file `test_avatars.csv`:

```csv
avatar_id,video_path,screenshot_path,human_movement_rating,human_visual_rating,human_comfort_rating,bug_description
avatar_001,,test_dataset/visual_llm_images/test1.png,4.5,4.0,4.2,Minor finger glitch
avatar_002,,test_dataset/visual_llm_images/test2.png,3.0,2.5,3.5,Red lines in eyes
avatar_003,,test_dataset/visual_llm_images/test3.png,5.0,4.5,4.8,No issues
```

**Note**: Leave `video_path` empty for now (video analysis coming soon). Only `screenshot_path` is used.

### **Option 2: Use Your Own Data**

1. Place your avatar screenshots in `test_dataset/visual_llm_images/`
2. Create a CSV with the format above
3. Update paths to match your images

### **Run Analysis:**

1. Select **Mode A: VR Avatar Validation**
2. Check the models you want to test (e.g., GPT-4V + Gemini)
3. Upload your CSV file
4. Select artifact types (or use defaults)
5. Click **"Run VR Avatar Analysis"**

**Expected Output:**
- Progress bar showing analysis status
- Raw results in JSON format (visualizations coming soon)
- Cost tracking in the sidebar

---

## 🌐 Mode B: General Visual Comparison (Quick Test)

### **Option 1: Manual Upload**

1. Select **Mode B: General Visual Comparison**
2. Choose **"Manual Upload"** for image collection
3. Upload 3-5 test images (any JPG/PNG files)
4. Enter analysis task: `"Detect objects and classify image category"`
5. Select models to test
6. Click **"Run Multi-Model Visual Analysis"**

### **Option 2: Web Search (Coming Soon)**

1. Select **"Web Search (Linkup API)"**
2. Enter search query: `"VR avatar screenshots"`
3. Set number of images: `10`
4. Click **"Search and Download Images"**

**Note**: Linkup API integration is in progress. Use manual upload for now.

---

## 🧪 Testing Checklist

### **Basic Functionality:**
- [ ] Test 6 tab loads without errors
- [ ] Mode selector switches between A and B
- [ ] Model checkboxes work
- [ ] CSV upload works (Mode A)
- [ ] Image upload works (Mode B)
- [ ] Progress bar displays during analysis
- [ ] Results appear after analysis

### **API Integration:**
- [ ] GPT-4 Vision API calls work
- [ ] Gemini Vision API calls work
- [ ] Claude Vision API calls work (via OpenRouter)
- [ ] Llama Vision API calls work (via OpenRouter)
- [ ] Cost tracking updates in sidebar

### **Error Handling:**
- [ ] Missing API key shows error message
- [ ] Invalid CSV shows warning
- [ ] Missing image file shows warning
- [ ] Failed model continues with others

---

## 🐛 Troubleshooting

### **Issue: "OPENAI_API_KEY not set"**
**Solution**: Add your OpenAI API key to `.env` file

### **Issue: "Image not found for avatar X"**
**Solution**: Verify the `screenshot_path` in your CSV points to an existing file

### **Issue: "Model X failed: ..."**
**Solution**: Check the specific API key for that model is set in `.env`

### **Issue: Tab doesn't load**
**Solution**: Check terminal for Python errors. Verify all imports are correct.

### **Issue: No results displayed**
**Solution**: Check `st.session_state.test6_results` in Streamlit's session state viewer

---

## 📊 Understanding Results

### **Current Output Format:**

```json
{
  "mode": "A",
  "timestamp": "2025-10-02T15:30:00",
  "results": [
    {
      "avatar_id": "avatar_001",
      "model_results": {
        "GPT-4 Vision": {
          "model_name": "GPT-4V (gpt-4o)",
          "movement_rating": null,
          "visual_quality_rating": null,
          "artifact_presence_rating": null,
          "detected_artifacts": [],
          "confidence": 0.8,
          "rationale": "The avatar shows...",
          "raw_response": "..."
        },
        "Gemini Vision": { ... }
      },
      "human_ratings": {
        "movement": 4.5,
        "visual": 4.0,
        "comfort": 4.2
      },
      "bug_description": "Minor finger glitch"
    }
  ],
  "selected_models": ["gpt-5-mini", "gemini"]
}
```

### **Coming Soon:**
- Structured rating extraction (1-5 scale)
- Human vs LLM comparison charts
- Artifact detection frequency
- Model agreement metrics

---

## 🎨 Example Use Cases

### **Use Case 1: VR Avatar QA**
**Goal**: Validate 18 VR avatars from headset recordings
**Setup**:
- Mode A
- Models: GPT-4V + Gemini + Claude
- Artifacts: Red lines, finger/feet movement, distortions
**Output**: Prioritized bug list with LLM confidence scores

### **Use Case 2: Product Defect Detection**
**Goal**: Compare visual LLMs on product images
**Setup**:
- Mode B
- Upload 20 product images
- Task: "Detect manufacturing defects"
- Models: All 4 (GPT-4V, Gemini, Claude, Llama)
**Output**: Model agreement on defects, best model recommendation

### **Use Case 3: Medical Imaging Comparison**
**Goal**: Evaluate which visual LLM is best for medical scans
**Setup**:
- Mode B
- Upload medical images (X-rays, MRIs)
- Task: "Identify anatomical structures and anomalies"
- Models: All 4
**Output**: Accuracy comparison, model recommendations

---

## 🔄 Iteration Workflow

### **Typical Development Cycle:**

1. **Run initial analysis** with 2-3 models
2. **Review raw results** in JSON output
3. **Identify patterns** in LLM responses
4. **Refine prompts** in `core/visual_llm_clients.py`
5. **Re-run analysis** with updated prompts
6. **Compare results** across iterations
7. **Generate visualizations** (coming soon)
8. **Export report** (coming soon)

---

## 📈 Performance Tips

### **Speed Optimization:**
- Use fewer models for initial testing (e.g., just GPT-4V)
- Start with small datasets (3-5 images)
- Enable parallel execution (already default)
- Cache results to avoid re-running

### **Cost Optimization:**
- Use Gemini Vision (cheapest) for bulk analysis
- Use GPT-4V for high-accuracy validation
- Cache results for follow-up questions
- Limit image resolution if possible

### **Accuracy Optimization:**
- Use multiple models and compare consensus
- Provide detailed prompts with examples
- Use domain-specific artifact lists
- Validate with human ratings

---

## 🚀 Next Steps After Testing

Once basic functionality works:

1. **Implement rating extraction** - Parse LLM responses to extract 1-5 ratings
2. **Add visualizations** - Human vs LLM scatter plots, artifact frequency charts
3. **Integrate Gemini code execution** - Generate custom analysis code
4. **Add result caching** - Enable follow-up Q&A without rerunning
5. **Implement Linkup API** - Web search for Mode B
6. **Add LLM judge** - Meta-analysis of which model performed best

---

## 📞 Need Help?

**Check these files:**
- `TEST6_IMPLEMENTATION_SUMMARY.md` - Detailed architecture
- `core/visual_llm_clients.py` - API implementation
- `ui/test6_visual_llm.py` - UI logic
- `core/models.py` - Data schemas

**Common Questions:**
- **Q**: Can I add custom visual LLM models?
  **A**: Yes! Add a new function in `core/visual_llm_clients.py` following the existing pattern.

- **Q**: How do I customize the analysis prompt?
  **A**: Edit `build_vr_avatar_analysis_prompt()` or `build_general_visual_analysis_prompt()` in `core/visual_llm_clients.py`.

- **Q**: Where are results stored?
  **A**: In `st.session_state.test6_results` (in-memory) and `test_output/visual_llm_cache/` (persistent, coming soon).

---

**Happy Testing! 🎉**

