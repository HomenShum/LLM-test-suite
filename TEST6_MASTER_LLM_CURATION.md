# Test 6: Master LLM Image Curation System

## 🎯 Overview

The Master LLM Image Curation System uses a master AI model to intelligently curate high-quality, task-relevant images and create ground truth expectations for testing other visual LLMs.

---

## 🧠 Why Master LLM Curation?

### **Problem with Standard Search:**
```
User selects: "🏭 Product Defect Detection"
Standard search query: "product defects manufacturing quality control"

Results:
❌ Generic stock photos
❌ Irrelevant marketing images
❌ Low-quality screenshots
❌ Images with watermarks
❌ Wrong subject matter
```

### **Solution with Master LLM:**
```
Master LLM analyzes task: "Detect defects, scratches, dents..."

Generated queries:
✅ "high resolution product defects close-up manufacturing"
✅ "quality control inspection defective parts scratches"
✅ "industrial defect detection professional photography"

Master LLM evaluates each image:
✅ Relevance score: 85/100 - Clear defect visible
❌ Relevance score: 45/100 - Watermarked stock photo (rejected)
✅ Relevance score: 92/100 - Professional inspection photo

Result: 20 high-quality, task-relevant images
```

---

## 🔄 Complete Workflow

### **Phase 1: Query Generation**
```
Input: Task description + Preset name

Master LLM (GPT-5-nano):
├─ Analyzes task requirements
├─ Generates 3 optimized search queries
├─ Provides rationale for each query
└─ Prioritizes queries (1=highest)

Output:
[
  {
    "query": "high resolution product defects close-up manufacturing",
    "rationale": "Focuses on clear, detailed defect visibility",
    "expected_results": "Professional inspection photos with visible defects",
    "priority": 1
  },
  ...
]
```

### **Phase 2: Image Search & Download**
```
For each query (by priority):
├─ Search Linkup API
├─ Download 15 candidate images
├─ Validate with PIL (format, size, corruption)
└─ Stop when enough candidates collected

Result: 30-45 candidate images
```

### **Phase 3: Relevance Evaluation**
```
For each candidate image:

Master LLM evaluates:
├─ Relevance to task (0-100)
├─ Image quality (0-100)
├─ Issues (watermarks, text, poor quality)
└─ Expected content description

Output:
{
  "relevance_score": 85,
  "is_relevant": true,
  "quality_score": 90,
  "issues": [],
  "expected_content": "Product with visible scratch defect"
}

Filter: Keep only images with relevance_score >= 70
```

### **Phase 4: Image Selection**
```
Sort by relevance score (highest first)
Select top 20 images
Result: 20 high-quality, task-relevant images
```

### **Phase 5: Ground Truth Creation**
```
For each selected image:

Master LLM creates ground truth:
├─ Expected analysis (definitive correct answer)
├─ Key findings (what should be detected)
├─ Expected rating (if applicable)
├─ Confidence range (reasonable range)
├─ Difficulty level (easy/medium/hard)
├─ Common mistakes (what other models might miss)
└─ Critical details (must-not-miss items)

Output:
{
  "expected_analysis": "Product shows clear scratch defect on surface...",
  "key_findings": ["scratch defect", "surface damage", "quality issue"],
  "expected_rating": 2,
  "confidence_range": "0.8-0.95",
  "difficulty_level": "medium",
  "common_mistakes": ["Missing subtle discoloration"],
  "critical_details": ["Scratch extends 3cm diagonally"]
}
```

---

## 📊 Curation Report

After curation, users see a detailed report:

```
✅ Master LLM curated 20 high-quality images

📋 Curation Report

┌─────────────────────┬────┐
│ Images Evaluated    │ 42 │
│ Images Selected     │ 20 │
│ Images Rejected     │ 22 │
└─────────────────────┴────┘

Generated Search Queries:
- high resolution product defects close-up manufacturing
  Rationale: Focuses on clear, detailed defect visibility

- quality control inspection defective parts scratches
  Rationale: Targets professional inspection documentation

- industrial defect detection professional photography
  Rationale: Ensures high-quality, well-lit images
```

---

## 🎯 Ground Truth Usage

### **During Analysis:**
Ground truth is stored in session state:
```python
st.session_state.test6_ground_truths = {
    "image_001.jpg": {
        "expected_analysis": "...",
        "key_findings": [...],
        ...
    },
    ...
}
```

### **In Results Display:**
Each image shows:
1. **Visual LLM outputs** (GPT-5, Gemini, Claude)
2. **Master LLM ground truth** (expected answer)
3. **Comparison** (how well each model matched)

```
📸 image_001.jpg

┌─────────────────────────────────────────┐
│ 🧠 Master LLM Ground Truth:             │
│                                         │
│ Product shows clear scratch defect...   │
│                                         │
│ Key Findings: scratch defect, surface   │
│ damage, quality issue                   │
│                                         │
│ Expected Rating: 2/5                    │
│ Difficulty: Medium                      │
└─────────────────────────────────────────┘

GPT-5 Vision:
"Detected scratch on surface, quality rating: 2/5"
✅ Matches ground truth

Gemini 2.5 Vision:
"Surface damage visible, minor defect, rating: 3/5"
⚠️ Missed severity (rated too high)

Claude 4.5 Vision:
"Scratch defect present, extends diagonally, rating: 2/5"
✅ Matches ground truth + extra detail
```

---

## 🔧 Technical Implementation

### **Core Module: `core/master_llm_curator.py`**

**Functions:**
1. `generate_optimized_search_queries()` - Generate 3 optimized queries
2. `evaluate_image_relevance()` - Score each image (0-100)
3. `create_ground_truth_expectations()` - Create definitive answer
4. `curate_image_dataset()` - Complete workflow orchestration

**API Calls:**
- Query generation: 1 call (GPT-5-nano)
- Image evaluation: N calls (1 per candidate image)
- Ground truth: M calls (1 per selected image)

**Total API calls:** ~1 + 40 + 20 = 61 calls per preset

**Cost estimate:** ~$0.10-0.20 per preset (GPT-5-nano is cheap)

---

## 📈 Benefits

### **For Users:**
✅ **Higher Quality Images** - Only task-relevant images selected  
✅ **Better Testing** - Ground truth enables objective evaluation  
✅ **Transparency** - See why images were selected/rejected  
✅ **Automatic** - No manual curation needed  

### **For Analysis:**
✅ **Objective Comparison** - Compare models against ground truth  
✅ **Identify Weaknesses** - See which models miss critical details  
✅ **Difficulty Levels** - Know which images are harder  
✅ **Common Mistakes** - Anticipate model errors  

### **For Development:**
✅ **Reproducible** - Same task = similar quality images  
✅ **Scalable** - Works for any task/preset  
✅ **Extensible** - Easy to add new evaluation criteria  

---

## 🎯 Example: Product Defect Detection

### **Input:**
```
Task: "Detect defects, scratches, dents, discoloration. Rate quality 1-5."
Preset: "🏭 Product Defect Detection"
```

### **Master LLM Process:**

**Step 1: Query Generation**
```
Generated 3 queries:
1. "high resolution product defects close-up manufacturing"
2. "quality control inspection defective parts scratches"
3. "industrial defect detection professional photography"
```

**Step 2: Image Search**
```
Query 1: Found 18 images
Query 2: Found 15 images
Query 3: Found 12 images
Total: 45 candidate images
```

**Step 3: Evaluation**
```
Image 1: Relevance 92/100 ✅ Selected
Image 2: Relevance 45/100 ❌ Rejected (watermark)
Image 3: Relevance 88/100 ✅ Selected
...
Image 45: Relevance 55/100 ❌ Rejected (poor quality)

Selected: 20 images (relevance >= 70)
```

**Step 4: Ground Truth**
```
Image 1:
  Expected: "Scratch defect 3cm diagonal, quality 2/5"
  Key findings: ["scratch", "surface damage"]
  Difficulty: Medium

Image 2:
  Expected: "Dent on corner, minor discoloration, quality 3/5"
  Key findings: ["dent", "discoloration"]
  Difficulty: Easy
...
```

### **Output:**
```
✅ 20 curated images with ground truth
📊 Ready for visual LLM testing
🎯 Objective evaluation enabled
```

---

## 🔄 Integration with Test 6

### **Automatic Activation:**
Master LLM curation is **always used** when searching for new images.

**User Flow:**
```
1. User selects preset: "🏭 Product Defect Detection"
2. System checks cache:
   ├─ Has cache: "Use Cached Images" or "Search New Images"
   └─ No cache: Automatically use Master LLM curation
3. If "Search New Images":
   └─ Master LLM curation runs automatically
4. Analysis proceeds with curated images + ground truth
```

**No User Choice Needed:**
- ✅ Automatic and seamless
- ✅ Always high quality
- ✅ Always has ground truth

---

## 📊 Session State Storage

All curation data is saved to session state:

```python
# Ground truth for each image
st.session_state.test6_ground_truths = {
    "path/to/image1.jpg": {...},
    "path/to/image2.jpg": {...},
}

# Curation report
st.session_state.test6_curation_report = {
    "timestamp": "2025-10-02T14:30:00",
    "queries_generated": [...],
    "images_evaluated": 42,
    "images_selected": 20,
    "images_rejected": 22,
    "rejection_reasons": [...]
}

# Analysis results
st.session_state.test6_analysis_results = [...]
st.session_state.test6_selected_models = [...]
st.session_state.test6_preset_name = "..."
st.session_state.test6_task_description = "..."
```

**Benefits:**
- ✅ Persists across tab switches
- ✅ Available for all analysis phases
- ✅ No re-computation needed
- ✅ Exportable with results

---

## 🚀 Future Enhancements

1. **Adaptive Thresholds:**
   - Adjust relevance threshold based on task difficulty
   - Lower threshold if not enough images found

2. **Multi-Model Consensus:**
   - Use multiple master models for ground truth
   - Combine their outputs for more robust expectations

3. **Active Learning:**
   - Learn from user feedback on image quality
   - Improve query generation over time

4. **Custom Evaluation Criteria:**
   - User-defined relevance criteria
   - Task-specific quality metrics

5. **Batch Curation:**
   - Curate multiple presets in parallel
   - Share images across similar tasks

---

## 📝 Files Created/Modified

### **New Files:**
1. ✅ `core/master_llm_curator.py` - Master LLM curation engine
2. ✅ `TEST6_MASTER_LLM_CURATION.md` - This documentation

### **Modified Files:**
1. ✅ `ui/test6_visual_llm.py` - Integrated automatic curation
2. ✅ `ui/test6_advanced_results.py` - Display ground truth in results

---

## ✅ Summary

**Master LLM Image Curation System:**
- ✅ Generates optimized search queries
- ✅ Evaluates image relevance automatically
- ✅ Selects only high-quality images
- ✅ Creates ground truth for testing
- ✅ Fully automatic (no user intervention)
- ✅ Integrated with Test 6 workflow
- ✅ Results saved to session state

**Result:** High-quality, task-relevant images with objective ground truth for comprehensive visual LLM testing.

---

**Last Updated:** 2025-10-02  
**Status:** ✅ Implemented and integrated

