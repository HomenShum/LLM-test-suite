# Test 6: Advanced Visual Analysis Workflow

## 🎯 Overview

The Advanced Visual Analysis Workflow provides a comprehensive 5-phase pipeline for analyzing images with multiple visual LLMs, performing computational analysis, evaluating model performance, and enabling interactive exploration.

---

## 📋 Workflow Phases

### **Phase 1: Image Collection & Initial Analysis**

**Purpose:** Gather images and run parallel visual LLM analysis

**Steps:**
1. **Image Collection:**
   - Use Linkup API to search for images based on preset query
   - Download and validate images (PIL validation)
   - Cache images for future use
   - User can choose between cached or new images

2. **Parallel Visual LLM Analysis:**
   - Run multiple models simultaneously (GPT-5, Gemini 2.5, Claude 4.5, Llama 3.2)
   - Extract structured outputs using Pydantic schemas
   - Track costs and tokens for each model
   - Handle errors gracefully (retry logic, fallbacks)

**Outputs:**
- List of analyzed images with model results
- Structured data (objects, scenes, artifacts, ratings, confidence)
- Cost tracking data

---

### **Phase 2: Meta-Analysis Planning**

**Purpose:** Use LLM to plan computational analysis

**Process:**
1. **Analysis Planner LLM** (GPT-5-nano or Gemini 2.5 Flash):
   - Reviews all visual LLM outputs
   - Identifies valuable computational analyses:
     - Statistical comparisons (agreement rates, correlation)
     - Clustering or pattern detection
     - Trend analysis across images
     - Confidence score distributions
     - Model performance metrics
   - Generates Python code for analysis
   - Describes expected outputs

**Outputs:**
- Analysis plan description
- Python code for computational analysis
- Expected results description
- Recommended visualizations

**Example Analysis Plan:**
```json
{
  "analysis_plan": "Compare model agreement rates and identify images with high disagreement",
  "python_code": "import pandas as pd\nimport numpy as np\n...",
  "expected_outputs": "Agreement matrix, disagreement hotspots, confidence correlations",
  "visualizations": ["heatmap", "scatter_plot", "distribution"]
}
```

---

### **Phase 3: Code Execution & Computational Analysis**

**Purpose:** Execute generated code to produce analytical insights

**Execution Methods:**

1. **Gemini Code Execution Framework** (Recommended):
   - Uses Gemini's built-in code execution
   - Sandboxed environment
   - Automatic result extraction
   - Supports pandas, numpy, scipy

2. **Local Sandboxed Execution** (Fallback):
   - Restricted namespace
   - Limited to safe libraries
   - Manual result extraction

**Outputs:**
- Statistical analysis results
- Computed metrics (agreement rates, correlations, etc.)
- Data summaries
- Trend insights

**Example Results:**
```
Model Agreement Analysis:
- GPT-5 vs Gemini: 87% agreement
- GPT-5 vs Claude: 82% agreement
- Gemini vs Claude: 91% agreement

High Disagreement Images:
- image_007.jpg: 3 different classifications
- image_012.jpg: Confidence variance 0.45

Confidence Correlation:
- GPT-5 confidence correlates with Gemini (r=0.73)
- Claude shows independent confidence patterns
```

---

### **Phase 4: Model Evaluation & Recommendations**

**Purpose:** Use LLM judge to evaluate model performance

**Process:**
1. **Judge LLM** (GPT-5-nano or Claude 4.5):
   - Analyzes all visual LLM outputs
   - Reviews computational analysis results
   - Evaluates model performance for specific task
   - Identifies strengths and weaknesses
   - Provides task-specific recommendations
   - Generates enhanced prompts for each model

**Outputs:**
- Best model for this task
- Model rankings with scores (0-100)
- Model-specific strengths
- General and task-specific recommendations
- Enhanced prompts for each model

**Example Evaluation:**
```json
{
  "best_model": "Gemini 2.5 Vision",
  "model_rankings": [
    {
      "model": "Gemini 2.5 Vision",
      "score": 92,
      "rationale": "Excellent medical image analysis, high confidence calibration"
    },
    {
      "model": "GPT-5 Vision",
      "score": 88,
      "rationale": "Strong object detection, good text recognition"
    },
    {
      "model": "Claude 4.5 Vision",
      "score": 85,
      "rationale": "Detailed descriptions, conservative confidence"
    }
  ],
  "strengths": {
    "Gemini 2.5 Vision": [
      "Best for medical imaging",
      "High accuracy on technical images",
      "Well-calibrated confidence scores"
    ],
    "GPT-5 Vision": [
      "Excellent text-in-image recognition",
      "Fast processing",
      "Good general-purpose performance"
    ]
  },
  "recommendations": {
    "general": "Use Gemini for medical/technical images, GPT-5 for general purpose",
    "task_specific": "For medical imaging, prioritize Gemini 2.5 with enhanced prompts"
  },
  "enhanced_prompts": {
    "Gemini 2.5 Vision": "Analyze this medical image focusing on anatomical structures, pathological findings, and image quality. Provide confidence scores for each finding.",
    "GPT-5 Vision": "Identify all visible objects, text, and anomalies in this image. Rate image quality and clarity."
  }
}
```

---

### **Phase 5: Interactive Follow-up**

**Purpose:** Enable interactive exploration and refinement

**Features:**

1. **Q&A Interface:**
   - Ask questions about analysis results
   - AI answers based on all available data
   - Conversation history maintained
   - Suggested follow-up actions

2. **Prompt Refinement:**
   - Use enhanced prompts from evaluation
   - Re-analyze specific images
   - Compare original vs enhanced results

3. **Selective Re-analysis:**
   - Choose specific images to re-analyze
   - Test different model combinations
   - Iterate without re-downloading images

**Example Q&A:**
```
Q: Why did GPT-5 and Gemini give different results for image_007.jpg?

A: GPT-5 classified it as "X-ray chest" with 0.85 confidence, while Gemini 
   classified it as "CT scan thorax" with 0.92 confidence. The difference 
   stems from image quality - it's actually a low-quality CT scan that 
   resembles an X-ray. Gemini's higher confidence suggests better 
   discrimination of subtle imaging modality differences.

Suggested Actions:
- View full analysis for image_007.jpg
- Compare model outputs side-by-side
- Re-analyze with enhanced prompt focusing on modality detection
```

---

## 🖥️ User Interface

### **6-Tab Results Display:**

#### **Tab 1: 📋 Summary**
- Total images analyzed
- Models used
- Success rates per model
- Cost and token usage
- Quick performance overview

#### **Tab 2: 📊 Data & Results**
- **Dataframe View:**
  - All results in tabular format
  - Filterable by image, model
  - Downloadable as CSV
  
- **Per-Image Detailed Results:**
  - Image preview
  - All model outputs side-by-side
  - Confidence scores
  - Expandable details

#### **Tab 3: 📈 Computational Analysis**
- **Analysis Planning:**
  - View generated analysis plan
  - See Python code
  - Expected outputs
  
- **Execution Results:**
  - Statistical analysis
  - Computed metrics
  - Data insights
  - Visualizations (coming soon)

- **Controls:**
  - Run analysis button
  - Re-run with different parameters
  - Cached results

#### **Tab 4: 🏆 Model Evaluation**
- **Best Model:**
  - Winner for this task
  - Overall score
  
- **Model Rankings:**
  - Ranked list with scores
  - Rationale for each ranking
  
- **Strengths:**
  - Model-specific strengths
  - Use case recommendations
  
- **Enhanced Prompts:**
  - Improved prompts for each model
  - Copy to use in re-analysis

#### **Tab 5: 💬 Interactive Q&A**
- **Conversation History:**
  - All previous Q&A exchanges
  - Expandable for details
  
- **Ask Questions:**
  - Free-form question input
  - AI answers with context
  - Suggested follow-up actions
  
- **Quick Actions:**
  - View specific images
  - Compare models
  - Re-analyze with enhancements

#### **Tab 6: 💾 Export**
- **JSON Export:**
  - Complete analysis data
  - Computational results
  - Evaluation results
  - Q&A history
  
- **CSV Export:**
  - Results table
  - Easy import to Excel/Sheets
  
- **PDF Report:** (Coming soon)
  - Formatted summary report
  - Visualizations included

---

## 🔧 Technical Implementation

### **Core Modules:**

1. **`core/visual_meta_analysis.py`:**
   - `plan_computational_analysis()` - LLM-based analysis planning
   - `execute_analysis_code()` - Code execution (Gemini or local)
   - `evaluate_visual_llm_performance()` - LLM judge evaluation

2. **`core/visual_qa_interface.py`:**
   - `answer_followup_question()` - Interactive Q&A
   - `refine_and_reanalyze()` - Re-analysis with enhanced prompts

3. **`ui/test6_advanced_results.py`:**
   - `display_advanced_results()` - Main 6-tab interface
   - Tab-specific display functions

### **Session State Caching:**

```python
# Cached to avoid re-running expensive operations
st.session_state.test6_computational_results  # Computational analysis
st.session_state.test6_evaluation_results     # Model evaluation
st.session_state.test6_qa_history             # Q&A conversation
```

### **API Usage:**

| Phase | API Calls | Purpose |
|-------|-----------|---------|
| Phase 1 | Visual LLMs (parallel) | Image analysis |
| Phase 2 | GPT-5-nano or Gemini | Analysis planning |
| Phase 3 | Gemini Code Execution | Run generated code |
| Phase 4 | GPT-5-nano or Claude | Model evaluation |
| Phase 5 | GPT-5-nano | Q&A responses |

---

## 📊 Example Workflow

### **Medical Image Analysis Preset:**

```
1. User selects "🏥 Medical Image Analysis" preset
2. System searches Linkup API for medical images
3. Downloads 18 valid images (X-rays, CT scans, MRIs)
4. Runs GPT-5, Gemini 2.5, Claude 4.5 in parallel
5. User chooses "Advanced Analysis Workflow"

Tab 1 - Summary:
  ✅ 18 images analyzed
  ✅ 3 models used
  ✅ 54 total analyses
  ✅ $0.23 total cost

Tab 2 - Data:
  📊 Dataframe with all 54 results
  🖼️ Per-image view with model comparisons

Tab 3 - Computational Analysis:
  🚀 User clicks "Run Computational Analysis"
  📋 Plan: "Compare model agreement on pathology detection"
  💻 Code: Statistical analysis of model outputs
  ✅ Results: "Gemini detected 23% more pathologies than GPT-5"

Tab 4 - Model Evaluation:
  🚀 User clicks "Run Model Evaluation"
  🏆 Best Model: Gemini 2.5 Vision (Score: 92/100)
  💪 Strengths: "Excellent pathology detection, high confidence"
  ✨ Enhanced Prompt: "Focus on anatomical structures and pathological findings..."

Tab 5 - Q&A:
  ❓ User asks: "Which images had the most disagreement?"
  💬 AI: "Images 7, 12, and 15 showed high disagreement..."
  💡 Suggested: "View detailed analysis for these images"

Tab 6 - Export:
  📥 Download complete analysis (JSON)
  📥 Download results table (CSV)
```

---

## ✅ Benefits

### **For Users:**
- ✅ **Comprehensive Analysis** - Beyond simple model comparison
- ✅ **Automated Insights** - LLM-generated computational analysis
- ✅ **Model Recommendations** - Know which model to use when
- ✅ **Interactive Exploration** - Ask questions, refine analysis
- ✅ **Complete Export** - All data and insights exportable

### **For Development:**
- ✅ **Modular Design** - Easy to extend with new analysis types
- ✅ **Cached Results** - Avoid expensive re-computation
- ✅ **Error Handling** - Graceful degradation
- ✅ **Cost Tracking** - Monitor API usage

---

## 🚀 Future Enhancements

1. **Advanced Visualizations:**
   - Interactive Plotly charts from computational analysis
   - Model agreement heatmaps
   - Confidence distribution plots

2. **PDF Report Generation:**
   - Formatted summary report
   - Embedded visualizations
   - Executive summary

3. **Batch Processing:**
   - Analyze multiple presets in sequence
   - Compare across different image sets

4. **Custom Analysis Code:**
   - User-provided Python code
   - Template library for common analyses

5. **Model Fine-tuning Suggestions:**
   - Identify training data gaps
   - Suggest fine-tuning datasets

---

**Last Updated:** 2025-10-02  
**Status:** ✅ Core implementation complete, visualizations in progress

