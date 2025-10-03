# Test 6: Complete Workflow Diagram

## 🔄 End-to-End Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEST 6: VISUAL LLM TESTING                   │
│                         Mode B Workflow                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: IMAGE COLLECTION                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Option A: Manual Upload          Option B: Preset Examples     │
│  ┌──────────────────┐            ┌──────────────────┐          │
│  │ User uploads     │            │ Select preset:   │          │
│  │ 3-20 images      │            │ • VR Avatars     │          │
│  │ (JPG/PNG)        │            │ • Medical Images │          │
│  └──────────────────┘            │ • Product Defects│          │
│                                   └──────────────────┘          │
│                                                                  │
│  Option C: Web Search (Future)                                  │
│  ┌──────────────────┐                                          │
│  │ Linkup API       │                                          │
│  │ Auto-download    │                                          │
│  └──────────────────┘                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: MODEL SELECTION                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Select 2-4 Visual LLM Models:                                  │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ ☑ GPT-5 Vision (OpenAI)                              │      │
│  │ ☑ Gemini 2.5 Vision (Google)                         │      │
│  │ ☑ Claude 4.5 Vision (Anthropic via OpenRouter)       │      │
│  │ ☑ Llama 3.2 Vision (Meta via OpenRouter)             │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                  │
│  Enter Analysis Task:                                           │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ "Analyze VR avatar for visual artifacts including    │      │
│  │  red lines, movement issues, and distortions.        │      │
│  │  Rate overall quality 1-5."                          │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: PARALLEL ANALYSIS                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   GPT-5     │  │   Gemini    │  │   Claude    │            │
│  │   Vision    │  │  2.5 Vision │  │ 4.5 Vision  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│         ↓                ↓                ↓                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Image 1     │  │ Image 1     │  │ Image 1     │            │
│  │ Image 2     │  │ Image 2     │  │ Image 2     │            │
│  │ Image 3     │  │ Image 3     │  │ Image 3     │            │
│  │ ...         │  │ ...         │  │ ...         │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                  │
│  Each model analyzes all images in parallel                     │
│  Extracts: ratings, artifacts, confidence, rationale            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: RESULTS COLLECTION                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Structured Results for Each Image:                             │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ {                                                     │      │
│  │   "image_name": "avatar_001.jpg",                    │      │
│  │   "model_results": {                                 │      │
│  │     "gpt5": {                                        │      │
│  │       "confidence": 0.85,                            │      │
│  │       "rationale": "...",                            │      │
│  │       "detected_artifacts": ["red lines"],           │      │
│  │       "movement_rating": 4.0,                        │      │
│  │       "visual_quality_rating": 3.5                   │      │
│  │     },                                               │      │
│  │     "gemini": { ... },                               │      │
│  │     "claude": { ... }                                │      │
│  │   }                                                  │      │
│  │ }                                                    │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                  │
│  ✅ Saved to Analysis History                                   │
│  ✅ Cost tracked in sidebar                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 5: 8-TAB RESULTS DISPLAY                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Tab 1: 📋 Summary & Performance                                │
│  ├─ Total images, models, cost                                  │
│  └─ Model performance table                                     │
│                                                                  │
│  Tab 2: 📊 Detailed Results                                     │
│  ├─ Per-image results                                           │
│  └─ Ground truth comparison                                     │
│                                                                  │
│  Tab 3: 📈 Visualizations                                       │
│  ├─ Agreement heatmap                                           │
│  ├─ Confidence distribution                                     │
│  └─ Performance comparison                                      │
│                                                                  │
│  Tab 4: 🎯 Synthesis & Insights ⭐ NEW!                         │
│  ├─ Model Rankings                                              │
│  ├─ Agreement Analysis                                          │
│  ├─ Actionable Insights                                         │
│  ├─ Prompt Optimization                                         │
│  └─ Advanced Visualizations                                     │
│                                                                  │
│  Tab 5: 🧠 Computational Analysis                               │
│  ├─ LLM-generated analysis code                                 │
│  └─ Gemini code execution                                       │
│                                                                  │
│  Tab 6: 🏆 Model Evaluation                                     │
│  ├─ LLM judge evaluation                                        │
│  └─ Model comparison                                            │
│                                                                  │
│  Tab 7: 💬 Interactive Q&A                                      │
│  ├─ Follow-up questions                                         │
│  └─ Context-aware responses                                     │
│                                                                  │
│  Tab 8: 💾 Export                                               │
│  └─ JSON/CSV export                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Synthesis Tab Deep Dive

```
┌─────────────────────────────────────────────────────────────────┐
│ 🎯 SYNTHESIS & INSIGHTS TAB                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Summary Metrics (Top Bar)                                      │
│  ┌──────────┬──────────┬──────────┬──────────┐                │
│  │ Total    │ Models   │ Best     │ Avg      │                │
│  │ Images   │ Tested   │ Model    │ Agreement│                │
│  │   10     │    4     │ GPT-5    │  82.3%   │                │
│  └──────────┴──────────┴──────────┴──────────┘                │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Sub-Tab 1: 📊 Model Rankings                            │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                          │   │
│  │  Performance Table:                                      │   │
│  │  ┌────┬────────┬───────┬──────┬────────┬─────────┐     │   │
│  │  │Rank│ Model  │Overall│ Conf │ Detail │Analyses │     │   │
│  │  ├────┼────────┼───────┼──────┼────────┼─────────┤     │   │
│  │  │ 1  │ GPT-5  │ 0.847 │ 85%  │ 0.623  │   10    │     │   │
│  │  │ 2  │ Gemini │ 0.812 │ 82%  │ 0.589  │   10    │     │   │
│  │  │ 3  │ Claude │ 0.798 │ 80%  │ 0.612  │   10    │     │   │
│  │  │ 4  │ Llama  │ 0.765 │ 77%  │ 0.534  │   10    │     │   │
│  │  └────┴────────┴───────┴──────┴────────┴─────────┘     │   │
│  │                                                          │   │
│  │  Complementary Strengths:                                │   │
│  │  ┌────────────────────────────────────────────────┐     │   │
│  │  │ GPT-5 Vision:                                  │     │   │
│  │  │ ✅ High confidence predictions                 │     │   │
│  │  │ ✅ Detailed explanations                       │     │   │
│  │  │                                                │     │   │
│  │  │ Gemini 2.5 Vision:                             │     │   │
│  │  │ ✅ Artifact detection                          │     │   │
│  │  │ ✅ Detailed explanations                       │     │   │
│  │  └────────────────────────────────────────────────┘     │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Sub-Tab 2: 🤝 Agreement Analysis                        │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                          │   │
│  │  Statistics:                                             │   │
│  │  ┌──────────┬──────────┬──────────┐                    │   │
│  │  │ Mean     │ Std Dev  │ Min      │                    │   │
│  │  │ 82.3%    │ 0.087    │ 68.5%    │                    │   │
│  │  └──────────┴──────────┴──────────┘                    │   │
│  │                                                          │   │
│  │  Pairwise Agreement:                                     │   │
│  │  ┌────────┬────────┬──────────┐                        │   │
│  │  │ Model1 │ Model2 │Agreement │                        │   │
│  │  ├────────┼────────┼──────────┤                        │   │
│  │  │ GPT-5  │ Gemini │  87.2%   │                        │   │
│  │  │ GPT-5  │ Claude │  85.1%   │                        │   │
│  │  │ GPT-5  │ Llama  │  78.9%   │                        │   │
│  │  └────────┴────────┴──────────┘                        │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Sub-Tab 3: 💡 Insights & Actions                        │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                          │   │
│  │  🔴 High Priority                                        │   │
│  │  ┌────────────────────────────────────────────────┐     │   │
│  │  │ Model Selection                                │     │   │
│  │  │ Insight: GPT-5 Vision performed best overall   │     │   │
│  │  │ Action: Use GPT-5 Vision as primary model      │     │   │
│  │  └────────────────────────────────────────────────┘     │   │
│  │                                                          │   │
│  │  ┌────────────────────────────────────────────────┐     │   │
│  │  │ Quality Control                                │     │   │
│  │  │ Insight: 2 images have low confidence          │     │   │
│  │  │ Action: Review avatar_003.jpg, avatar_007.jpg  │     │   │
│  │  └────────────────────────────────────────────────┘     │   │
│  │                                                          │   │
│  │  🟡 Medium Priority (expandable)                         │   │
│  │  🟢 Low Priority (expandable)                            │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Sub-Tab 4: ✨ Prompt Optimization                       │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                          │   │
│  │  Original Prompt:                                        │   │
│  │  ┌────────────────────────────────────────────────┐     │   │
│  │  │ "Analyze VR avatar for visual artifacts"      │     │   │
│  │  └────────────────────────────────────────────────┘     │   │
│  │                                                          │   │
│  │  Improved Prompt for Llama 3.2 Vision:                  │   │
│  │  ┌────────────────────────────────────────────────┐     │   │
│  │  │ "Analyze VR avatar for visual artifacts       │     │   │
│  │  │                                                │     │   │
│  │  │ Additional guidance:                           │     │   │
│  │  │ - Be specific and confident                    │     │   │
│  │  │ - Provide detailed explanations"               │     │   │
│  │  └────────────────────────────────────────────────┘     │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Sub-Tab 5: 📈 Visualizations                            │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                          │   │
│  │  Agreement Heatmap:                                      │   │
│  │  ┌────────────────────────────────────────────────┐     │   │
│  │  │        GPT-5  Gemini Claude Llama              │     │   │
│  │  │ GPT-5   1.00   0.87   0.85   0.79              │     │   │
│  │  │ Gemini  0.87   1.00   0.84   0.80              │     │   │
│  │  │ Claude  0.85   0.84   1.00   0.73              │     │   │
│  │  │ Llama   0.79   0.80   0.73   1.00              │     │   │
│  │  │                                                │     │   │
│  │  │ 🟢 Green = High agreement                      │     │   │
│  │  │ 🔴 Red = Low agreement                         │     │   │
│  │  └────────────────────────────────────────────────┘     │   │
│  │                                                          │   │
│  │  Confidence Correlation:                                 │   │
│  │  ┌────────────────────────────────────────────────┐     │   │
│  │  │ Scatter plot: Confidence vs Agreement          │     │   │
│  │  │ Pearson r = 0.68 (p = 0.031)                   │     │   │
│  │  │ ✅ Positive correlation (well-calibrated)      │     │   │
│  │  └────────────────────────────────────────────────┘     │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow

```
Images → Models → Results → Synthesis → Insights → Actions

┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Images  │ →  │  Models  │ →  │ Results  │ →  │Synthesis │
│  (10)    │    │   (4)    │    │  (40)    │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                       ↓
                                                 ┌──────────┐
                                                 │ Insights │
                                                 │ Actions  │
                                                 └──────────┘
```

**Example:**
- 10 images × 4 models = 40 analyses
- Synthesis processes all 40 analyses
- Generates rankings, insights, and recommendations
- User takes action based on insights

---

## 📊 Synthesis Algorithm

```
1. Normalize Agreement Data
   ├─ Fill NaN values with 0.0
   ├─ Calculate pairwise agreements
   └─ Create clean DataFrame

2. Rank Models
   ├─ Calculate overall score (confidence 60% + detail 40%)
   ├─ Sort by score
   └─ Identify best model

3. Identify Complementary Strengths
   ├─ Analyze high confidence rate
   ├─ Analyze detail score
   ├─ Analyze artifact detection
   └─ Assign unique strengths

4. Generate Insights
   ├─ Model selection (best model)
   ├─ Agreement analysis (task clarity)
   ├─ Ensemble strategy (complementary strengths)
   └─ Quality control (flagged images)

5. Optimize Prompts
   ├─ Analyze model weaknesses
   ├─ Generate targeted enhancements
   └─ Create improved prompts

6. Create Visualizations
   ├─ Agreement heatmap
   ├─ Confidence correlation
   └─ Statistical analysis
```

---

## 🎯 Decision Tree

```
After viewing synthesis results:

Is best model clear (score > 0.80)?
├─ YES → Use best model for future tasks
└─ NO → Consider ensemble approach

Is agreement high (> 80%)?
├─ YES → Task is clear, single model OK
└─ NO → Refine task or use multiple models

Are there flagged images?
├─ YES → Review manually
└─ NO → Proceed with confidence

Do models have complementary strengths?
├─ YES → Use ensemble for critical tasks
└─ NO → Stick with best model

Are prompts optimized?
├─ YES → Use as-is
└─ NO → Apply improved prompts
```

---

**This diagram shows the complete Test 6 workflow from image collection to actionable insights!**

