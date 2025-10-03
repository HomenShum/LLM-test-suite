# Enhanced Interactive Q&A with Image Display

## 🎯 **Feature Enhancement**

The Interactive Q&A feature now supports **image-specific questions** with automatic image display in expandable sections.

---

## ✨ **New Capabilities**

### **1. General Analysis Questions** (Existing)
Ask about overall analysis results:
- *"Which model performed best overall?"*
- *"What were the main findings?"*
- *"How did GPT-5 compare to Gemini?"*

### **2. Image-Specific Questions** (NEW!)
Ask about specific images and see them displayed:
- *"Show me the analysis for urban_scene_003.jpg"*
- *"What did the models say about image_005.png?"*
- *"Display the results for avatar_012.jpg"*

### **3. Comparative Image Questions** (NEW!)
Compare analyses across images:
- *"Why did GPT-5 and Gemini disagree on image_003?"*
- *"Show me images where models had low confidence"*
- *"Which images had the most artifacts detected?"*

---

## 🖼️ **Image Display in Q&A**

### **Automatic Detection**

The system automatically detects when you're asking about specific images by looking for:

✅ **Image names:** `urban_scene_003.jpg`, `image_005.png`, `avatar_012.jpg`
✅ **Display keywords:** `show me`, `display`, `view`, `look at`, `see`
✅ **Image keywords:** `image`, `picture`, `photo`
✅ **File extensions:** `.jpg`, `.png`, `.jpeg`

---

### **Display Format**

When you ask about an image, the answer includes:

```
📜 Conversation History

Q1: Show me the analysis for urban_scene_003.jpg

**Question:** Show me the analysis for urban_scene_003.jpg

**Answer:** 
The analysis for urban_scene_003.jpg shows a busy urban street scene. 
All three models detected multiple pedestrians and vehicles. GPT-5 Vision 
identified 12 people with 92% confidence, while Gemini 2.5 Vision found 
11 people with 88% confidence. Both models agreed on the presence of 
taxis and tour buses...

**📸 Relevant Images:**

🖼️ urban_scene_003.jpg [expandable]
  ┌─────────────┬──────────────────────────────────────┐
  │   Image     │  Model Analyses                      │
  │             │                                      │
  │  ┌──────┐   │  **GPT-5 Vision (gpt-5-nano):**     │
  │  │      │   │  The image shows a busy urban       │
  │  │      │   │  street with multiple pedestrians...│
  │  └──────┘   │  Confidence: 92%                     │
  │             │                                      │
  │             │  **Gemini 2.5 Vision:**              │
  │             │  Urban scene with clear visibility  │
  │             │  of people and vehicles...           │
  │             │  Confidence: 88%                     │
  │             │                                      │
  │             │  **Llama 3.2 Vision:**               │
  │             │  Street scene with pedestrians...    │
  │             │  Confidence: 85%                     │
  └─────────────┴──────────────────────────────────────┘

**Suggested Actions:**
- Compare this image with others
- Ask about other images
- View full analysis for this image
```

---

## 🔧 **Implementation Details**

### **1. Enhanced UI (`ui/test6_advanced_results.py`)**

**Updated Info Message:**
```python
st.info("💡 Ask questions about the analysis results **OR** specific images. 
        The AI will answer based on all available data and show relevant images.")
```

**Added Example Questions:**
```python
st.caption("💡 **Examples:**")
st.caption("- General: *Which model performed best overall?*")
st.caption("- Image-specific: *Show me the analysis for urban_scene_003.jpg*")
st.caption("- Comparison: *Why did GPT-5 and Gemini disagree on image_005?*")
```

**Image Display in History:**
```python
# Display relevant images if any
if exchange.get('relevant_images'):
    st.markdown("**📸 Relevant Images:**")
    for img_data in exchange['relevant_images']:
        with st.expander(f"🖼️ {img_data['image_name']}", expanded=False):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display image
                img = Image.open(img_data['image_path'])
                st.image(img, use_container_width=True)
            
            with col2:
                # Display model analyses
                for model_name, analysis in img_data.get('model_results', {}).items():
                    st.markdown(f"**{model_name}:**")
                    st.caption(f"{analysis.rationale[:200]}...")
                    st.caption(f"Confidence: {analysis.confidence:.0%}")
```

---

### **2. Enhanced Q&A Logic (`core/visual_qa_interface.py`)**

**New Function: `_extract_relevant_images()`**

```python
def _extract_relevant_images(
    question: str,
    visual_llm_outputs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract images relevant to the question.
    
    Returns list of dicts with:
    - image_name: Name of the image
    - image_path: Path to the image file
    - model_results: Analysis results from all models
    """
    relevant_images = []
    question_lower = question.lower()
    
    # Keywords that indicate image-specific questions
    image_keywords = [
        "show me", "display", "look at", "view", "see",
        "image", "picture", "photo", ".jpg", ".png", ".jpeg"
    ]
    
    # Check if question is asking about images
    is_image_question = any(keyword in question_lower for keyword in image_keywords)
    
    if not is_image_question:
        return []
    
    # Extract specific image names mentioned in question
    for output in visual_llm_outputs:
        image_name = output.get("image_name", "")
        image_path = output.get("image_path", "")
        
        # Check if image name is mentioned in question
        if image_name.lower() in question_lower:
            relevant_images.append({
                "image_name": image_name,
                "image_path": image_path,
                "model_results": output.get("model_results", {})
            })
    
    return relevant_images[:5]  # Limit to 5 images
```

**Updated System Prompt:**
```python
**When asked about specific images:**
- Reference the image by name
- Summarize key findings from all models
- Highlight agreements and disagreements
- Note: The UI will automatically display the image alongside your answer
```

**Updated Response:**
```python
return {
    "answer": answer,
    "relevant_data": relevant_data,
    "relevant_images": relevant_images,  # ← NEW!
    "suggested_actions": suggested_actions,
    "timestamp": datetime.now().isoformat()
}
```

---

## 📊 **Example Interactions**

### **Example 1: Specific Image Request**

**User Question:**
```
Show me the analysis for urban_scene_003.jpg
```

**AI Response:**
```
The analysis for urban_scene_003.jpg shows a busy urban street scene with 
multiple pedestrians and vehicles. Here's what each model found:

**GPT-5 Vision (gpt-5-nano):**
- Detected 12 people with 92% confidence
- Identified taxis and tour buses
- Noted clear architectural details

**Gemini 2.5 Vision (gemini-2.5-flash-lite):**
- Found 11 people with 88% confidence
- Detected similar vehicles
- Highlighted good lighting conditions

**Llama 3.2 Vision (meta-llama/llama-3.2-90b-vision-instruct):**
- Counted 10 people with 85% confidence
- Identified street scene elements
- Noted pedestrian activity

All models agreed on the general scene composition, with minor differences 
in people count (likely due to occlusion and distance).
```

**Displayed Images:**
- 🖼️ urban_scene_003.jpg (with full model analyses in expandable section)

**Suggested Actions:**
- Compare this image with others
- Ask about other images
- View full analysis for this image

---

### **Example 2: Comparative Question**

**User Question:**
```
Why did GPT-5 and Gemini disagree on image_005?
```

**AI Response:**
```
GPT-5 and Gemini had different interpretations of image_005 due to:

1. **Object Count Difference:**
   - GPT-5: Detected 8 objects
   - Gemini: Detected 6 objects
   - Likely due to different confidence thresholds

2. **Classification Disagreement:**
   - GPT-5: Classified as "urban scene"
   - Gemini: Classified as "street photography"
   - Both are valid interpretations

3. **Confidence Levels:**
   - GPT-5: 87% confidence
   - Gemini: 82% confidence
   - GPT-5 was slightly more certain

The disagreement is minor and both analyses are reasonable given the 
image content.
```

**Displayed Images:**
- 🖼️ image_005.png (showing both GPT-5 and Gemini analyses side-by-side)

**Suggested Actions:**
- View detailed comparison matrix
- Compare with similar images
- Generate comparison visualization

---

### **Example 3: General Question (No Images)**

**User Question:**
```
Which model performed best overall?
```

**AI Response:**
```
Based on the comprehensive analysis:

**Best Overall: GPT-5 Vision (gpt-5-nano)**
- Highest average confidence: 89.2%
- Most detailed explanations
- Best artifact detection

**Runner-up: Gemini 2.5 Vision (gemini-2.5-flash-lite)**
- Strong performance: 85.7% confidence
- Fastest processing time
- Most cost-effective

**Third: Llama 3.2 Vision**
- Good performance: 82.3% confidence
- Competitive pricing
- Solid general analysis

Recommendation: Use GPT-5 for critical analysis, Gemini for high-volume 
processing.
```

**Displayed Images:**
- None (general question, no specific images)

**Suggested Actions:**
- View model evaluation report
- Ask about specific images
- Request visualization

---

## 🎯 **Benefits**

### **1. Enhanced User Experience**
- ✅ See images directly in Q&A conversation
- ✅ No need to switch between tabs
- ✅ Context-aware image display

### **2. Better Understanding**
- ✅ Visual confirmation of what's being discussed
- ✅ Side-by-side model comparison
- ✅ Easier to spot agreements/disagreements

### **3. Efficient Workflow**
- ✅ Ask about images by name
- ✅ Get instant visual feedback
- ✅ Explore results interactively

---

## 📁 **Files Modified**

### **1. `ui/test6_advanced_results.py`**

**Changes:**
- **Lines 1135-1184:** Enhanced Q&A tab with image display
- **Lines 1186-1234:** Added example questions and image storage

**Key Updates:**
- Added image display in conversation history
- Added example questions for users
- Store `relevant_images` in Q&A history

---

### **2. `core/visual_qa_interface.py`**

**Changes:**
- **Lines 48-72:** Enhanced system prompt with image capabilities
- **Lines 81-103:** Added `relevant_images` to response
- **Lines 141-235:** Added `_extract_relevant_images()` function
- **Lines 250-287:** Updated suggested actions for image questions

**Key Updates:**
- Detect image-specific questions
- Extract relevant images from results
- Return image data with answers

---

## ✅ **Summary**

### **Before:**
```
💬 Interactive Q&A
💡 Ask questions about the analysis results.

Q: Show me urban_scene_003.jpg
A: The analysis shows... [text only, no image]
```

### **After:**
```
💬 Interactive Q&A
💡 Ask questions about the analysis results OR specific images.

Examples:
- General: Which model performed best?
- Image-specific: Show me urban_scene_003.jpg
- Comparison: Why did models disagree on image_005?

Q: Show me urban_scene_003.jpg
A: The analysis shows... [text answer]

📸 Relevant Images:
🖼️ urban_scene_003.jpg [expandable]
  [Image displayed with all model analyses]
```

---

**Status:** ✅ **IMPLEMENTED**
**Last Updated:** 2025-10-03
**Files Modified:** 
- `ui/test6_advanced_results.py`
- `core/visual_qa_interface.py`

