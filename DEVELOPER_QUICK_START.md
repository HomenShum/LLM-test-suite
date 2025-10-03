# 🚀 Developer Quick Start Guide

## Overview

This guide helps developers quickly understand the refactored LLM Test Suite codebase and start contributing.

---

## 📁 Project Structure

```
LLM_test_suite/
├── streamlit_test_v5.py          # Main Streamlit application (10,020 lines)
├── cost_tracker.py                # Cost tracking utilities
├── leaf_agent_scaffold.py         # Agent framework
│
├── core/                          # Core business logic
│   ├── pricing.py                 # Pricing & model discovery
│   ├── models.py                  # Pydantic models
│   ├── orchestrator.py            # Base orchestrator classes
│   ├── unified_orchestrator.py    # Main orchestrator (3,341 lines)
│   ├── api_clients.py             # API interaction functions
│   └── test_runners.py            # Test execution logic
│
├── utils/                         # Utility functions
│   ├── model_discovery.py         # UI model discovery
│   ├── data_helpers.py            # Dataset management
│   ├── helpers.py                 # General utilities
│   ├── execution_tracker.py       # Execution tracking
│   ├── advanced_visualizations.py # Advanced charts
│   ├── visualizations.py          # Basic charts
│   ├── gantt_charts.py            # Timeline visualizations
│   └── ui_components.py           # Reusable UI components
│
├── config/                        # Configuration
│   └── scenarios.py               # Test scenarios & prompts
│
└── datasets/                      # Test datasets
    ├── classification_dataset.csv
    ├── tool_sequence_dataset.csv
    └── context_pruning_dataset.csv
```

---

## 🎯 Key Modules

### Core Modules

#### `core/pricing.py`
**Purpose:** Pricing cache and model discovery  
**Key Functions:**
- `fetch_openrouter_pricing()` - Get pricing from OpenRouter API
- `custom_openrouter_price_lookup()` - Custom price lookup
- `get_all_available_models()` - Get all available models

**Usage:**
```python
from core.pricing import get_all_available_models
models = get_all_available_models()
```

#### `core/models.py`
**Purpose:** Pydantic models for validation  
**Key Models:**
- `Classification` - Basic classification result
- `ClassificationWithConf` - Classification with confidence
- `SyntheticDataItem` - Generated data item
- `TestSummaryAndRefinement` - Test summary

**Usage:**
```python
from core.models import Classification
result = Classification(
    classification_result="positive",
    rationale="The text expresses positive sentiment"
)
```

#### `core/unified_orchestrator.py`
**Purpose:** Main orchestrator for agent execution  
**Key Class:** `UnifiedOrchestrator`  
**Modes:**
- Direct Inference (classification, prediction)
- Computational Analysis (statistics, simulations)
- Research Tasks (multi-source information gathering)

**Patterns:**
- Solo (single agent)
- Subagent (hierarchical delegation)
- Multi-Agent (peer collaboration)

**Usage:**
```python
from core.unified_orchestrator import UnifiedOrchestrator
from core.orchestrator import Budget

orchestrator = UnifiedOrchestrator(
    goal="Research the latest AI developments",
    budget=Budget(mode="turns", max_turns=5),
    mode="research",
    coordination_pattern="multi_agent"
)
result = await orchestrator.execute()
```

#### `core/api_clients.py`
**Purpose:** API interaction functions  
**Key Functions:**
- `classify_with_openai()` - OpenAI classification
- `classify_with_gemini()` - Gemini classification
- `classify_with_openrouter()` - OpenRouter classification
- `generate_text_async()` - Async text generation

**Usage:**
```python
from core.api_clients import classify_with_openai
result = await classify_with_openai(
    client=openai_client,
    model="gpt-4",
    query="Is this positive or negative?",
    allowed_labels=["positive", "negative"]
)
```

#### `core/test_runners.py`
**Purpose:** Test execution functions  
**Key Function:** `run_classification_flow()`

**Usage:**
```python
from core.test_runners import run_classification_flow
run_classification_flow(
    include_third_model=True,
    use_openai_override=True,
    openrouter_model_override="mistralai/mistral-small"
)
```

### Utils Modules

#### `utils/data_helpers.py`
**Purpose:** Dataset management  
**Key Functions:**
- `load_classification_dataset()` - Load classification data
- `save_dataset_to_file()` - Save dataset
- `_normalize_label()` - Normalize labels
- `auto_generate_default_datasets()` - Generate datasets

**Usage:**
```python
from utils.data_helpers import load_classification_dataset
df = load_classification_dataset()
```

#### `utils/execution_tracker.py`
**Purpose:** Track execution events  
**Key Classes:**
- `ExecutionEvent` - Single execution event
- `ExecutionTracker` - Event tracker

**Usage:**
```python
from utils.execution_tracker import ExecutionTracker, ExecutionEvent

tracker = ExecutionTracker()
tracker.emit(ExecutionEvent(
    test_name="Test 1",
    task_id="task_1",
    task_name="Classification",
    status="started",
    timestamp=time.time()
))
```

#### `utils/advanced_visualizations.py`
**Purpose:** Advanced visualization functions  
**Key Functions:**
- `render_model_comparison_chart()` - Compare models
- `render_organized_results()` - Organized result tabs
- `render_universal_gantt_chart()` - Timeline chart

**Usage:**
```python
from utils.advanced_visualizations import render_model_comparison_chart
render_model_comparison_chart(
    df=results_df,
    model_cols=["openrouter", "openai"],
    model_names=["Mistral", "GPT-4"]
)
```

---

## 🔧 Common Tasks

### Adding a New Test

1. **Create test function in main file:**
```python
def run_my_new_test():
    st.subheader("My New Test")
    
    if st.button("Run Test"):
        # Your test logic here
        pass
```

2. **Add tab in main file:**
```python
tabs = st.tabs([
    "Test 1", "Test 2", "Test 3", "Test 4", "Test 5", 
    "My New Test"  # Add here
])

with tabs[5]:  # New tab index
    run_my_new_test()
```

3. **Add to test runners if complex:**
```python
# In core/test_runners.py
def run_my_new_test_flow():
    # Complex test logic
    pass
```

### Adding a New Visualization

1. **Create function in `utils/advanced_visualizations.py`:**
```python
def render_my_chart(df: pd.DataFrame):
    """Render my custom chart."""
    fig = go.Figure()
    # Chart logic
    st.plotly_chart(fig, use_container_width=True)
```

2. **Import in main file:**
```python
from utils.advanced_visualizations import render_my_chart
```

3. **Use in test:**
```python
render_my_chart(results_df)
```

### Adding a New Model Provider

1. **Add API client function in `core/api_clients.py`:**
```python
async def classify_with_my_provider(
    client,
    model: str,
    query: str,
    allowed_labels: List[str]
) -> Classification:
    # API call logic
    pass
```

2. **Add pricing lookup in `core/pricing.py`:**
```python
def custom_my_provider_price_lookup(provider: str, model: str):
    # Pricing logic
    pass
```

3. **Add to UI in main file:**
```python
use_my_provider = st.checkbox("Use My Provider")
if use_my_provider:
    my_provider_model = st.selectbox("Model", options=models)
```

### Adding a New Pydantic Model

1. **Define in `core/models.py`:**
```python
class MyNewModel(BaseModel):
    field1: str = Field(description="Description")
    field2: int = Field(ge=0, description="Non-negative integer")
```

2. **Import in main file:**
```python
from core.models import MyNewModel
```

3. **Use for validation:**
```python
result = MyNewModel(field1="value", field2=42)
```

---

## 🐛 Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Session State

```python
st.write("Session State:", st.session_state)
```

### Inspect DataFrame

```python
st.write("DataFrame Info:")
st.write(df.info())
st.write(df.head())
```

### Profile Performance

```python
import time
start = time.time()
# Your code
print(f"Execution time: {time.time() - start:.2f}s")
```

### Check API Calls

```python
# In core/api_clients.py, add logging
print(f"API Call: {model}, Query: {query[:50]}...")
```

---

## 📝 Code Style Guidelines

### Naming Conventions

- **Functions:** `snake_case` (e.g., `load_dataset()`)
- **Classes:** `PascalCase` (e.g., `ExecutionTracker`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `OPENROUTER_MODEL`)
- **Private functions:** `_leading_underscore` (e.g., `_normalize_label()`)

### Docstrings

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param2 is negative
    """
    pass
```

### Type Hints

```python
from typing import List, Dict, Optional, Any

def process_data(
    data: List[Dict[str, Any]],
    config: Optional[Dict] = None
) -> pd.DataFrame:
    pass
```

---

## 🧪 Testing

### Run Syntax Check

```bash
python -m py_compile streamlit_test_v5.py
```

### Run Application

```bash
streamlit run streamlit_test_v5.py
```

### Test Specific Module

```python
# test_pricing.py
from core.pricing import get_all_available_models

def test_get_models():
    models = get_all_available_models()
    assert len(models) > 0
    assert "google/gemini-2.5-flash" in models
```

---

## 📚 Resources

### Documentation
- `FINAL_REFACTORING_REPORT.md` - Complete refactoring summary
- `TESTING_GUIDE.md` - Comprehensive testing guide
- `PHASE2_EXTRACTION_PLAN.md` - Extraction planning

### Key Files
- `streamlit_test_v5.py` - Main application
- `core/unified_orchestrator.py` - Orchestrator logic
- `core/api_clients.py` - API interactions

### External Resources
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Plotly Documentation](https://plotly.com/python/)

---

## 🤝 Contributing

### Before Making Changes

1. Create a backup:
   ```bash
   cp streamlit_test_v5.py streamlit_test_v5.py.backup_$(date +%Y%m%d_%H%M%S)
   ```

2. Create a new branch (if using git):
   ```bash
   git checkout -b feature/my-new-feature
   ```

3. Test your changes:
   ```bash
   python -m py_compile streamlit_test_v5.py
   streamlit run streamlit_test_v5.py
   ```

### After Making Changes

1. Update documentation
2. Add tests if applicable
3. Run full test suite
4. Create pull request (if using git)

---

## ❓ FAQ

**Q: Where should I add new utility functions?**  
A: Add to `utils/helpers.py` or create a new module in `utils/` if it's a distinct category.

**Q: How do I add a new Pydantic model?**  
A: Add to `core/models.py` and import in main file.

**Q: Where are the test datasets stored?**  
A: In `datasets/` directory. Use `utils/data_helpers.py` to load them.

**Q: How do I add a new visualization?**  
A: Add function to `utils/advanced_visualizations.py` or `utils/visualizations.py`.

**Q: What's the difference between `core/` and `utils/`?**  
A: `core/` contains business logic (models, API clients, orchestrator). `utils/` contains helper functions (data loading, visualization, tracking).

---

**Quick Start Guide Version:** 1.0  
**Last Updated:** October 2, 2025  
**Maintainer:** Development Team

