# 🧪 Testing Guide - LLM Test Suite Refactoring

## Overview

This guide provides step-by-step instructions for testing the refactored LLM Test Suite to ensure all functionality has been preserved after the aggressive code extraction.

---

## 📋 Pre-Testing Checklist

Before running tests, ensure:

- [ ] All extracted modules are in place (`core/`, `utils/`, `config/`)
- [ ] Imports have been added to `streamlit_test_v5.py`
- [ ] Environment variables are set (`.env` file configured)
- [ ] Dependencies are installed (`pip install -r requirements.txt`)
- [ ] Backup files are available for rollback if needed

---

## 🔧 Setup Instructions

### 1. Verify File Structure

```bash
# Check that all modules exist
ls core/
# Should show: __init__.py, pricing.py, models.py, orchestrator.py, 
#              unified_orchestrator.py, api_clients.py, test_runners.py

ls utils/
# Should show: __init__.py, model_discovery.py, data_helpers.py, helpers.py,
#              execution_tracker.py, advanced_visualizations.py, visualizations.py,
#              gantt_charts.py, ui_components.py, dashboard_logger.py, stateful_components.py

ls config/
# Should show: __init__.py, scenarios.py
```

### 2. Verify Imports

Check that `streamlit_test_v5.py` has all necessary imports:

```python
# Core imports
from core.models import (...)
from core.pricing import (...)
from core.unified_orchestrator import UnifiedOrchestrator
from core.api_clients import (...)
from core.test_runners import run_classification_flow

# Utils imports
from utils.model_discovery import (...)
from utils.data_helpers import (...)
from utils.helpers import (...)
from utils.execution_tracker import (...)
from utils.advanced_visualizations import (...)
from utils.visualizations import (...)
from utils.gantt_charts import (...)
from utils.ui_components import (...)

# Config imports
from config.scenarios import (...)
```

### 3. Syntax Validation

```bash
# Run Python syntax check
python -m py_compile streamlit_test_v5.py

# Check all extracted modules
python -m py_compile core/*.py
python -m py_compile utils/*.py
python -m py_compile config/*.py
```

**Expected Result:** No syntax errors

---

## 🧪 Testing Phases

### Phase 1: Basic Functionality Tests (15 minutes)

#### Test 1.1: Application Startup

```bash
streamlit run streamlit_test_v5.py
```

**Expected Results:**
- [ ] Application starts without errors
- [ ] No import errors in console
- [ ] Sidebar loads correctly
- [ ] All tabs are visible

**Common Issues:**
- Missing imports → Add to main file
- Module not found → Check file paths
- Syntax errors → Run py_compile to identify

#### Test 1.2: Sidebar Configuration

**Steps:**
1. Open application
2. Check sidebar elements load
3. Verify model selection dropdowns populate
4. Test API key configuration fields

**Expected Results:**
- [ ] OpenRouter model dropdown shows models
- [ ] OpenAI model dropdown shows models
- [ ] Third model selection works
- [ ] Row limit selector works
- [ ] API routing mode toggle works

#### Test 1.3: Dataset Loading

**Steps:**
1. Navigate to each test tab
2. Check if datasets load automatically
3. Verify dataset preview displays

**Expected Results:**
- [ ] Classification dataset loads (Test 1, 2, 3)
- [ ] Tool sequence dataset loads (Test 3)
- [ ] Context pruning dataset loads (Test 4)
- [ ] No errors in console

---

### Phase 2: Core Functionality Tests (30 minutes)

#### Test 2.1: Test 1 - Basic Classification

**Steps:**
1. Navigate to "Test 1: Classification" tab
2. Select models (OpenRouter + OpenAI)
3. Click "▶️ Run Test 1"
4. Wait for completion

**Expected Results:**
- [ ] Test executes without errors
- [ ] Progress indicators show
- [ ] Results display in organized tabs
- [ ] F1 scores calculated correctly
- [ ] Latency metrics shown
- [ ] Cost tracking updates
- [ ] Visualizations render (charts, confusion matrix)

**Validation:**
- Check console for errors
- Verify results DataFrame has expected columns
- Confirm cost tracker shows API costs

#### Test 2.2: Test 2 - Advanced Ensembling

**Steps:**
1. Navigate to "Test 2: Advanced Ensembling" tab
2. Enable third model
3. Select all three models
4. Click "▶️ Run Test 2"

**Expected Results:**
- [ ] All three models execute
- [ ] Ensemble logic runs
- [ ] Weighted voting works
- [ ] Per-class F1 weighting applied
- [ ] Results show ensemble column
- [ ] Comparison charts display

#### Test 2.3: Test 3 - Judge Evaluation

**Steps:**
1. Navigate to "Test 3: Judge" tab
2. Configure judge model
3. Run test

**Expected Results:**
- [ ] Judge model evaluates results
- [ ] Judge rationale displayed
- [ ] Judge accuracy calculated
- [ ] Tool sequence test works

#### Test 2.4: Test 4 - Context Pruning

**Steps:**
1. Navigate to "Test 4: Context Pruning" tab
2. Select pruner model
3. Run test

**Expected Results:**
- [ ] Pruning logic executes
- [ ] Action accuracy calculated
- [ ] Key similarity metrics shown
- [ ] Results saved correctly

#### Test 2.5: Test 5 - Agent Orchestration

**Steps:**
1. Navigate to "Test 5: Agent Self-Refinement" tab
2. Enter a research goal
3. Select orchestration mode
4. Run test

**Expected Results:**
- [ ] UnifiedOrchestrator initializes
- [ ] Agent execution begins
- [ ] Execution tracker logs events
- [ ] Gantt chart displays timeline
- [ ] Results synthesized correctly
- [ ] Code execution works (if applicable)

---

### Phase 3: Advanced Feature Tests (45 minutes)

#### Test 3.1: Visualization Functions

**Test each visualization:**
- [ ] `render_kpi_metrics()` - KPI dashboard
- [ ] `render_cost_dashboard()` - Cost tracking
- [ ] `render_model_comparison_chart()` - Model comparison
- [ ] `render_organized_results()` - Result tabs
- [ ] `render_progress_replay()` - Progress animation
- [ ] `render_universal_gantt_chart()` - Timeline
- [ ] `render_agent_gantt_chart()` - Agent timeline
- [ ] `render_test5_gantt_chart()` - Test 5 timeline

**Expected Results:**
- All charts render without errors
- Interactive features work (hover, zoom, pan)
- Data displays correctly

#### Test 3.2: Data Generation

**Steps:**
1. Navigate to dataset generation section
2. Generate synthetic classification data
3. Generate tool sequence data
4. Generate pruning data

**Expected Results:**
- [ ] `generate_synthetic_data()` works
- [ ] Datasets saved to files
- [ ] Metadata files created (.meta.json)
- [ ] Generated data loads correctly

#### Test 3.3: Cost Tracking

**Steps:**
1. Run multiple tests
2. Check cost dashboard
3. Verify cost calculations

**Expected Results:**
- [ ] Costs tracked per API call
- [ ] Total cost displayed
- [ ] Cost breakdown by model
- [ ] Pricing cache works (30-day TTL)

#### Test 3.4: Execution Tracking

**Steps:**
1. Run Test 5 with execution tracker
2. Check timeline visualization
3. Verify event logging

**Expected Results:**
- [ ] `ExecutionTracker` logs events
- [ ] Timeline exports correctly
- [ ] Task cards display
- [ ] Live status updates work

---

### Phase 4: Integration Tests (30 minutes)

#### Test 4.1: Multi-Model Execution

**Steps:**
1. Enable all providers (OpenRouter, OpenAI, Ollama)
2. Run Test 2 with all models
3. Verify concurrent execution

**Expected Results:**
- [ ] All models execute in parallel
- [ ] Rate limiting works (Semaphore)
- [ ] No race conditions
- [ ] Results merge correctly

#### Test 4.2: Error Handling

**Test error scenarios:**
- [ ] Invalid API key → Shows error message
- [ ] Network timeout → Retries with `_retry()`
- [ ] Invalid model ID → Validation error
- [ ] Empty dataset → Warning displayed
- [ ] Malformed JSON → Pydantic validation catches

#### Test 4.3: Session State Management

**Steps:**
1. Run Test 1
2. Switch to Test 2
3. Return to Test 1
4. Verify state preserved

**Expected Results:**
- [ ] Results persist across tab switches
- [ ] Configuration saved
- [ ] Cost tracker maintains state
- [ ] Execution history preserved

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'core.unified_orchestrator'
```

**Solution:**
1. Check file exists: `ls core/unified_orchestrator.py`
2. Verify `__init__.py` exists in `core/`
3. Add import to main file:
   ```python
   from core.unified_orchestrator import UnifiedOrchestrator
   ```

#### Issue: Missing Variables

**Symptoms:**
```
NameError: name 'GEMINI_MODEL_METADATA' is not defined
```

**Solution:**
Add variable definitions after imports:
```python
from core.pricing import fetch_gemini_models_from_linkup
GEMINI_MODELS_FULL = fetch_gemini_models_from_linkup()
GEMINI_MODEL_METADATA = {
    model_id: {
        'context': meta.get('context', 'N/A'),
        'input_cost': meta.get('input_cost', 'N/A'),
        'output_cost': meta.get('output_cost', 'N/A')
    }
    for model_id, meta in GEMINI_MODELS_FULL.items()
}
```

#### Issue: Function Not Found

**Symptoms:**
```
NameError: name 'run_classification_flow' is not defined
```

**Solution:**
Add to imports:
```python
from core.test_runners import run_classification_flow
```

#### Issue: Syntax Errors

**Symptoms:**
```
SyntaxError: invalid syntax (line 520)
```

**Solution:**
1. Run: `python -m py_compile streamlit_test_v5.py`
2. Check for:
   - Stray decorators (`@dataclass` without class)
   - Mismatched indentation
   - Incomplete function definitions
3. Restore from backup if needed

---

## ✅ Test Completion Checklist

### Basic Tests
- [ ] Application starts successfully
- [ ] All tabs load without errors
- [ ] Sidebar configuration works
- [ ] Datasets load correctly

### Core Functionality
- [ ] Test 1 (Classification) runs successfully
- [ ] Test 2 (Ensembling) runs successfully
- [ ] Test 3 (Judge) runs successfully
- [ ] Test 4 (Pruning) runs successfully
- [ ] Test 5 (Orchestration) runs successfully

### Advanced Features
- [ ] All visualizations render correctly
- [ ] Data generation works
- [ ] Cost tracking accurate
- [ ] Execution tracking functional

### Integration
- [ ] Multi-model execution works
- [ ] Error handling robust
- [ ] Session state preserved
- [ ] No memory leaks

---

## 📊 Test Results Template

```markdown
# Test Results - [Date]

## Environment
- Python Version: 
- Streamlit Version:
- OS:

## Test Summary
- Total Tests: 
- Passed: ✅
- Failed: ❌
- Skipped: ⏭️

## Detailed Results

### Phase 1: Basic Functionality
- Application Startup: ✅/❌
- Sidebar Configuration: ✅/❌
- Dataset Loading: ✅/❌

### Phase 2: Core Functionality
- Test 1: ✅/❌
- Test 2: ✅/❌
- Test 3: ✅/❌
- Test 4: ✅/❌
- Test 5: ✅/❌

### Phase 3: Advanced Features
- Visualizations: ✅/❌
- Data Generation: ✅/❌
- Cost Tracking: ✅/❌
- Execution Tracking: ✅/❌

### Phase 4: Integration
- Multi-Model: ✅/❌
- Error Handling: ✅/❌
- Session State: ✅/❌

## Issues Found
1. [Issue description]
   - Severity: High/Medium/Low
   - Status: Open/Fixed
   - Solution: [Description]

## Recommendations
- [Recommendation 1]
- [Recommendation 2]

## Sign-off
Tested by: [Name]
Date: [Date]
Status: ✅ Approved / ❌ Needs Work
```

---

## 🚀 Next Steps After Testing

1. **If All Tests Pass:**
   - Update documentation
   - Create release notes
   - Deploy to production
   - Archive old backups

2. **If Tests Fail:**
   - Document failures
   - Restore from backup if critical
   - Fix issues systematically
   - Re-run tests

3. **Performance Optimization:**
   - Profile slow functions
   - Optimize database queries
   - Cache expensive operations
   - Monitor memory usage

---

**Testing Guide Version:** 1.0  
**Last Updated:** October 2, 2025  
**Status:** Ready for Use

