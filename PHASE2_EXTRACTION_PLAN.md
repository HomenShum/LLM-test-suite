# Phase 2: Aggressive Extraction Plan

## Goal
Reduce `streamlit_test_v5.py` from **10,215 lines** to **under 2,000 lines** (~80% reduction)

## Current Status
- ✅ Phase 1 Complete: Reduced from 12,609 to 10,215 lines (19% reduction)
- ✅ Created: config/scenarios.py, utils/visualizations.py, utils/gantt_charts.py, utils/ui_components.py
- ✅ Created: core/pricing.py, utils/model_discovery.py, utils/data_helpers.py, utils/helpers.py
- ⏳ Phase 2 In Progress: Need to extract ~8,200 more lines

## Files Created (Not Yet Removed from Main)

### Already Created:
1. **core/pricing.py** (391 lines) - Pricing and model discovery
2. **utils/model_discovery.py** (169 lines) - UI model discovery
3. **utils/data_helpers.py** (300 lines) - Data loading/saving
4. **utils/helpers.py** (250 lines) - Misc helpers
5. **core/orchestrator.py** (300 lines, partial) - Orchestrator classes

**Total created so far: ~1,410 lines**

## Remaining Extractions Needed

### 1. Complete core/orchestrator.py (~3,200 lines remaining)
**Lines in main file:** 3526-6808
**Content:**
- UnifiedOrchestrator class (massive ~3,280 lines)
- All orchestrator methods
- Leaf agent integration
- Research, analysis, and inference modes

**Strategy:** This is too large. Need to split into:
- `core/orchestrator_base.py` - Base classes and helpers
- `core/orchestrator_modes.py` - Inference, analysis, research modes
- `core/orchestrator_patterns.py` - Solo, subagent, multi-agent patterns

### 2. Create core/api_clients.py (~1,200 lines)
**Lines in main file:** 6981-7784
**Content:**
- `classify_with_openai()` (66 lines)
- `openai_structured_json()` (25 lines)
- `classify_with_gemini()` (60 lines)
- `classify_with_ollama()` (23 lines)
- `ollama_json()` (14 lines)
- `classify_with_openrouter()` (45 lines)
- `openrouter_json()` (66 lines)
- `generate_text_async()` (41 lines)
- `_classify_df_async()` (177 lines)
- `run_judge_flexible()` (20 lines)
- `run_judge_ollama()` (2 lines)
- `run_judge_openai()` (2 lines)
- `run_pruner()` (191 lines)
- `generate_synthetic_data()` (large function)

### 3. Extract remaining visualizations (~600 lines)
**Content:**
- `render_model_comparison_chart()` (98 lines) - lines 1763-1861
- `render_organized_results()` (200 lines) - lines 1861-2061
- `render_progress_replay()` (130 lines) - lines 2061-2191
- `render_universal_gantt_chart()` (175 lines) - lines 2469-2644

**Move to:** utils/visualizations.py (already exists)

### 4. Extract execution tracking (~500 lines)
**Lines in main file:** 298-490
**Content:**
- `ExecutionEvent` class
- `ExecutionTracker` class
- Event logging and tracking

**Move to:** utils/execution_tracker.py (new)

### 5. Extract Pydantic models (~100 lines)
**Lines in main file:** 1120-1235
**Content:**
- `Classification`
- `ClassificationWithConf`
- `SyntheticDataItem`
- `ToolCallSequenceItem`
- `PruningDataItem`
- `TestSummaryAndRefinement`
- `FactualConstraint`
- `ValidationResultArtifact`
- `convert_validation_to_artifact()`

**Move to:** core/models.py (new)

### 6. Extract report generation (~300 lines)
**Lines in main file:** 3021-3133, 6810-6875
**Content:**
- `generate_classification_report()`
- `get_structured_summary_and_refinement()`
- `display_final_summary_for_test()`
- `run_gemini_code_execution()`

**Move to:** utils/reporting.py (new)

### 7. Extract test execution functions (~1,500 lines)
**Content:**
- `run_classification_flow()` (large function)
- Test-specific execution logic
- Batch processing functions

**Move to:** core/test_runners.py (new)

## Extraction Order (Priority)

### Step 1: Extract Large, Self-Contained Blocks
1. ✅ Pricing and model discovery (DONE)
2. ✅ Data helpers (DONE)
3. ✅ Helper functions (DONE)
4. ⏳ Pydantic models → core/models.py
5. ⏳ API clients → core/api_clients.py
6. ⏳ Execution tracker → utils/execution_tracker.py

### Step 2: Extract Orchestrator (Split into 3 files)
7. ⏳ Orchestrator base classes → core/orchestrator_base.py
8. ⏳ Orchestrator modes → core/orchestrator_modes.py
9. ⏳ Orchestrator patterns → core/orchestrator_patterns.py

### Step 3: Extract Remaining Functions
10. ⏳ Remaining visualizations → utils/visualizations.py
11. ⏳ Report generation → utils/reporting.py
12. ⏳ Test runners → core/test_runners.py

### Step 4: Clean Up Main File
13. ⏳ Remove all extracted code
14. ⏳ Add proper imports
15. ⏳ Test functionality

## Expected Final Size

| Component | Lines |
|-----------|-------|
| **Main file (streamlit_test_v5.py)** | **~1,800** |
| - UI layout and tabs | ~400 |
| - Session state initialization | ~200 |
| - Tab content (9 tabs × ~100 lines) | ~900 |
| - Sidebar configuration | ~200 |
| - Misc glue code | ~100 |
| **Extracted modules** | **~10,400** |
| - core/orchestrator_*.py | ~3,500 |
| - core/api_clients.py | ~1,200 |
| - core/models.py | ~150 |
| - core/pricing.py | ~400 |
| - core/test_runners.py | ~1,500 |
| - utils/* | ~3,650 |
| **Total** | **~12,200** |

## Benefits

1. **Maintainability**: Each module has a single, clear responsibility
2. **Testability**: Can test each module independently
3. **Reusability**: Modules can be imported by other projects
4. **Readability**: Main file focuses on UI and orchestration
5. **Performance**: No impact on runtime performance
6. **Collaboration**: Multiple developers can work on different modules

## Next Actions

1. Create core/models.py for Pydantic models
2. Create core/api_clients.py for all API interaction functions
3. Create utils/execution_tracker.py for execution tracking
4. Split orchestrator into 3 files
5. Extract remaining visualizations
6. Create utils/reporting.py for report generation
7. Create core/test_runners.py for test execution
8. Remove extracted code from main file
9. Add imports to main file
10. Test all functionality

