# Main File Cleanup Checklist

## Goal
Remove extracted code from `streamlit_test_v5.py` to reduce from **10,215 lines** to **under 2,000 lines**.

## Extraction Files Created ✅

1. **config/scenarios.py** (250 lines) - Static configuration
2. **core/pricing.py** (391 lines) - Pricing and model discovery
3. **core/models.py** (145 lines) - Pydantic models
4. **core/orchestrator.py** (300 lines, partial) - Orchestrator base classes
5. **core/api_clients.py** (411 lines) - API client functions
6. **utils/model_discovery.py** (169 lines) - UI model discovery
7. **utils/data_helpers.py** (300 lines) - Data loading/saving
8. **utils/helpers.py** (250 lines) - Misc helpers
9. **utils/execution_tracker.py** (180 lines) - Execution tracking
10. **utils/visualizations.py** (450 lines) - Plotly charts
11. **utils/gantt_charts.py** (300 lines) - Gantt visualizations
12. **utils/ui_components.py** (300 lines) - UI components

**Total extracted: ~3,446 lines in modules**

## Code to Remove from Main File

### 1. Execution Tracker (lines 298-399) - ~100 lines ✅
- `ExecutionEvent` class
- `ExecutionTracker` class
**Replacement:** `from utils.execution_tracker import ExecutionEvent, ExecutionTracker`

### 2. Pricing Functions (lines 490-1096) - ~606 lines ✅
- `_load_pricing_from_disk()`
- `_save_pricing_to_disk()`
- `fetch_openrouter_pricing()`
- `_to_openrouter_model_id()`
- `_to_native_model_id()`
- `_get_provider_from_model_id()`
- `_fetch_models_from_openrouter()`
- `fetch_gemini_models_from_linkup()`
- `_get_default_gemini_models()`
- `_parse_gemini_models_from_linkup()`
- `custom_gemini_price_lookup()`
- `fetch_openai_models_from_linkup()`
- `_get_default_openai_models()`
- `_parse_openai_models_from_linkup()`
- `get_all_available_models()`
- `custom_openrouter_price_lookup()`
**Replacement:** `from core.pricing import *`

### 3. Pydantic Models (lines 1120-1232) - ~112 lines ✅
- `Classification`
- `ClassificationWithConf`
- `SyntheticDataItem`
- `ToolCallSequenceItem`
- `PruningDataItem`
- `TestSummaryAndRefinement`
- `FactualConstraint`
- `ValidationResultArtifact`
- `convert_validation_to_artifact()`
**Replacement:** `from core.models import *`

### 4. Model Discovery UI (lines 1235-1415) - ~180 lines ✅
- `fetch_openrouter_models_for_ui()`
- `fetch_openai_models()`
- `get_third_model_display_name()`
- `_normalize_ollama_root()`
**Replacement:** `from utils.model_discovery import *`

### 5. Data Helpers (lines 1415-1742) - ~327 lines ✅
- `_subset_for_run()`
- `_style_selected_rows()`
- `save_results_df()`
- `ensure_dataset_directory()`
- `save_dataset_to_file()`
- `load_classification_dataset()`
- `load_tool_sequence_dataset()`
- `_normalize_label()`
- `load_context_pruning_dataset()`
- `_load_df_from_path()`
- `auto_generate_default_datasets()`
- `check_and_generate_datasets()`
- `_allowed_labels()`
- `_retry()`
**Replacement:** `from utils.data_helpers import *` and `from utils.helpers import _retry`

### 6. Visualizations Already Extracted (lines 1761-2644) - ~883 lines ✅
- `visualize_dataset_composition()` - ALREADY REMOVED
- `render_model_comparison_chart()` - lines 1763-1861
- `render_organized_results()` - lines 1861-2061
- `render_progress_replay()` - lines 2061-2191
- `render_agent_gantt_chart()` - ALREADY IN utils/gantt_charts.py
- `render_test5_gantt_chart()` - ALREADY IN utils/gantt_charts.py
- `render_universal_gantt_chart()` - lines 2469-2644
- `generate_gantt_data()` - lines 2644-2703
- `render_task_cards()` - lines 2703-2763
- `render_single_task_card()` - lines 2763-2830
- `render_live_agent_status()` - lines 2830-2866
- `render_agent_task_cards()` - lines 2866-2920
**Replacement:** Already imported from utils/visualizations.py and utils/gantt_charts.py

### 7. Config Helpers (lines 2920-3020) - ~100 lines ✅
- `capture_run_config()`
- `display_run_config()`
- `_non_empty()`
**Replacement:** `from utils.helpers import capture_run_config, display_run_config, _non_empty`

### 8. Report Generation (lines 3021-3133) - ~112 lines
- `generate_classification_report()`
- `run_gemini_code_execution()`
**Move to:** utils/reporting.py (NEW)

### 9. Orchestrator Classes (lines 3239-6808) - ~3,569 lines ⚠️
- `Budget`
- `TurnMetrics`
- `OrchestratorResult`
- `Task`
- `VerificationResult`
- `TaskCache`
- `KnowledgeIndex`
- `FineTuneDatasetCollector`
- `AgentCoordinationPattern`
- `GeminiLLMClient`
- `GeminiTaskPlanner`
- `GeminiResultSynthesizer`
- `UnifiedOrchestrator` (MASSIVE CLASS)
**Status:** Partially extracted to core/orchestrator.py
**Action:** Need to complete extraction or leave in main file for now

### 10. Summary Helpers (lines 6810-6875) - ~65 lines
- `get_structured_summary_and_refinement()`
- `display_final_summary_for_test()`
**Move to:** utils/reporting.py (NEW)

### 11. Test Execution (lines 6875-6981) - ~106 lines
- `run_classification_flow()`
**Move to:** core/test_runners.py (NEW) or leave in main

### 12. API Clients (lines 6981-7784) - ~803 lines ✅
- `classify_with_openai()`
- `openai_structured_json()`
- `classify_with_gemini()`
- `classify_with_ollama()`
- `ollama_json()`
- `classify_with_openrouter()`
- `openrouter_json()`
- `generate_text_async()`
- `_classify_df_async()`
- `_smarter_weighted_pick_row()`
- `run_judge_flexible()`
- `run_judge_ollama()`
- `run_judge_openai()`
- `run_pruner()`
- `generate_synthetic_data()`
**Replacement:** `from core.api_clients import *`

## Removal Strategy

### Phase 1: Remove Small, Self-Contained Blocks (Quick Wins)
1. ✅ Remove Execution Tracker (lines 298-399)
2. ✅ Remove Pricing Functions (lines 490-1096)
3. ✅ Remove Pydantic Models (lines 1120-1232)
4. ✅ Remove Model Discovery UI (lines 1235-1415)
5. ✅ Remove Data Helpers (lines 1415-1742)
6. ✅ Remove duplicate visualizations (lines 1763-2920)
7. ✅ Remove Config Helpers (lines 2920-3020)

**Expected reduction: ~2,500 lines**

### Phase 2: Remove API Clients
8. ✅ Remove API client functions (lines 6981-7784)

**Expected reduction: ~800 lines**

### Phase 3: Handle Orchestrator (Decision Point)
9. **Option A:** Complete orchestrator extraction (saves ~3,500 lines but complex)
10. **Option B:** Leave orchestrator in main file for now (Test 5 only)

### Phase 4: Add Imports
11. Add all necessary imports at top of file
12. Update function calls to use imported modules

## Expected Final Size

**Conservative estimate (leaving orchestrator in main):**
- Current: 10,215 lines
- Remove Phases 1-2: -3,300 lines
- **Result: ~6,900 lines**

**Aggressive estimate (extracting orchestrator):**
- Current: 10,215 lines
- Remove Phases 1-3: -6,800 lines
- **Result: ~3,400 lines**

**Target: Under 2,000 lines**
- Need to extract orchestrator AND test execution logic
- Or simplify/consolidate test tabs

## Next Actions

1. Start systematic removal of extracted code
2. Add comprehensive imports
3. Test each removal to ensure no breakage
4. Decide on orchestrator extraction strategy

