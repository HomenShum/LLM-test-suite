# streamlit_app.py
# ============================================================
# Structured classification + evaluation suite (Enhanced Version)
# - Uses your file context, env ingestion, and async style
# - Providers: OpenRouter + OpenAI
# - Tests:
#   1) CSV classify + F1/Latency (two models) + Error Analysis
#   2) Add third model + weighted pick by (per-class F1 * row confidence)
#   3) Mistral as Judge over 3 models + Judge evaluation
#   4) Quantitative context pruning + action decision test
# ============================================================

import os, asyncio, json, time, re
import pandas as pd
import streamlit as st
import httpx, nest_asyncio
from typing import List, Optional, Tuple, Dict, Any, Set
from pydantic import BaseModel, ValidationError, Field
from openai import AsyncOpenAI
from google import genai
from google.genai import types
from pathlib import Path

from dotenv import load_dotenv
from collections import Counter, defaultdict

# --- NEW IMPORTS for orchestrator infrastructure ---
from dataclasses import dataclass, field
from enum import Enum
import hashlib

# --- LEAF AGENT SCAFFOLD IMPORTS ---
# Only import what we directly use in streamlit_test_v5.py
# (Other classes like LeafAgent, SubTask, etc. are used internally by these)
from leaf_agent_scaffold import (
    SupervisorAgent,
    AgentType,
    WebResearchAgent,
    CodeExecutorAgent,
    ContentGeneratorAgent,
    ValidatorAgent,
    TaskPlanner,
    ResultSynthesizer,
    PolicyUpdater  # For self-correcting research pipeline
)

# --- STATEFUL COMPONENTS IMPORTS ---
from utils.dashboard_logger import DashboardLogger
from utils.stateful_components import (
    MemoryManager,
    SecurityAuditAgent,
    SelfCorrectionManager
)

# --- NEW IMPORTS for enhanced testing ---
from sklearn.metrics import classification_report, confusion_matrix

# --- PLOTLY IMPORTS for interactive visualizations ---
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings

# --- REFACTORED IMPORTS: Extracted modules ---
from config.scenarios import (
    PI_AGENT_GOAL_PROMPT,
    FOLDING_POLICY_BLOCK,
    CYBERSECURITY_GOAL_PROMPT,
    THREAT_POLICY_BLOCK,
    SMOKE_TEST_SCENARIOS,
    SUGGESTED_PROMPTS,
    DEFAULT_DATASET_PROMPTS,
    SKELETON_COLUMNS,
    ROW_LIMIT_OPTIONS,
    CANON_MAP as _CANON_MAP,
    TEST_FLOWS,
    JUDGE_SCHEMA,
    JUDGE_INSTRUCTIONS
)

from utils.visualizations import (
    render_test_flow_diagram,
    render_kpi_metrics,
    render_cost_dashboard,
    visualize_dataset_composition,
    render_model_comparison_chart
)

from utils.gantt_charts import (
    render_agent_gantt_chart,
    render_test5_gantt_chart
)
from utils.plotly_config import PLOTLY_CONFIG
from ui import agent_dashboard, data_generation, footer, sidebar, test_tabs

from utils.ui_components import (
    ModelSelector,
    ConfigDisplay,
    TestResultTabs
)

# --- NEW PHASE 2 IMPORTS: Additional extracted modules ---
from core.models import (
    Classification,
    ClassificationWithConf,
    SyntheticDataItem,
    ToolCallSequenceItem,
    PruningDataItem,
    TestSummaryAndRefinement,
    FactualConstraint,
    ValidationResultArtifact,
    convert_validation_to_artifact
)

from core.pricing import (
    fetch_openrouter_pricing,
    _to_openrouter_model_id,
    _to_native_model_id,
    _get_provider_from_model_id,
    custom_openrouter_price_lookup,
    custom_gemini_price_lookup,
    custom_openai_price_lookup,
    get_all_available_models,
    fetch_gemini_models_from_linkup
)

from utils.model_discovery import (
    fetch_openrouter_models_for_ui,
    fetch_openai_models,
    get_third_model_display_name,
    _normalize_ollama_root,
    OPENROUTER_MODEL,
    OPENAI_MODEL,
    THIRD_MODEL_KIND,
    THIRD_MODEL
)
from utils.model_metadata import load_model_metadata, OLLAMA_MODEL_METADATA

from utils.data_helpers import (
    ensure_dataset_directory,
    save_dataset_to_file,
    save_results_df,
    load_classification_dataset,
    load_tool_sequence_dataset,
    load_context_pruning_dataset,
    _load_df_from_path,
    auto_generate_default_datasets,
    check_and_generate_datasets,
    _allowed_labels,
    _subset_for_run,
    _style_selected_rows,
    _normalize_label,
    DATASET_DIR as DEFAULT_DATASET_DIR,
    CLASSIFICATION_DATASET_PATH as DEFAULT_CLASSIFICATION_DATASET_PATH,
    TOOL_SEQUENCE_DATASET_PATH as DEFAULT_TOOL_SEQUENCE_DATASET_PATH,
    CONTEXT_PRUNING_DATASET_PATH as DEFAULT_CONTEXT_PRUNING_DATASET_PATH
)
import utils.data_helpers as data_helpers

from utils.helpers import (
    _retry,
    enhance_prompt_with_user_input,
    capture_run_config,
    display_run_config,
    _non_empty
)

from utils.execution_tracker import (
    ExecutionEvent,
    ExecutionTracker
)

from core.api_clients import (
    classify_with_openai,
    classify_with_gemini,
    classify_with_ollama,
    classify_with_openrouter,
    openai_structured_json,
    openrouter_json,
    ollama_json,
    generate_text_async,
    generate_synthetic_data
)

# --- PHASE 3 IMPORTS: Aggressive extraction ---
from core.unified_orchestrator import UnifiedOrchestrator
from core import summaries, judges, api_clients, test_runners

from core.test_runners import run_classification_flow
from core.reporting import generate_classification_report

from utils.advanced_visualizations import (
    render_model_comparison_chart as render_advanced_model_comparison,
    render_organized_results,
    render_progress_replay,
    render_universal_gantt_chart,
    render_task_cards,
    render_single_task_card,
    render_live_agent_status,
    render_agent_task_cards
)

from core.orchestrator import (
    Budget,
    TurnMetrics,
    OrchestratorResult,
    Task,
    VerificationResult,
    TaskCache,
    KnowledgeIndex,
    AgentCoordinationPattern,
    GeminiLLMClient,
    GeminiTaskPlanner,
    GeminiResultSynthesizer
)

# ============================================================
# DEMONSTRATION SCENARIO CONFIGURATIONS
# ============================================================
# NOTE: Scenario configurations moved to config/scenarios.py
# Imported above as: PI_AGENT_GOAL_PROMPT, FOLDING_POLICY_BLOCK, etc.

# NOTE: run_live_smoke_test() function removed - orchestrator logic should be in core/
# If needed, this can be recreated as a helper in core/test_runners.py

# ============================================================
# HELPER FUNCTIONS
# ============================================================
# NOTE: enhance_prompt_with_user_input() moved to utils/helpers.py
# Already imported above

# ============================================================
# UNIVERSAL EXECUTION TRACKING SYSTEM
# ============================================================

# ExecutionEvent and ExecutionTracker moved to utils/execution_tracker.py

# --- COST TRACKER IMPORTS ---
from cost_tracker import CostTracker, combined_price_lookup, register_all_extractors


# --- TARGETED SUPPRESSION FOR PLOTLY/STREAMLIT KEYWORD ARGUMENTS ---
# Suppress the specific deprecation message and any Plotly deprecations that bubble up via Streamlit logs
warnings.filterwarnings(
    "ignore",
    message=r".*keyword arguments have been deprecated.*Use `config` instead.*",
)
# Also suppress general Plotly deprecation/future warnings that Streamlit may surface
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"^plotly(\.|$)")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"^plotly(\.|$)")

load_dotenv()  # load env vars
nest_asyncio.apply()  # patch loop for Streamlit


# ---------- Config / env ----------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
# --- FIX: Use latest GPT-5 series as default (OpenRouter format) ---
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "openai/gpt-5-mini")

# --- NEW: Dataset Directory Configuration ---
DATASET_DIR = st.secrets.get("DATASET_DIR", DEFAULT_DATASET_DIR)
data_helpers.configure_dataset_paths(DATASET_DIR)
CLASSIFICATION_DATASET_PATH = data_helpers.CLASSIFICATION_DATASET_PATH
TOOL_SEQUENCE_DATASET_PATH = data_helpers.TOOL_SEQUENCE_DATASET_PATH
CONTEXT_PRUNING_DATASET_PATH = data_helpers.CONTEXT_PRUNING_DATASET_PATH
# -------------------------------------------

OLLAMA_BASE_URL = st.secrets.get("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL", "mistral-small:24b-instruct-2501-q4_K_M")

# Optional third model (env-driven)
THIRD_KIND = st.secrets.get("THIRD_KIND", "OpenAI")
THIRD_MODEL = st.secrets.get("THIRD_MODEL", "gpt-4.1-mini" if THIRD_KIND == "OpenAI" else "mistralai/mistral-small-3.2-24b-instruct")
if "third_model_kind" not in st.session_state:
    st.session_state["third_model_kind"] = THIRD_KIND
if "third_model" not in st.session_state:
    st.session_state["third_model"] = THIRD_MODEL

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
# --- FIX: Use latest Gemini 2.5 series as default (OpenRouter format) ---
GEMINI_MODEL = st.secrets.get("GEMINI_MODEL", "google/gemini-2.5-flash")



OPENROUTER_MODEL_METADATA, OPENAI_MODEL_METADATA, GEMINI_MODEL_METADATA, AVAILABLE_MODELS = load_model_metadata()

# OpenRouter config
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", "mistralai/mistral-small-3.2-24b-instruct")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Default provider toggles (sidebar deprecated; per-test controls set overrides)
use_ollama = False
use_openai = True
use_ollama_local = False


# --- NEW: API Routing Configuration ---
# Set to "openrouter" to route all calls through OpenRouter (unified pricing, simpler)
# Set to "native" to use native APIs (OpenAI SDK, Google Genai SDK) for advanced features
API_ROUTING_MODE = st.secrets.get("API_ROUTING_MODE", "openrouter").lower()  # "openrouter" or "native"

# --- PATCH 22/23: OpenRouter Model Fetching with Metadata ---
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
# A default dictionary in case the API call fails or key is missing
DEFAULT_OPENROUTER_MODEL_METADATA = {
    "mistralai/mistral-small-3.2-24b-instruct": {'context': '131,072', 'input_cost': '$0.06', 'output_cost': '$0.18'},
    "deepseek/deepseek-v3.1-terminus": {'context': '128,000', 'input_cost': '$0.14', 'output_cost': '$0.28'},
}


# --- PATCH 25: Ollama Local Model Definition ---
OLLAMA_MODEL_METADATA = {
    "mistral:latest": {'context': '4,096', 'local_info': 'General Purpose'},
    "llama3:8b": {'context': '8,192', 'local_info': 'Next-Gen Llama'},
    "mistral-small:24b-instruct-2501-q4_K_M": {'context': '32,768', 'local_info': 'Mistral Small Quantized'},
    "Custom...": {'context': 'N/A', 'local_info': 'User Defined'}
}
# -----------------------------------------------

# --- Initialize cost tracker ---
if "cost_tracker" not in st.session_state:
    st.session_state.cost_tracker = CostTracker()
    register_all_extractors()

# --- PATCH 30: Dynamic OpenRouter Pricing Cache (TTL 30 days) ---
# Pricing functions moved to core/pricing.py

# Import pricing cache for backward compatibility
from core.pricing import fetch_openrouter_pricing

OPENROUTER_PRICING_CACHE = fetch_openrouter_pricing()
GEMINI_PRICING_CACHE = OPENROUTER_PRICING_CACHE  # Gemini pricing is in the same cache
OPENAI_PRICING_CACHE = OPENROUTER_PRICING_CACHE  # OpenAI pricing is in the same cache

# Skip to Pydantic models (pricing functions removed)
# All pricing functions (_load_pricing_from_disk, _save_pricing_to_disk, fetch_openrouter_pricing,
# _to_openrouter_model_id, _to_native_model_id, _get_provider_from_model_id,
# _fetch_models_from_openrouter, fetch_gemini_models_from_linkup, _get_default_gemini_models,
# _parse_gemini_models_from_linkup, custom_gemini_price_lookup, fetch_openai_models_from_linkup,
# _get_default_openai_models, _parse_openai_models_from_linkup, get_all_available_models,
# custom_openrouter_price_lookup, _normalize_ollama_root) moved to core/pricing.py and utils/model_discovery.py

# ---------- Pydantic structured outputs ----------
# Pydantic models moved to core/models.py

# ============================================================
# STREAMLIT UI CODE
# ============================================================

# ---------- UI ----------
st.set_page_config(page_title="Classification + Eval (OpenRouter Mistral + OpenAI)", layout="wide")
st.title("🧩 Enhanced Classification + Evaluation Suite")
st.caption(f"Dataset directory: `{DATASET_DIR}` — Datasets auto-generated and persisted.")
sidebar.configure(globals())
sidebar.render_api_sidebar()

# --- API Routing Configuration UI ---
# ---------- NEW: Dataset Generation and Persistence Helpers ----------





# ---------- Label Normalization Helpers ----------
_CANON_MAP = {
    # general
    "general": "general_answer", "general answer": "general_answer",
    "general_answer": "general_answer",

    # knowledge lookups
    "kb": "kb_lookup", "kb lookup": "kb_lookup", "kb_lookup": "kb_lookup",
    "knowledge": "kb_lookup", "knowledge_lookup": "kb_lookup",

    # tool calls
    "tool": "tool_call", "tool call": "tool_call", "tool_call": "tool_call", "toolcall": "tool_call",
}



# ---------- Data loading ----------

# --- NEW: Auto-generate datasets on first run if they don't exist ---


# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = _load_df_from_path()

# --- Initialize Universal Execution Tracker ---
if 'execution_tracker' not in st.session_state:
    st.session_state['execution_tracker'] = ExecutionTracker()

# --- PATCH 16: Initialize Pruning DF in session state ---
if "agent_df" not in st.session_state:
    st.session_state.agent_df = load_tool_sequence_dataset()
if "pruning_df" not in st.session_state:
    st.session_state.pruning_df = load_context_pruning_dataset()
# --------------------------------------------------------

# Check for missing datasets on startup
if "datasets_checked" not in st.session_state:
    check_and_generate_datasets()
    st.session_state.datasets_checked = True

# ---------- Helpers ----------


# ============================================================
# DASHBOARD VISUALIZATION HELPER FUNCTIONS
# ============================================================
# NOTE: Visualization functions moved to utils/visualizations.py and utils/gantt_charts.py
# Imported above as: render_test_flow_diagram, render_kpi_metrics, etc.

# render_kpi_metrics moved to utils/visualizations.py
# render_cost_dashboard moved to utils/visualizations.py

# visualize_dataset_composition moved to utils/visualizations.py























# ---------- Rigorous Reporting with Scikit-learn and LLM Explanation ----------

# --- PATCH 3: Gemini Code Execution Helper ---


# ============================================
# Agent Coordination Patterns
# ============================================



# ============================================
# Leaf Agent Scaffold Integration
# ============================================








# --- PATCH 4: Structured Summary Helper (Point 4) ---



# ------------------------------------------------------------



# ---------- Model Callers (OpenAI, Ollama, OpenRouter) ----------



# --- PATCH 28: Gemini Classification Function (with routing support) ---
# ---------------------------------------------------------------








# ---------- Core Runner (UPDATED FOR LATENCY) ----------

# ---------- NEW: Smarter Weighted Pick (Test 2) ----------

# ======================= Sidebar & Main Layout =======================
def _configure_modules():
    summaries.configure({
        "use_openai": use_openai,
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "OPENAI_MODEL": OPENAI_MODEL,
        "use_ollama": use_ollama,
        "OPENROUTER_MODEL": OPENROUTER_MODEL,
        "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
    })
    api_clients.configure({
        "API_ROUTING_MODE": API_ROUTING_MODE,
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "OPENAI_MODEL": OPENAI_MODEL,
        "OPENROUTER_MODEL": OPENROUTER_MODEL,
        "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
    })
    globals()['display_final_summary_for_test'] = summaries.display_final_summary_for_test
    globals()['get_structured_summary_and_refinement'] = summaries.get_structured_summary_and_refinement
    judges.configure({
        "API_ROUTING_MODE": API_ROUTING_MODE,
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "OPENAI_MODEL": OPENAI_MODEL,
        "OPENROUTER_MODEL": OPENROUTER_MODEL,
        "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
    })
    globals()['run_judge_flexible'] = judges.run_judge_flexible
    globals()['run_judge_openai'] = judges.run_judge_openai
    globals()['run_judge_ollama'] = judges.run_judge_ollama
    globals()['run_pruner'] = judges.run_pruner
    test_runners.configure({
        "use_openai": use_openai,
        "use_ollama": use_ollama,
        "use_ollama_local": use_ollama_local,
        "OPENROUTER_MODEL": OPENROUTER_MODEL,
        "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
        "OLLAMA_MODEL": OLLAMA_MODEL,
        "THIRD_KIND": THIRD_KIND,
        "THIRD_MODEL": THIRD_MODEL,
    })

_configure_modules()
del _configure_modules
# --- PATCH 5: Update Tab Definitions ---
tabs = st.tabs([
    "Preparation: Data Generation", # New Tab 0
    "Test 1: Classify, F1, Latency & Analysis",
    "Test 2: Advanced Ensembling",
    "Test 3: LLM as Judge",
    "Test 4: Quantitative Pruning",
    "Test 5: Agent Self-Refinement (Code Ex.)", # Tab 5
    "Test 6: Visual LLM Testing", # NEW Tab 6
    "Agent Dashboard" # Tab 7
])
# -------------------------------------

data_generation.configure({
    "dataset_paths": {
        "classification": CLASSIFICATION_DATASET_PATH,
        "tool_sequence": TOOL_SEQUENCE_DATASET_PATH,
        "context_pruning": CONTEXT_PRUNING_DATASET_PATH,
    },
    "dataset_prompts": DEFAULT_DATASET_PROMPTS,
    "available_models": AVAILABLE_MODELS,
    "api_routing_mode": API_ROUTING_MODE,
    "generate_synthetic_data": api_clients.generate_synthetic_data,
    "loaders": {
        "classification": load_classification_dataset,
        "tool_sequence": load_tool_sequence_dataset,
        "context_pruning": load_context_pruning_dataset,
    },
})
data_generation.render_preparation_tab(tabs[0])

test_tabs.configure(globals())
test_tabs.render_test1_tab(tabs[1])
test_tabs.render_test2_tab(tabs[2])
test_tabs.render_test3_tab(tabs[3])
test_tabs.render_test4_tab(tabs[4])
test_tabs.render_test5_tab(tabs[5])

# Test 6: Visual LLM Testing
from ui import test6_visual_llm
test6_visual_llm.configure(globals())
test6_visual_llm.render_test6_tab(tabs[6])

agent_dashboard.configure(globals())
agent_dashboard.render_agent_dashboard(tabs[7])
footer.render_footer()




