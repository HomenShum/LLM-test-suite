"""
Data loading, saving, and normalization helpers.
Extracted from streamlit_test_v5.py to reduce main file size.

This module contains:
- Dataset loading functions
- Dataset saving functions
- Label normalization
- Data generation helpers
"""

import os
import re
import json
import time
import pandas as pd
import streamlit as st
from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel, Field

# Import from config
from config.scenarios import (
    SKELETON_COLUMNS,
    DEFAULT_DATASET_PROMPTS
)


# Dataset paths
DATASET_DIR = "test_dataset"
CLASSIFICATION_DATASET_PATH = os.path.join(DATASET_DIR, "classification_dataset.csv")
TOOL_SEQUENCE_DATASET_PATH = os.path.join(DATASET_DIR, "tool_sequence_dataset.csv")
CONTEXT_PRUNING_DATASET_PATH = os.path.join(DATASET_DIR, "context_pruning_dataset.csv")


def configure_dataset_paths(base_dir: str):
    """Configure dataset directory and related file paths at runtime."""
    global DATASET_DIR, CLASSIFICATION_DATASET_PATH, TOOL_SEQUENCE_DATASET_PATH, CONTEXT_PRUNING_DATASET_PATH
    DATASET_DIR = base_dir
    CLASSIFICATION_DATASET_PATH = os.path.join(DATASET_DIR, "classification_dataset.csv")
    TOOL_SEQUENCE_DATASET_PATH = os.path.join(DATASET_DIR, "tool_sequence_dataset.csv")
    CONTEXT_PRUNING_DATASET_PATH = os.path.join(DATASET_DIR, "context_pruning_dataset.csv")



# Pydantic models for data validation
class PruningDataItem(BaseModel):
    instruction: str = Field(description="System instruction/persona given to the LLM.")
    new_question: str = Field(description="The user's new question.")
    conversation_history: str = Field(description="Previous conversation context (JSON or text).")
    knowledge_base: str = Field(description="Available knowledge base (JSON or text).")
    expected_action: str = Field(description="Expected action: general_answer, kb_lookup, or tool_call.")
    expected_kept_keys: Optional[str] = Field(default=None, description="Comma-separated keys to keep if pruning.")
    expected_rationale: Optional[str] = Field(default=None, description="Explanation for the expected action.")


# Label normalization map
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


def _normalize_label(s: Optional[str]) -> str:
    """Normalize classification labels to canonical form."""
    if not s:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("-", " ").replace("_", " ")
    canon = _CANON_MAP.get(s)
    if canon:
        return canon
    return s.replace(" ", "_")


def _allowed_labels(df: pd.DataFrame) -> List[str]:
    """Extract unique labels from classification column."""
    try:
        if "classification" in df.columns:
            return sorted(df["classification"].dropna().unique().tolist())
        return []
    except Exception:
        return []


def _subset_for_run(df: pd.DataFrame, n: Optional[int]) -> pd.DataFrame:
    """Return first n rows of dataframe, or all if n is None."""
    try:
        if n is None or not len(df) or (isinstance(n, int) and n >= len(df)):
            return df
        if isinstance(n, int) and n > 0:
            return df.head(n)
        return df
    except Exception:
        return df


def _style_selected_rows(df: pd.DataFrame, n: Optional[int]):
    """Style dataframe to highlight selected rows."""
    try:
        # Truncate long rationale-like columns for display safety
        display_df = df.copy()
        rationale_cols = [c for c in display_df.columns if c.endswith("_rationale") or c == "judge_rationale"]
        for col in rationale_cols:
            try:
                display_df[col] = display_df[col].astype(str).str.slice(0, 150) + "..."
            except Exception:
                pass

        if n is None:
            selected = set(display_df.index)
        else:
            selected = set(display_df.index[: max(0, int(n))])
        
        def _hl(row):
            return ["background-color: #1E4594"] * len(row) if row.name in selected else [""] * len(row)
        
        return display_df.style.apply(lambda r: _hl(r), axis=1)
    except Exception:
        return df


def ensure_dataset_directory():
    """Ensure the test_dataset directory exists."""
    os.makedirs(DATASET_DIR, exist_ok=True)


def save_dataset_to_file(df: pd.DataFrame, dataset_type: str, model_used: Optional[str] = None, routing_mode: Optional[str] = None):
    """Save a dataset to the appropriate file based on type and write a .meta.json with model/routing."""
    ensure_dataset_directory()

    if dataset_type == "Classification":
        path = CLASSIFICATION_DATASET_PATH
    elif dataset_type == "Tool/Agent Sequence":
        path = TOOL_SEQUENCE_DATASET_PATH
    elif dataset_type == "Context Pruning":
        path = CONTEXT_PRUNING_DATASET_PATH
    else:
        st.warning(f"Unknown dataset type: {dataset_type}")
        return

    try:
        df.to_csv(path, index=False)
        # Write sidecar metadata for traceability
        meta_path = path.replace(".csv", ".meta.json")
        meta = {
            "model": model_used,
            "routing_mode": routing_mode,
            "when": pd.Timestamp.utcnow().isoformat(),
            "dataset_type": dataset_type,
        }
        # Attach cost tracker snapshot if available
        try:
            ct = st.session_state.cost_tracker
            meta["cost_tracker_totals"] = dict(ct.totals)
            meta["cost_tracker_calls"] = len(ct.by_call)
        except Exception:
            pass
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

        if model_used:
            st.success(f"✅ Saved {dataset_type} dataset to `{path}` using {model_used} [{routing_mode}]")
        else:
            st.success(f"✅ Saved {dataset_type} dataset to `{path}`")
    except Exception as e:
        st.error(f"Failed to save {dataset_type} dataset: {e}")


def save_results_df(df: pd.DataFrame, test_name: str, row_limit: Optional[int], is_pruning_test: bool = False):
    """Saves the results DataFrame to the test_output directory."""
    try:
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        if is_pruning_test:
            # For Test 4, the context is the length of the testset df
            limit_str = f"{len(df)}_rows"
            test_slug = "test_4_pruning"
        else:
            limit_str = f"{row_limit}_rows" if row_limit is not None else "all_rows"
            test_slug = test_name.lower().replace(" ", "_").replace(":", "").replace(",", "")

        filename = f"{output_dir}/{test_slug}_{limit_str}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        st.success(f"Results saved to `{filename}`")
    except Exception as e:
        st.warning(f"Failed to save results: {e}")


def load_classification_dataset() -> pd.DataFrame:
    """Load classification dataset from file, or return empty DataFrame."""
    try:
        if os.path.exists(CLASSIFICATION_DATASET_PATH):
            df = pd.read_csv(CLASSIFICATION_DATASET_PATH)
            cols_map = {c.lower(): c for c in df.columns}
            q, c = cols_map.get("query"), cols_map.get("classification")
            if q and c:
                df2 = df[[q, c]].rename(columns={q: "query", c: "classification"})
            else:
                df2 = pd.DataFrame(columns=["query", "classification"])

            # Add skeleton columns
            for col in SKELETON_COLUMNS:
                if col not in df2.columns:
                    df2[col] = None

            # Backward compatibility
            backcompat_map = {
                "classification_result_ollama": "classification_result_openrouter_mistral",
                "classification_result_ollama_rationale": "classification_result_openrouter_mistral_rationale"
            }
            for old_col, new_col in backcompat_map.items():
                if old_col in df.columns and new_col in df2.columns:
                    try:
                        df2[new_col] = df[old_col]
                    except Exception:
                        pass

            return df2[SKELETON_COLUMNS]
        else:
            return pd.DataFrame(columns=SKELETON_COLUMNS)
    except Exception as e:
        st.warning(f"Could not load classification dataset: {e}")
        return pd.DataFrame(columns=SKELETON_COLUMNS)


def load_tool_sequence_dataset() -> pd.DataFrame:
    """Load tool/agent sequence dataset from file."""
    try:
        if os.path.exists(TOOL_SEQUENCE_DATASET_PATH):
            return pd.read_csv(TOOL_SEQUENCE_DATASET_PATH)
        else:
            return pd.DataFrame(columns=["query", "expected_sequence"])
    except Exception as e:
        st.warning(f"Could not load tool sequence dataset: {e}")
        return pd.DataFrame(columns=["query", "expected_sequence"])


def load_context_pruning_dataset() -> pd.DataFrame:
    """Load context pruning dataset from file."""
    try:
        if os.path.exists(CONTEXT_PRUNING_DATASET_PATH):
            df = pd.read_csv(CONTEXT_PRUNING_DATASET_PATH)
            # Optional hard fail on legacy labels
            VALID_ACTIONS = {"general_answer", "kb_lookup", "tool_call"}
            if "expected_action" in df.columns:
                invalid = set(df["expected_action"].dropna().map(_normalize_label)) - VALID_ACTIONS
                if invalid:
                    raise ValueError(f"Invalid action labels found: {sorted(invalid)}")
            return df
        else:
            required_cols = list(PruningDataItem.model_fields.keys())
            return pd.DataFrame(columns=required_cols)
    except Exception as e:
        st.warning(f"Could not load context pruning dataset: {e}")
        required_cols = list(PruningDataItem.model_fields.keys())
        return pd.DataFrame(columns=required_cols)


def _load_df_from_path() -> pd.DataFrame:
    """Load classification dataset (backward compatibility wrapper)."""
    return load_classification_dataset()


async def auto_generate_default_datasets():
    """Generate default datasets if they don't exist in the test_dataset directory."""
    # Import here to avoid circular dependency
    from core.api_clients import generate_synthetic_data
    
    ensure_dataset_directory()

    datasets_to_generate = []

    # Check which datasets are missing
    if not os.path.exists(CLASSIFICATION_DATASET_PATH):
        datasets_to_generate.append(("Classification", CLASSIFICATION_DATASET_PATH, 100))
    if not os.path.exists(TOOL_SEQUENCE_DATASET_PATH):
        datasets_to_generate.append(("Tool/Agent Sequence", TOOL_SEQUENCE_DATASET_PATH, 50))
    if not os.path.exists(CONTEXT_PRUNING_DATASET_PATH):
        datasets_to_generate.append(("Context Pruning", CONTEXT_PRUNING_DATASET_PATH, 50))

    if not datasets_to_generate:
        return  # All datasets exist

    st.info(f"🔄 Generating {len(datasets_to_generate)} missing dataset(s)... This may take a moment.")

    # Use a simple, fast model for default generation
    default_gen_model = "openai/gpt-5-mini"

    for data_type, path, size in datasets_to_generate:
        try:
            with st.spinner(f"Generating {data_type} dataset ({size} items)..."):
                prompt = DEFAULT_DATASET_PROMPTS[data_type]
                df = await generate_synthetic_data(prompt, size, data_type, default_gen_model)

                if not df.empty:
                    df.to_csv(path, index=False)
                    st.success(f"✅ Generated and saved {data_type} dataset to `{path}`")
                else:
                    st.warning(f"⚠️ Failed to generate {data_type} dataset")
        except Exception as e:
            st.error(f"Error generating {data_type} dataset: {e}")


def check_and_generate_datasets():
    """Check if datasets exist, and generate them if needed (synchronous wrapper)."""
    ensure_dataset_directory()

    # Check if any datasets are missing
    missing = []
    if not os.path.exists(CLASSIFICATION_DATASET_PATH):
        missing.append("Classification")
    if not os.path.exists(TOOL_SEQUENCE_DATASET_PATH):
        missing.append("Tool/Agent Sequence")
    if not os.path.exists(CONTEXT_PRUNING_DATASET_PATH):
        missing.append("Context Pruning")

    if missing:
        import asyncio
        asyncio.run(auto_generate_default_datasets())

