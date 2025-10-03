"""Golden dataset training pipeline for ensemble weighting and calibration."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from core.ensemble_weights import (
    EnsembleWeightStore,
    ModelTrainingSpec,
    apply_calibrated_probabilities,
    train_weights_from_dataframe,
)
from core.prompt_versions import CLASSIFICATION_PROMPT_REVISION


def _build_training_specs(
    openrouter_model: str,
    openai_model: str,
    third_model: Optional[str],
    third_provider: str,
) -> List[ModelTrainingSpec]:
    specs = [
        ModelTrainingSpec(
            provider="OpenRouter",
            model_identifier=openrouter_model,
            prediction_column="classification_result_openrouter_mistral",
            confidence_column="classification_result_openrouter_mistral_confidence",
            probability_raw_column="probabilities_openrouter_mistral_raw",
            probability_calibrated_column="probabilities_openrouter_mistral_calibrated",
        ),
        ModelTrainingSpec(
            provider="OpenAI",
            model_identifier=openai_model,
            prediction_column="classification_result_openai",
            confidence_column="classification_result_openai_confidence",
            probability_raw_column="probabilities_openai_raw",
            probability_calibrated_column="probabilities_openai_calibrated",
        ),
    ]

    if third_model and third_provider in {"OpenAI", "OpenRouter"}:
        specs.append(
            ModelTrainingSpec(
                provider=third_provider,
                model_identifier=third_model,
                prediction_column="classification_result_third",
                confidence_column="classification_result_third_confidence",
                probability_raw_column="probabilities_third_raw",
                probability_calibrated_column="probabilities_third_calibrated",
            )
        )

    return specs


def train_golden_ensemble(
    df: pd.DataFrame,
    openrouter_model: str,
    openai_model: str,
    third_model: Optional[str],
    third_provider: str,
    prompt_revision: str = CLASSIFICATION_PROMPT_REVISION,
    n_splits: int = 5,
    weight_store: Optional[EnsembleWeightStore] = None,
) -> EnsembleWeightStore:
    """Train/update ensemble weights from a golden classification dataset."""

    specs = _build_training_specs(openrouter_model, openai_model, third_model, third_provider)
    store = train_weights_from_dataframe(
        df=df,
        specs=specs,
        prompt_revision=prompt_revision,
        n_splits=n_splits,
        weight_store=weight_store,
    )

    apply_calibrated_probabilities(df, specs, prompt_revision, store)
    return store


def train_golden_from_csv(
    dataset_path: Path,
    openrouter_model: str,
    openai_model: str,
    third_model: Optional[str],
    third_provider: str,
    prompt_revision: str = CLASSIFICATION_PROMPT_REVISION,
    n_splits: int = 5,
) -> EnsembleWeightStore:
    """Convenience wrapper that loads a CSV file and trains ensemble weights."""

    df = pd.read_csv(dataset_path)
    store = train_golden_ensemble(
        df=df,
        openrouter_model=openrouter_model,
        openai_model=openai_model,
        third_model=third_model,
        third_provider=third_provider,
        prompt_revision=prompt_revision,
        n_splits=n_splits,
    )
    df.to_csv(dataset_path, index=False)
    return store
