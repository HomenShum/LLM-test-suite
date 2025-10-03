"""Utilities for training and applying ensemble weights and calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from utils.data_helpers import _normalize_label

ARTIFACT_DIR = Path("artifacts")
DEFAULT_WEIGHT_PATH = ARTIFACT_DIR / "ensemble_weights.json"


@dataclass
class CalibrationModel:
    """Serializable representation of a 1D calibration model."""

    method: str
    x_thresholds: List[float] = field(default_factory=list)
    y_values: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "x_thresholds": self.x_thresholds,
            "y_values": self.y_values,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["CalibrationModel"]:
        if not data:
            return None
        return cls(
            method=data.get("method", "isotonic"),
            x_thresholds=[float(x) for x in data.get("x_thresholds", [])],
            y_values=[float(y) for y in data.get("y_values", [])],
        )

    def transform(self, value: float) -> float:
        if not self.x_thresholds or not self.y_values:
            return value
        return float(np.clip(np.interp(value, self.x_thresholds, self.y_values), 0.0, 1.0))


@dataclass
class WeightRecord:
    """Per-model ensemble metadata."""

    provider: str
    model: str
    prompt_revision: str
    class_weights: Dict[str, float]
    macro_f1: float
    calibration: Optional[CalibrationModel] = None
    trained_rows: int = 0
    trained_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_revision": self.prompt_revision,
            "class_weights": self.class_weights,
            "macro_f1": self.macro_f1,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "trained_rows": self.trained_rows,
            "trained_at": self.trained_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeightRecord":
        return cls(
            provider=data["provider"],
            model=data["model"],
            prompt_revision=data.get("prompt_revision", "unknown"),
            class_weights={k: float(v) for k, v in data.get("class_weights", {}).items()},
            macro_f1=float(data.get("macro_f1", 0.0)),
            calibration=CalibrationModel.from_dict(data.get("calibration")),
            trained_rows=int(data.get("trained_rows", 0)),
            trained_at=data.get("trained_at", datetime.utcnow().isoformat()),
        )


@dataclass
class ModelTrainingSpec:
    """Maps DataFrame columns to provider/model identifiers."""

    provider: str
    model_identifier: str
    prediction_column: str
    confidence_column: str
    probability_raw_column: str
    probability_calibrated_column: str


class EnsembleWeightStore:
    """Loads, stores, and applies ensemble weights and calibration metadata."""

    def __init__(self, records: List[WeightRecord], metadata: Dict[str, Any], path: Path):
        self._records = records
        self._metadata = metadata
        self._path = path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "EnsembleWeightStore":
        target = path or DEFAULT_WEIGHT_PATH
        if not target.exists():
            return cls(records=[], metadata={"created_at": None}, path=target)

        with target.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        records = [WeightRecord.from_dict(item) for item in payload.get("weights", [])]
        metadata = payload.get("metadata", {})
        return cls(records=records, metadata=metadata, path=target)

    def save(self) -> None:
        if not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {**self._metadata, "updated_at": datetime.utcnow().isoformat()},
            "weights": [record.to_dict() for record in self._records],
        }
        with self._path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def upsert_record(self, record: WeightRecord) -> None:
        for idx, existing in enumerate(self._records):
            if (
                existing.provider == record.provider
                and existing.model == record.model
                and existing.prompt_revision == record.prompt_revision
            ):
                self._records[idx] = record
                break
        else:
            self._records.append(record)

    def find_record(self, provider: str, model: str, prompt_revision: str) -> Optional[WeightRecord]:
        for record in self._records:
            if (
                record.provider == provider
                and record.model == model
                and record.prompt_revision == prompt_revision
            ):
                return record
        return None

    def get_class_weights(self, provider: str, model: str, prompt_revision: str) -> Optional[Dict[str, float]]:
        record = self.find_record(provider, model, prompt_revision)
        return record.class_weights if record else None

    def calibrate_distribution(
        self,
        provider: str,
        model: str,
        prompt_revision: str,
        probabilities: Dict[str, float],
        predicted_label: str,
    ) -> Dict[str, float]:
        record = self.find_record(provider, model, prompt_revision)
        if not record or not record.calibration or not predicted_label:
            return probabilities

        raw_pred_prob = float(probabilities.get(predicted_label, 0.0))
        calibrated_pred = record.calibration.transform(raw_pred_prob)
        calibrated_pred = float(np.clip(calibrated_pred, 0.0, 1.0))

        # Renormalize remaining probability mass
        other_labels = [label for label in probabilities.keys() if label != predicted_label]
        remainder = max(0.0, 1.0 - calibrated_pred)
        other_total = float(sum(max(probabilities.get(label, 0.0), 0.0) for label in other_labels))

        calibrated = {predicted_label: calibrated_pred}
        if other_labels:
            if other_total > 0:
                for label in other_labels:
                    proportion = max(probabilities.get(label, 0.0), 0.0) / other_total
                    calibrated[label] = remainder * proportion
            else:
                equal_share = remainder / len(other_labels)
                for label in other_labels:
                    calibrated[label] = equal_share

        # Include labels that were not explicitly present in the input map
        for label, value in probabilities.items():
            if label not in calibrated:
                calibrated[label] = float(value)

        # Final normalization guard
        total = sum(calibrated.values())
        if total > 0:
            for label in list(calibrated.keys()):
                calibrated[label] = float(calibrated[label] / total)

        return calibrated

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def path(self) -> Path:
        return self._path


def _parse_probability_blob(blob: Any) -> Dict[str, float]:
    if isinstance(blob, dict):
        return {str(k): float(v) for k, v in blob.items()}
    if not blob or (isinstance(blob, float) and np.isnan(blob)):
        return {}
    try:
        data = json.loads(blob)
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
    except (TypeError, ValueError):
        pass
    return {}


def _fit_isotonic(raw_probs: List[float], outcomes: List[float]) -> Optional[CalibrationModel]:
    if len(raw_probs) < 10:
        return None
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_probs, outcomes)
        return CalibrationModel(
            method="isotonic",
            x_thresholds=[float(x) for x in iso.X_thresholds_],
            y_values=[float(y) for y in iso.y_thresholds_],
        )
    except Exception:
        return None


def _aggregate_class_f1(reports: List[Dict[str, Any]]) -> Dict[str, float]:
    per_label: Dict[str, List[float]] = {}
    for report in reports:
        for label, metrics in report.items():
            if not isinstance(metrics, dict):
                continue
            if "f1-score" not in metrics:
                continue
            per_label.setdefault(label, []).append(float(metrics["f1-score"]))
    return {label: float(np.mean(values)) for label, values in per_label.items() if values}


def train_weights_from_dataframe(
    df: pd.DataFrame,
    specs: List[ModelTrainingSpec],
    prompt_revision: str,
    n_splits: int = 5,
    weight_store: Optional[EnsembleWeightStore] = None,
) -> EnsembleWeightStore:
    """Compute per-class F1 weights and calibration curves from a labelled DataFrame."""

    if weight_store is None:
        weight_store = EnsembleWeightStore.load()

    labels_series = df["classification"].fillna("").map(_normalize_label)
    valid_mask = labels_series.astype(bool)
    df_working = df.loc[valid_mask].copy()

    if df_working.empty:
        raise ValueError("Training DataFrame has no labelled rows for weight computation.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for spec in specs:
        fold_reports: List[Dict[str, Any]] = []
        raw_probs: List[float] = []
        outcomes: List[float] = []

        predictions = df_working[spec.prediction_column].fillna("").map(_normalize_label)
        confidences = df_working[spec.confidence_column]
        prob_series = df_working[spec.probability_raw_column]

        for train_idx, _ in skf.split(df_working, labels_series.loc[df_working.index]):
            train_subset = df_working.iloc[train_idx]
            y_true = train_subset["classification"].fillna("").map(_normalize_label)
            y_pred = train_subset[spec.prediction_column].fillna("").map(_normalize_label)
            valid_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t and p]
            if not valid_indices:
                continue
            y_true_f = [y_true.iloc[i] for i in valid_indices]
            y_pred_f = [y_pred.iloc[i] for i in valid_indices]
            report = classification_report(y_true_f, y_pred_f, output_dict=True, zero_division=0)
            fold_reports.append(report)

        for (_, row_pred, row_conf, row_prob, row_true) in zip(
            df_working.index,
            predictions,
            confidences,
            prob_series,
            labels_series.loc[df_working.index],
        ):
            if not row_pred or not row_true:
                continue
            probs = _parse_probability_blob(row_prob)
            raw_prob = probs.get(row_pred)
            if raw_prob is None:
                try:
                    raw_prob = float(row_conf)
                except (TypeError, ValueError):
                    continue
            raw_probs.append(float(raw_prob))
            outcomes.append(1.0 if row_pred == row_true else 0.0)

        if not fold_reports:
            continue

        class_weights = _aggregate_class_f1(fold_reports)
        macro_values = [report.get("macro avg", {}).get("f1-score", 0.0) for report in fold_reports]
        macro_f1 = float(np.mean(macro_values)) if macro_values else 0.0
        calibration = _fit_isotonic(raw_probs, outcomes)

        record = WeightRecord(
            provider=spec.provider,
            model=spec.model_identifier,
            prompt_revision=prompt_revision,
            class_weights=class_weights,
            macro_f1=macro_f1,
            calibration=calibration,
            trained_rows=int(df_working.shape[0]),
        )
        weight_store.upsert_record(record)

    weight_store.save()
    return weight_store


def apply_calibrated_probabilities(
    df: pd.DataFrame,
    specs: List[ModelTrainingSpec],
    prompt_revision: str,
    weight_store: Optional[EnsembleWeightStore] = None,
) -> None:
    """Populate calibrated probability columns on an existing DataFrame."""
    store = weight_store or EnsembleWeightStore.load()
    for spec in specs:
        if spec.probability_raw_column not in df.columns:
            continue
        if spec.probability_calibrated_column not in df.columns:
            df[spec.probability_calibrated_column] = None
        raw_series = df[spec.probability_raw_column]
        preds = df[spec.prediction_column].fillna("").map(_normalize_label)
        calibrated_values: List[str] = []
        for raw_blob, pred in zip(raw_series, preds):
            prob_map = _parse_probability_blob(raw_blob)
            if not prob_map and pred:
                prob_map = {pred: 1.0}
            calibrated = store.calibrate_distribution(
                spec.provider,
                spec.model_identifier,
                prompt_revision,
                prob_map,
                pred,
            )
            calibrated_values.append(json.dumps(calibrated))
        df.loc[raw_series.index, spec.probability_calibrated_column] = calibrated_values
