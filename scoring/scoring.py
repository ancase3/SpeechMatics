#!/usr/bin/env python3
"""Official Codabench scorer for the SpeechMatics shared task.

Expected CSV format for both gold and participant files:

    id,label_t1,label_t2

Allowed labels are:
    - ironia
    - no_ironia
    - empty string, used when a task is not submitted for a given item

The script writes /app/output/scores.json in the format expected by Codabench.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


# Codabench standard paths. They can be overridden for local testing.
INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/app/input"))
REF_DIR = Path(os.environ.get("REF_DIR", INPUT_DIR / "ref"))
RES_DIR = Path(os.environ.get("RES_DIR", INPUT_DIR / "res"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/app/output"))

EXPECTED_COLUMNS = ("id", "label_t1", "label_t2")
LABEL_MAP = {
    "ironia": 1,
    "no_ironia": 0,
    "": None,
}


@dataclass(frozen=True)
class Record:
    """Gold and predicted labels for one evaluated item."""

    item_id: str
    t1_true: str
    t1_pred: str
    t2_true: str
    t2_pred: str


def find_single_csv(directory: Path) -> Path:
    """Return the only CSV file contained in a Codabench input directory."""
    files = sorted(directory.glob("*.csv"))
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one CSV file in {directory}, found: {[f.name for f in files]}")
    return files[0]


def normalize_label(value: object) -> str:
    """Normalize CSV labels while preserving empty task submissions."""
    if value is None:
        return ""
    return str(value).strip()


def read_prediction_csv(path: Path) -> List[Dict[str, str]]:
    """Read a SpeechMatics CSV file with columns id,label_t1,label_t2."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {path} is empty or has no header")

        fieldnames = tuple(name.strip() for name in reader.fieldnames)
        if fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"Invalid CSV header in {path}. Expected {EXPECTED_COLUMNS}, found {fieldnames}"
            )

        rows: List[Dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            item_id = normalize_label(row.get("id"))
            label_t1 = normalize_label(row.get("label_t1"))
            label_t2 = normalize_label(row.get("label_t2"))

            if not item_id:
                raise ValueError(f"Missing id at row {row_number} in {path}")

            rows.append({"id": item_id, "label_t1": label_t1, "label_t2": label_t2})

    return rows


def validate_label(label: str, task_name: str, item_id: str, file_role: str) -> None:
    """Validate a single label value."""
    if label not in LABEL_MAP:
        allowed = "ironia, no_ironia or empty string"
        raise ValueError(
            f"Invalid {file_role} label for {task_name} at id={item_id!r}: {label!r}. "
            f"Allowed values are: {allowed}."
        )


def build_records(gold_rows: Sequence[Mapping[str, str]], pred_rows: Sequence[Mapping[str, str]]) -> List[Record]:
    """Align gold and prediction rows by id and validate their labels."""
    pred_by_id: Dict[str, Mapping[str, str]] = {}
    for row in pred_rows:
        item_id = row["id"]
        if item_id in pred_by_id:
            raise ValueError(f"Duplicate id in prediction file: {item_id}")
        validate_label(row["label_t1"], "Task 1", item_id, "prediction")
        validate_label(row["label_t2"], "Task 2", item_id, "prediction")
        pred_by_id[item_id] = row

    gold_ids = [row["id"] for row in gold_rows]
    gold_id_set = set(gold_ids)

    missing = [item_id for item_id in gold_ids if item_id not in pred_by_id]
    if missing:
        raise ValueError(f"Missing ids in prediction file, showing up to 10: {missing[:10]}")

    extra = sorted(set(pred_by_id) - gold_id_set)
    if extra:
        raise ValueError(f"Extra ids in prediction file, showing up to 10: {extra[:10]}")

    records: List[Record] = []
    for gold_row in gold_rows:
        item_id = gold_row["id"]
        pred_row = pred_by_id[item_id]

        validate_label(gold_row["label_t1"], "Task 1", item_id, "gold")
        validate_label(gold_row["label_t2"], "Task 2", item_id, "gold")

        records.append(
            Record(
                item_id=item_id,
                t1_true=gold_row["label_t1"],
                t1_pred=pred_row["label_t1"],
                t2_true=gold_row["label_t2"],
                t2_pred=pred_row["label_t2"],
            )
        )

    return records


def compute_task_scores(y_true: Sequence[int], y_pred: Sequence[int], prefix: str) -> Dict[str, float]:
    """Compute macro Precision, Recall and F1 for a task."""
    if len(y_true) == 0:
        return {
            f"{prefix}_precision": 0.0,
            f"{prefix}_recall": 0.0,
            f"{prefix}_f1": 0.0,
        }

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        f"{prefix}_precision": float(precision),
        f"{prefix}_recall": float(recall),
        f"{prefix}_f1": float(f1),
    }


def compute_scores(records: Sequence[Record]) -> Dict[str, float]:
    """Compute task-level and global scores from aligned records."""
    y_true_t1: List[int] = []
    y_pred_t1: List[int] = []
    y_true_t2: List[int] = []
    y_pred_t2: List[int] = []

    for record in records:
        if record.t1_true != "" and record.t1_pred != "":
            y_true_t1.append(int(LABEL_MAP[record.t1_true]))
            y_pred_t1.append(int(LABEL_MAP[record.t1_pred]))

        if record.t2_true != "" and record.t2_pred != "":
            y_true_t2.append(int(LABEL_MAP[record.t2_true]))
            y_pred_t2.append(int(LABEL_MAP[record.t2_pred]))

    scores: Dict[str, float] = {}
    scores.update(compute_task_scores(y_true_t1, y_pred_t1, "task1"))
    scores.update(compute_task_scores(y_true_t2, y_pred_t2, "task2"))

    submitted_task_f1 = [
        scores["task1_f1"] if scores["task1_f1"] != 0.0 else None,
        scores["task2_f1"] if scores["task2_f1"] != 0.0 else None,
    ]
    valid_f1 = [score for score in submitted_task_f1 if score is not None]
    scores["macro_f1"] = float(np.mean(valid_f1)) if valid_f1 else 0.0

    return scores


class BootstrapAggregator:
    """Percentile bootstrap over item-level records."""

    def __init__(self, n_samples: int = 1000, ci: float = 0.95, seed: int = 123) -> None:
        self.n_samples = int(n_samples)
        self.ci = float(ci)
        self.rng = np.random.default_rng(seed)

        alpha = (1.0 - self.ci) / 2.0
        self.quantiles = {
            "low": 100.0 * alpha,
            "mid": 50.0,
            "high": 100.0 * (1.0 - alpha),
        }

    def aggregate(
        self,
        records: Sequence[Record],
        score_fn: Callable[[Sequence[Record]], Mapping[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        if len(records) == 0:
            raise ValueError("Cannot bootstrap an empty record set")

        point_scores = score_fn(records)
        distributions: Dict[str, List[float]] = {metric: [] for metric in point_scores}

        n_records = len(records)
        for _ in range(self.n_samples):
            sample_indices = self.rng.integers(0, n_records, size=n_records)
            sample = [records[i] for i in sample_indices]
            sample_scores = score_fn(sample)
            for metric, value in sample_scores.items():
                distributions[metric].append(float(value))

        ci_scores: Dict[str, Dict[str, float]] = {}
        for metric, values in distributions.items():
            arr = np.asarray(values, dtype=float)
            ci_scores[metric] = {
                name: float(np.percentile(arr, percentile))
                for name, percentile in self.quantiles.items()
            }

        return ci_scores


def flatten_bootstrap_scores(scores: Dict[str, float], ci_scores: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    """Merge point scores and bootstrap interval scores into Codabench's flat JSON format."""
    output = dict(scores)
    for metric, interval in ci_scores.items():
        output[f"{metric}_low"] = float(interval["low"])
        output[f"{metric}_mid"] = float(interval["mid"])
        output[f"{metric}_high"] = float(interval["high"])
    return output


def main() -> None:
    gold_file = find_single_csv(REF_DIR)
    prediction_file = find_single_csv(RES_DIR)

    gold_rows = read_prediction_csv(gold_file)
    prediction_rows = read_prediction_csv(prediction_file)
    records = build_records(gold_rows, prediction_rows)

    point_scores = compute_scores(records)
    bootstrap = BootstrapAggregator(n_samples=1000, ci=0.95, seed=123)
    ci_scores = bootstrap.aggregate(records, compute_scores)

    scores = flatten_bootstrap_scores(point_scores, ci_scores)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "scores.json").open("w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
