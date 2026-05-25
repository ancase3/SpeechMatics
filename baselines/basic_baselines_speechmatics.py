#!/usr/bin/env python3
"""Basic baseline and submission generator for SpeechMatics.

This script trains simple machine-learning baselines and creates a valid
SpeechMatics submission file with the columns:

    id,label_t1,label_t2

The competition data are not distributed with this repository. Download the
official data from Codabench and provide the paths through the command line.

Example:
    python baselines/basic_baseline_speechmatics.py \
        --train-csv train/corpus_ironia_iberlef2026_train.csv \
        --test-csv test/corpus_ironia_iberlef2026_test.csv \
        --train-audio-dir train/audios/audios_flac \
        --test-audio-dir test/audios/audios_flac \
        --output submission_file.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import librosa
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from tqdm import tqdm


REQUIRED_COLUMNS = {"id", "transcripcion"}
LABEL_COLUMN = "label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SpeechMatics basic baseline and create a submission CSV.")
    parser.add_argument("--train-csv", required=True, type=Path, help="Path to the official training CSV file.")
    parser.add_argument("--test-csv", required=True, type=Path, help="Path to the official test CSV file.")
    parser.add_argument("--train-audio-dir", required=True, type=Path, help="Directory containing training .flac files.")
    parser.add_argument("--test-audio-dir", required=True, type=Path, help="Directory containing test .flac files.")
    parser.add_argument("--output", default=Path("submission_file.csv"), type=Path, help="Output submission CSV path.")
    parser.add_argument("--max-features", default=10_000, type=int, help="Maximum number of TF-IDF features.")
    parser.add_argument("--random-state", default=42, type=int, help="Random seed for reproducibility.")
    parser.add_argument("--skip-grid-search", action="store_true", help="Use a fixed SVC configuration for Task 2.")
    return parser.parse_args()


def validate_dataframe(df: pd.DataFrame, *, require_labels: bool, name: str) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if require_labels and LABEL_COLUMN not in df.columns:
        missing.add(LABEL_COLUMN)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def extract_mfcc_features(audio_path: Path) -> np.ndarray:
    data, sample_rate = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate)
    return np.mean(mfcc.T, axis=0)


def get_audio_features(df: pd.DataFrame, audio_dir: Path, *, split_name: str) -> np.ndarray:
    features: List[np.ndarray] = []
    for item_id in tqdm(df["id"], desc=f"Extracting {split_name} MFCC features"):
        audio_path = audio_dir / f"{item_id}.flac"
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        features.append(extract_mfcc_features(audio_path))
    return np.vstack(features)


def train_text_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    max_features: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train Task 1 text-only baseline and return predictions plus text features."""
    vectorizer = TfidfVectorizer(
        analyzer="word",
        max_features=max_features,
        lowercase=False,
    )

    text_x_train_sparse = vectorizer.fit_transform(train_df["transcripcion"].fillna(""))
    text_x_test_sparse = vectorizer.transform(test_df["transcripcion"].fillna(""))

    scaler = MinMaxScaler()
    text_x_train = scaler.fit_transform(text_x_train_sparse.toarray())
    text_x_test = scaler.transform(text_x_test_sparse.toarray())

    classifier = LinearSVC(dual="auto")
    classifier.fit(text_x_train, train_df[LABEL_COLUMN])
    task1_predictions = classifier.predict(text_x_test)

    return task1_predictions, text_x_train, text_x_test


def train_multimodal_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_x_train: np.ndarray,
    text_x_test: np.ndarray,
    train_audio_dir: Path,
    test_audio_dir: Path,
    *,
    skip_grid_search: bool,
) -> np.ndarray:
    """Train Task 2 multimodal baseline using text TF-IDF and audio MFCC features."""
    labels = sorted(train_df[LABEL_COLUMN].unique().tolist(), reverse=True)
    id_to_label: Dict[int, str] = {idx: label for idx, label in enumerate(labels)}
    label_to_id: Dict[str, int] = {label: idx for idx, label in id_to_label.items()}

    y_train = train_df[LABEL_COLUMN].map(label_to_id).to_numpy()

    mfcc_x_train = get_audio_features(train_df, train_audio_dir, split_name="training")
    mfcc_x_test = get_audio_features(test_df, test_audio_dir, split_name="test")

    scaler = MinMaxScaler()
    mfcc_x_train = scaler.fit_transform(mfcc_x_train)
    mfcc_x_test = scaler.transform(mfcc_x_test)

    x_train = np.concatenate((text_x_train, mfcc_x_train), axis=1)
    x_test = np.concatenate((text_x_test, mfcc_x_test), axis=1)

    if skip_grid_search:
        classifier = SVC(class_weight="balanced", C=1.0, gamma="scale", kernel="rbf")
        classifier.fit(x_train, y_train)
    else:
        param_grid = {
            "C": [0.01, 0.1, 1, 10],
            "gamma": [0.1, 0.01, 0.001],
            "kernel": ["rbf"],
        }
        classifier = GridSearchCV(
            SVC(class_weight="balanced"),
            param_grid,
            refit=True,
            verbose=1,
            n_jobs=-1,
        )
        classifier.fit(x_train, y_train)
        print(f"Best Task 2 parameters: {classifier.best_params_}")

    task2_label_ids = classifier.predict(x_test)
    return np.asarray([id_to_label[int(label_id)] for label_id in task2_label_ids])


def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    validate_dataframe(train_df, require_labels=True, name="Training CSV")
    validate_dataframe(test_df, require_labels=False, name="Test CSV")

    task1_predictions, text_x_train, text_x_test = train_text_baseline(
        train_df,
        test_df,
        max_features=args.max_features,
    )

    task2_predictions = train_multimodal_baseline(
        train_df,
        test_df,
        text_x_train,
        text_x_test,
        args.train_audio_dir,
        args.test_audio_dir,
        skip_grid_search=args.skip_grid_search,
    )

    output_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "label_t1": task1_predictions,
            "label_t2": task2_predictions,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"Submission file written to: {args.output}")


if __name__ == "__main__":
    main()
