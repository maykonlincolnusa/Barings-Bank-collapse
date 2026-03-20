from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class SequenceAutoencoder:
    sequence_length: int = 8
    tensorflow_enabled: bool = False

    def __post_init__(self) -> None:
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95, svd_solver="full")
        self.model = None
        try:
            import tensorflow as tf  # type: ignore
            from tensorflow.keras import Sequential  # type: ignore
            from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed  # type: ignore

            self.tf = tf
            self.Sequential = Sequential
            self.LSTM = LSTM
            self.RepeatVector = RepeatVector
            self.TimeDistributed = TimeDistributed
            self.Dense = Dense
            self.tensorflow_enabled = True
        except Exception:
            self.tensorflow_enabled = False

    def fit(self, frame: pd.DataFrame, feature_cols: list[str], entity_col: str = "entity_id") -> None:
        sequences, _ = self._build_sequences(frame, feature_cols, entity_col)
        if len(sequences) == 0:
            self.reference_error_ = 0.0
            return
        if self.tensorflow_enabled:
            self._fit_tensorflow(sequences)
        else:
            flattened = sequences.reshape(sequences.shape[0], -1)
            scaled = self.scaler.fit_transform(flattened)
            reduced = self.pca.fit_transform(scaled)
            reconstructed = self.pca.inverse_transform(reduced)
            self.reference_error_ = float(np.mean((scaled - reconstructed) ** 2))

    def score(self, frame: pd.DataFrame, feature_cols: list[str], entity_col: str = "entity_id") -> np.ndarray:
        sequences, indexes = self._build_sequences(frame, feature_cols, entity_col)
        scores = np.zeros(len(frame), dtype=float)
        if len(sequences) == 0:
            return scores
        if self.tensorflow_enabled and self.model is not None:
            reconstructed = self.model.predict(sequences, verbose=0)
            errors = np.mean((sequences - reconstructed) ** 2, axis=(1, 2))
        else:
            flattened = sequences.reshape(sequences.shape[0], -1)
            scaled = self.scaler.transform(flattened)
            reduced = self.pca.transform(scaled)
            reconstructed = self.pca.inverse_transform(reduced)
            errors = np.mean((scaled - reconstructed) ** 2, axis=1)
        for row_index, error in zip(indexes, errors, strict=False):
            scores[row_index] = float(error)
        return scores

    def _build_sequences(self, frame: pd.DataFrame, feature_cols: list[str], entity_col: str) -> tuple[np.ndarray, list[int]]:
        ordered = frame.sort_values([entity_col, "date"]).reset_index()
        sequences = []
        indexes: list[int] = []
        for _, group in ordered.groupby(entity_col, sort=False):
            values = group[feature_cols].astype(float).to_numpy()
            group_indexes = group["index"].to_list()
            if len(values) < self.sequence_length:
                continue
            for end in range(self.sequence_length - 1, len(values)):
                start = end - self.sequence_length + 1
                sequences.append(values[start : end + 1])
                indexes.append(group_indexes[end])
        if not sequences:
            return np.empty((0, self.sequence_length, len(feature_cols))), []
        return np.asarray(sequences, dtype=float), indexes

    def _fit_tensorflow(self, sequences: np.ndarray) -> None:
        tf = self.tf
        tf.random.set_seed(42)
        n_features = sequences.shape[2]
        model = self.Sequential(
            [
                self.LSTM(32, activation="tanh", input_shape=(self.sequence_length, n_features), return_sequences=False),
                self.RepeatVector(self.sequence_length),
                self.LSTM(32, activation="tanh", return_sequences=True),
                self.TimeDistributed(self.Dense(n_features)),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(sequences, sequences, epochs=8, batch_size=min(32, len(sequences)), verbose=0)
        reconstructed = model.predict(sequences, verbose=0)
        self.reference_error_ = float(np.mean((sequences - reconstructed) ** 2))
        self.model = model

