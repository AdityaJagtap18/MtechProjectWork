"""Train-only preprocessing (plan §15/§16/§25/§26).

`FeaturePreprocessor.fit` is given, for every node type, the *subset of
rows* that are legitimately "train" for fitting purposes, and never looks
at anything outside that subset:

- Supplier frames are indexed by (supplier_id, time); the fit subset is
  exactly the (supplier_id, time) pairs that are train-split prediction
  examples.
- Material/plant/product frames are also indexed by (id, time), but these
  node types have no prediction examples of their own -- the fit subset is
  every row whose `time` is a train-split prediction time (plan §14: their
  *dynamic* features are time-varying and must respect the same temporal
  boundary as the supplier target, even though nothing about them is
  itself being classified).
- Region/procurement frames have no time axis at all: every field is a
  generation-time constant (feature_audit.csv's blanket
  `graph/nodes.csv,*,...,True` rule) that is identical in every split, so
  the fit subset is simply every row.

Missing values (plan §16/§26): a numeric column's NaNs get an explicit
`{col}_missing` indicator plus training-median imputation -- never a
random value. Categorical columns are one-hot encoded against a vocabulary
fixed from the fit subset, with any category not seen there (or
NaN/`UNKNOWN`) mapped to an explicit `__UNKNOWN__` bucket (plan §24/§26).

Artifacts are plain JSON (not pickle) so they're inspectable and don't tie
the saved experiment to a specific library version for basic auditing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..schema.nodes import NodeType

UNKNOWN_CATEGORY = "__UNKNOWN__"


@dataclass
class _NumericStat:
    median: float
    mean: float
    std: float


@dataclass
class _TypePreprocessor:
    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_stats: dict[str, _NumericStat] = field(default_factory=dict)
    categorical_vocab: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, fit_mask: pd.Series) -> "_TypePreprocessor":
        fit_df = df.loc[fit_mask]
        for col in self.numeric_columns:
            values = pd.to_numeric(fit_df[col], errors="coerce")
            non_null = values.dropna()
            median = float(non_null.median()) if len(non_null) else 0.0
            imputed = values.fillna(median)
            mean = float(imputed.mean())
            std = float(imputed.std(ddof=0))
            if std < 1e-8:
                std = 1.0
            self.numeric_stats[col] = _NumericStat(median=median, mean=mean, std=std)
        for col in self.categorical_columns:
            categories = sorted(x for x in fit_df[col].dropna().unique() if x != UNKNOWN_CATEGORY)
            self.categorical_vocab[col] = categories
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.numeric_columns:
            stat = self.numeric_stats[col]
            values = pd.to_numeric(df[col], errors="coerce")
            missing = values.isna()
            imputed = values.fillna(stat.median)
            out[col] = (imputed - stat.mean) / stat.std
            out[f"{col}_missing"] = missing.astype(float)
        for col in self.categorical_columns:
            vocab = self.categorical_vocab[col]
            values = df[col].where(df[col].isin(vocab), UNKNOWN_CATEGORY)
            for category in vocab:
                out[f"{col}__{category}"] = (values == category).astype(float)
            out[f"{col}__{UNKNOWN_CATEGORY}"] = (values == UNKNOWN_CATEGORY).astype(float)
        return out

    def output_dim(self) -> int:
        dim = len(self.numeric_columns) * 2  # value + missing indicator
        for col in self.categorical_columns:
            dim += len(self.categorical_vocab[col]) + 1  # + UNKNOWN bucket
        return dim

    def to_dict(self) -> dict:
        return {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "numeric_stats": {k: vars(v) for k, v in self.numeric_stats.items()},
            "categorical_vocab": self.categorical_vocab,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "_TypePreprocessor":
        obj = cls(numeric_columns=d["numeric_columns"], categorical_columns=d["categorical_columns"])
        obj.numeric_stats = {k: _NumericStat(**v) for k, v in d["numeric_stats"].items()}
        obj.categorical_vocab = d["categorical_vocab"]
        return obj


@dataclass
class FeaturePreprocessor:
    by_type: dict[NodeType, _TypePreprocessor] = field(default_factory=dict)

    def fit(
        self,
        frames: dict[NodeType, pd.DataFrame],
        numeric_columns: dict[NodeType, list[str]],
        categorical_columns: dict[NodeType, list[str]],
        fit_masks: dict[NodeType, pd.Series],
    ) -> "FeaturePreprocessor":
        for node_type, df in frames.items():
            tp = _TypePreprocessor(numeric_columns=numeric_columns[node_type], categorical_columns=categorical_columns[node_type])
            tp.fit(df, fit_masks[node_type])
            self.by_type[node_type] = tp
        return self

    def transform(self, node_type: NodeType, df: pd.DataFrame) -> pd.DataFrame:
        return self.by_type[node_type].transform(df)

    def output_dim(self, node_type: NodeType) -> int:
        return self.by_type[node_type].output_dim()

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for node_type, tp in self.by_type.items():
            with open(os.path.join(output_dir, f"{node_type.value}_preprocessor.json"), "w") as f:
                json.dump(tp.to_dict(), f, indent=2)

    @classmethod
    def load(cls, input_dir: str) -> "FeaturePreprocessor":
        obj = cls()
        for node_type in NodeType:
            path = os.path.join(input_dir, f"{node_type.value}_preprocessor.json")
            if os.path.exists(path):
                with open(path) as f:
                    obj.by_type[node_type] = _TypePreprocessor.from_dict(json.load(f))
        return obj


def full_time_fit_mask(df: pd.DataFrame, train_times: set[int]) -> pd.Series:
    """Fit mask for material/plant/product frames indexed by (id, time):
    every row whose `time` level is a train-split prediction time."""
    times = df.index.get_level_values("time")
    return pd.Series(np.isin(times, list(train_times)), index=df.index)


def supplier_fit_mask(df: pd.DataFrame, train_examples: pd.DataFrame) -> pd.Series:
    """Fit mask for the supplier frame: exactly the (supplier_id, time)
    pairs that are train-split prediction examples."""
    train_keys = set(zip(train_examples["supplier_id"], train_examples["time"]))
    index_keys = list(zip(df.index.get_level_values(0), df.index.get_level_values(1)))
    return pd.Series([k in train_keys for k in index_keys], index=df.index)
