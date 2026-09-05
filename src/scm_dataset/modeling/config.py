"""GraphSAGE experiment configuration (plan §44/§60), mirroring
`generator/config.py`'s dataclass + `load_config(path)` convention so the
modeling stage reads the same way the dataset generator does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class DatasetConfig:
    benchmark_path: str = "data/benchmark"
    dataset_id: str = "scm_v1_black_swan_seed43"


@dataclass
class PredictionConfig:
    target: str = "supplier_disrupted"
    horizon_periods: int = 4


@dataclass
class FeaturesConfig:
    rolling_windows: list[int] = field(default_factory=lambda: [4, 8, 12])
    min_history_periods: int = 12
    # full | dynamic_only | static_only | region_risk_only | static_plus_graph
    # -- GRAPH_SAGE_IMPROVEMENT_PLAN.md §6 Phase B feature-information
    # ablations. See modeling/features.py::apply_feature_mode for exactly
    # what each mode restricts and why.
    feature_mode: str = "full"


@dataclass
class SplitConfig:
    # temporal | scenario | severity. For "scenario"/"severity", the held-out
    # event type(s)/severity threshold are NOT configured here -- they were
    # already fixed when the benchmark's own splits/{scenario,severity}_split.csv
    # were generated (see scripts/build_benchmark.py's SCENARIO_SPLIT_TEST_TYPES
    # and severity_split()'s train_max_severity default). This modeling stage
    # reads those files as-is (plan §27/§30: "use the existing benchmark split
    # files ... do not modify the scenario split merely to improve results"),
    # it does not re-derive a split from events.csv with its own parameters.
    strategy: str = "temporal"
    train_frac: float = 0.7
    val_frac: float = 0.15
    generalization_val_frac: float = 0.15


@dataclass
class ModelConfig:
    name: str = "hetero_graphsage"
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.20
    aggregation: str = "mean"


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 10
    class_weighting: str = "balanced"  # balanced | none


@dataclass
class ThresholdConfig:
    policy: str = "fixed"  # fixed | f1_optimal | recall_constrained | precision_constrained
    value: float = 0.5
    target_value: float = 0.8


@dataclass
class ExperimentConfig:
    output_dir: str = "experiments/classical_gnn"
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])


@dataclass
class GraphSAGEConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    seed: int = 42


def load_config(path: str) -> GraphSAGEConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return GraphSAGEConfig(
        dataset=DatasetConfig(**raw.get("dataset", {})),
        prediction=PredictionConfig(**raw.get("prediction", {})),
        features=FeaturesConfig(**raw.get("features", {})),
        split=SplitConfig(**raw.get("split", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        threshold=ThresholdConfig(**raw.get("threshold", {})),
        experiment=ExperimentConfig(**raw.get("experiment", {})),
        seed=raw.get("seed", 42),
    )
