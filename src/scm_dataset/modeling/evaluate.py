"""Prediction generation, metrics, calibration, risk ranking, and plots
(plan §4/§37-49/§65).

`evaluate_experiment` produces predictions for every (supplier, prediction
time) example in every split at once (one forward pass per snapshot,
`model.eval()` / `torch.no_grad()`), then:

- selects the operating threshold using the VALIDATION split only (plan
  §39/§79 -- never test),
- computes metrics/calibration per split from those same predictions,
- builds `supplier_risk_ranking.csv` from the most recent test-split
  prediction time (the natural "current risk dashboard" reading -- see
  docstring on `build_risk_ranking`),
- and renders the required plots from the actual numbers computed above
  (plan §62: "do not fabricate plots").

Nothing here ever selects a checkpoint, threshold, or feature using test
labels (plan §63/§79).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from ..schema.nodes import NodeType  # noqa: E402
from .calibration import compute_calibration  # noqa: E402
from .config import ThresholdConfig  # noqa: E402
from .graphsage import HeteroGraphSAGE  # noqa: E402
from .metrics import compute_classification_metrics, select_threshold  # noqa: E402
from .pipeline import PreparedData  # noqa: E402

SPLITS = ("train", "validation", "test")


def generate_predictions(model: HeteroGraphSAGE, prepared: PreparedData) -> pd.DataFrame:
    model.eval()
    supplier_order = prepared.snapshot_builder.supplier_id_order()
    all_times = sorted(prepared.examples["time"].unique())
    rows = []
    with torch.no_grad():
        for t in all_times:
            data = prepared.snapshot_builder.build(int(t))
            logits = model(data.x_dict, data.edge_index_dict)
            probs = torch.sigmoid(logits).numpy()
            for i, supplier_id in enumerate(supplier_order):
                rows.append({"supplier_id": supplier_id, "time": int(t), "risk_probability": float(probs[i])})
    pred_df = pd.DataFrame(rows)

    merged = prepared.examples.merge(pred_df, on=["supplier_id", "time"], how="left")
    merged = merged.rename(columns={"target": "actual_disruption"})
    if merged["risk_probability"].isna().any():
        raise ValueError("some prediction examples got no model prediction -- snapshot/time mismatch")
    if not merged["risk_probability"].between(0.0, 1.0).all():
        raise ValueError("risk_probability outside [0, 1] -- sigmoid output should never happen; check model output")
    return merged[["supplier_id", "time", "risk_probability", "actual_disruption", "split"]]


def build_supplier_context(prepared: PreparedData) -> pd.DataFrame:
    rows = []
    for node in prepared.benchmark.graph.nodes_of_type(NodeType.SUPPLIER):
        rows.append({"supplier_id": node.supplier_id, "criticality": node.criticality, "region_id": node.region_id})
    return pd.DataFrame(rows)


def build_risk_ranking(predictions: pd.DataFrame, threshold: float, supplier_context: pd.DataFrame) -> pd.DataFrame:
    """plan §42/§49: `rank, supplier_id, time, risk_probability,
    predicted_disruption, criticality, region_id`, sorted by
    `risk_probability` descending. Uses the most recent TEST-split
    prediction time as the ranking snapshot (the plan's own example table
    shows one row per supplier, not per supplier x time, implying a single
    "as of now" reading rather than every example repeated) -- falls back
    to the most recent time overall if the test split is empty (e.g. a
    scenario/severity generalization run where held-out periods may not
    include the latest week)."""
    test_times = predictions.loc[predictions["split"] == "test", "time"]
    ranking_time = int(test_times.max()) if len(test_times) else int(predictions["time"].max())

    snapshot = predictions[predictions["time"] == ranking_time].copy()
    snapshot["predicted_disruption"] = (snapshot["risk_probability"] >= threshold).astype(int)
    snapshot = snapshot.merge(supplier_context, on="supplier_id", how="left")
    snapshot = snapshot.sort_values("risk_probability", ascending=False).reset_index(drop=True)
    snapshot.insert(0, "rank", np.arange(1, len(snapshot) + 1))
    return snapshot[["rank", "supplier_id", "time", "risk_probability", "predicted_disruption", "criticality", "region_id"]]


def compute_warning_times(predictions: pd.DataFrame, raw_supplier_labels: pd.DataFrame, horizon_periods: int) -> pd.DataFrame:
    """plan §40 (secondary operational metric): for every raw 0->1
    disruption onset in `labels/supplier_labels.csv`, the earliest
    prediction time strictly before the onset whose `predicted_disruption`
    was already 1 for an example whose target window (t, t+H] covers that
    onset. Uses only predictions already computed at times < onset -- no
    future information is consulted to determine "when the model warned.\""""
    raw = raw_supplier_labels.sort_values(["supplier_id", "time"])
    onsets = []
    for supplier_id, g in raw.groupby("supplier_id"):
        prev = 0
        for row in g.itertuples():
            if row.supplier_disrupted == 1 and prev == 0:
                onsets.append((supplier_id, int(row.time)))
            prev = row.supplier_disrupted
    onset_df = pd.DataFrame(onsets, columns=["supplier_id", "onset_time"])
    if onset_df.empty:
        return pd.DataFrame(columns=["supplier_id", "onset_time", "first_warning_time", "warning_periods"])

    positive_preds = predictions[predictions["predicted_disruption"] == 1][["supplier_id", "time"]]
    rows = []
    for supplier_id, onset_time in onset_df.itertuples(index=False):
        window_start = max(0, onset_time - horizon_periods)
        candidates = positive_preds[
            (positive_preds["supplier_id"] == supplier_id)
            & (positive_preds["time"] < onset_time)
            & (positive_preds["time"] >= window_start)
        ]
        if len(candidates):
            first_t = int(candidates["time"].min())
            rows.append({"supplier_id": supplier_id, "onset_time": onset_time, "first_warning_time": first_t, "warning_periods": onset_time - first_t})
        else:
            rows.append({"supplier_id": supplier_id, "onset_time": onset_time, "first_warning_time": None, "warning_periods": None})
    return pd.DataFrame(rows)


def disruption_onset_breakdown(predictions: pd.DataFrame, raw_supplier_labels: pd.DataFrame) -> dict:
    """Splits every actual-positive example into two very different cases
    and reports recall AND ranking quality (PR-AUC/ROC-AUC against the
    split's negatives) separately for each, per split:

    - "already ongoing": `supplier_disrupted[t] == 1` -- the supplier is
      already visibly disrupted at the prediction time itself, so the
      target window's positive label mostly reflects the same disruption
      persisting, not a new one appearing. Legitimate dynamic features
      (e.g. a collapsing `fulfillment_ratio`) can make this easy to detect
      without any leakage -- it just isn't early warning of anything new.
    - "fresh onset": `supplier_disrupted[t] == 0` -- nothing is visibly
      wrong yet at `t`; a positive prediction here is a genuine advance
      warning of a disruption that hasn't started.

    A blended recall/PR-AUC over "actual_disruption" can look strong while
    recall on fresh onsets alone is near zero -- this was found by hand
    while investigating why the severity-generalization run's recall was
    bit-identical across 5 independent seeds (it turned out to be catching
    the exact same 360 already-ongoing examples every time, and zero of
    the 84 fresh onsets in every seed). Surfacing it as a real, saved
    metric rather than a one-off check.

    `pr_auc_fresh_onset`/`roc_auc_fresh_onset` (GRAPH_SAGE_IMPROVEMENT_PLAN.md
    §5/§14): computed by pairing ONLY the fresh-onset positives against
    the split's negatives (already-ongoing positives excluded from that
    comparison) -- this isolates ranking quality on the genuinely hard
    subgroup instead of drowning it in the easy one. Same construction for
    `*_already_ongoing`."""
    raw_lookup = raw_supplier_labels.set_index(["supplier_id", "time"])["supplier_disrupted"]
    positives = predictions[predictions["actual_disruption"] == 1].copy()
    if positives.empty:
        return {}
    positives["already_disrupted_at_t"] = [
        int(raw_lookup.get((sid, t), 0)) for sid, t in zip(positives["supplier_id"], positives["time"])
    ]

    def _ranking_metrics(positive_subset: pd.DataFrame, negatives: pd.DataFrame) -> tuple[float | None, float | None]:
        if positive_subset.empty:
            return None, None
        combined = pd.concat([positive_subset, negatives])
        m = compute_classification_metrics(combined["actual_disruption"].values, combined["risk_probability"].values, threshold=0.5)
        return m["pr_auc"], m["roc_auc"]

    result = {}
    for split in SPLITS:
        split_all = predictions[predictions["split"] == split]
        negatives = split_all[split_all["actual_disruption"] == 0]
        sub = positives[positives["split"] == split]
        if sub.empty:
            continue
        fresh = sub[sub["already_disrupted_at_t"] == 0]
        ongoing = sub[sub["already_disrupted_at_t"] == 1]
        fresh_pr_auc, fresh_roc_auc = _ranking_metrics(fresh, negatives)
        ongoing_pr_auc, ongoing_roc_auc = _ranking_metrics(ongoing, negatives)
        result[split] = {
            "n_positive": int(len(sub)),
            "n_fresh_onset": int(len(fresh)),
            "n_already_ongoing": int(len(ongoing)),
            "recall_fresh_onset": float(fresh["predicted_disruption"].mean()) if len(fresh) else None,
            "recall_already_ongoing": float(ongoing["predicted_disruption"].mean()) if len(ongoing) else None,
            "pr_auc_fresh_onset": fresh_pr_auc,
            "roc_auc_fresh_onset": fresh_roc_auc,
            "pr_auc_already_ongoing": ongoing_pr_auc,
            "roc_auc_already_ongoing": ongoing_roc_auc,
        }
    return result


@dataclass
class EvaluationResult:
    predictions: pd.DataFrame
    threshold: float
    threshold_policy: str
    metrics_by_split: dict
    calibration_by_split: dict
    risk_ranking: pd.DataFrame
    warning_times: pd.DataFrame
    onset_breakdown: dict


def evaluate_experiment(model: HeteroGraphSAGE, prepared: PreparedData, threshold_cfg: ThresholdConfig) -> EvaluationResult:
    predictions = generate_predictions(model, prepared)

    val = predictions[predictions["split"] == "validation"]
    threshold = select_threshold(
        val["actual_disruption"].values, val["risk_probability"].values,
        policy=threshold_cfg.policy, value=threshold_cfg.value, target_value=threshold_cfg.target_value,
    )
    predictions = predictions.copy()
    predictions["predicted_disruption"] = (predictions["risk_probability"] >= threshold).astype(int)

    metrics_by_split, calibration_by_split = {}, {}
    for split in SPLITS:
        sub = predictions[predictions["split"] == split]
        if sub.empty:
            continue
        m = compute_classification_metrics(sub["actual_disruption"].values, sub["risk_probability"].values, threshold)
        m["prediction_horizon"] = prepared.config.prediction.horizon_periods
        metrics_by_split[split] = m
        calibration_by_split[split] = compute_calibration(sub["actual_disruption"].values, sub["risk_probability"].values)

    supplier_context = build_supplier_context(prepared)
    risk_ranking = build_risk_ranking(predictions, threshold, supplier_context)
    warning_times = compute_warning_times(predictions, prepared.benchmark.labels["supplier"], prepared.config.prediction.horizon_periods)
    onset_breakdown = disruption_onset_breakdown(predictions, prepared.benchmark.labels["supplier"])

    return EvaluationResult(
        predictions=predictions, threshold=threshold, threshold_policy=threshold_cfg.policy,
        metrics_by_split=metrics_by_split, calibration_by_split=calibration_by_split,
        risk_ranking=risk_ranking, warning_times=warning_times, onset_breakdown=onset_breakdown,
    )


# ---------------------------------------------------------------------------
# Plots (plan §62/§65) -- rendered only from already-computed numbers.
# ---------------------------------------------------------------------------


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, path: str) -> None:
    if len(np.unique(y_true)) < 2:
        return
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, path: str) -> None:
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label="GraphSAGE")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: dict, path: str) -> None:
    matrix = np.array([[cm["tp"], cm["fp"]], [cm["fn"], cm["tn"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues")
    labels = [["TP", "FP"], ["FN", "TN"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{matrix[i, j]}", ha="center", va="center")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration_curve(calibration: dict, path: str) -> None:
    curve = calibration.get("curve")
    if not curve:
        return
    x = [v for v in curve["mean_predicted_probability"] if v is not None]
    y = [v for v, m in zip(curve["observed_frequency"], curve["mean_predicted_probability"]) if m is not None]
    if not x:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    ax.plot(x, y, marker="o", label="GraphSAGE")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curve")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curve(history: pd.DataFrame, path: str) -> None:
    if history.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["epoch"], history["train_loss"], label="train_loss")
    ax1.plot(history["epoch"], history["validation_loss"], label="validation_loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.set_title("Loss")

    ax2.plot(history["epoch"], history["validation_pr_auc"], label="validation_pr_auc")
    ax2.plot(history["epoch"], history["validation_f1"], label="validation_f1")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score"); ax2.legend(); ax2.set_title("Validation metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_all_plots(result: EvaluationResult, history: pd.DataFrame, plots_dir: str) -> None:
    os.makedirs(plots_dir, exist_ok=True)
    test_pred = result.predictions[result.predictions["split"] == "test"]
    if not test_pred.empty:
        plot_pr_curve(test_pred["actual_disruption"].values, test_pred["risk_probability"].values, os.path.join(plots_dir, "pr_curve.png"))
        plot_roc_curve(test_pred["actual_disruption"].values, test_pred["risk_probability"].values, os.path.join(plots_dir, "roc_curve.png"))
        if "test" in result.metrics_by_split:
            plot_confusion_matrix(result.metrics_by_split["test"]["confusion_matrix"], os.path.join(plots_dir, "confusion_matrix.png"))
        if "test" in result.calibration_by_split:
            plot_calibration_curve(result.calibration_by_split["test"], os.path.join(plots_dir, "calibration_curve.png"))
    plot_training_curve(history, os.path.join(plots_dir, "training_curve.png"))
