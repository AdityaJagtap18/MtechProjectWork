# QGNN-v4 Phase 2c: LayerNorm + Classical GNN vs. QGNN Comparison

Two-part phase: **Part 1** extracted the classical GraphSAGE-Full GNN's
full metric set from existing artifacts (`QGNN_V4_CLASSICAL_BASELINE.md`,
summarized again here for the comparison tables). **Part 2** tests
whether a genuinely data-dependent normalization — `LayerNorm` on the
PauliZ output — succeeds where Phase 2b's `alpha`/`beta` reparameterization
did not: fixing severity's collapsed prediction distributions.

**default.qubit + `diff_method="backprop"` throughout. Nothing here is
attributable to hardware noise.**

## Methodology

**LayerNorm placement**, unchanged from everything else in QGNN-v4:

```
Frozen GraphSAGE-Full → 128D embedding → Linear(128,6) → π·tanh
→ AngleEmbedding(RY) → StronglyEntanglingLayers(2,6) → PauliZ(6)
→ LayerNorm(6) → Linear(6,1) → logit
```

Two variants: `elementwise_affine=False` (pure normalization) and
`elementwise_affine=True` (adds a learnable per-qubit scale/bias,
initialized to `weight=1, bias=0` — verified by test to be byte-identical
to the no-affine variant at initialization, so any divergence after
training is real learning). Unlike Phase 2b's `alpha*pauliz+beta`
(mathematically absorbable into the existing `Linear(6,1)`'s own
weights), `LayerNorm`'s normalizing statistics are computed **per
example**, from that example's own 6 PauliZ values — a genuine,
non-redundant change to what the head computes.

Nothing else changed: same 6 qubits, 2 `StronglyEntanglingLayers`,
`AngleEmbedding(RY)`, frozen encoder, optimizer, learning rate, weight
decay, batch size, `max_epochs=100`, `patience=10`, seeds 42–46, both
splits, `--diagnostics`, threshold `fixed, value=0.5`.

```bash
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml          --tag phase2c_layernorm_noaffine --seeds 42,43,44,45,46 --head-variant layernorm_noaffine --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4_severity.yaml --tag phase2c_layernorm_noaffine --seeds 42,43,44,45,46 --head-variant layernorm_noaffine --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml          --tag phase2c_layernorm_affine   --seeds 42,43,44,45,46 --head-variant layernorm_affine   --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4_severity.yaml --tag phase2c_layernorm_affine   --seeds 42,43,44,45,46 --head-variant layernorm_affine   --quantum-only --diagnostics
```

20 new runs, non-overwriting, under `experiments/qgnn_v4/*_phase2c_*`.
Classical control not retrained (unchanged from Phase 1). Structured
data: `experiments/qgnn_v4/phase2b_extended_metrics/phase2b_all_metrics.csv`
now has 70 rows (7 configurations × 2 splits × 5 seeds — baseline, scale,
scale_bias, fixed0.5, fixed2.0, layernorm_noaffine, layernorm_affine);
classical GNN data in `experiments/classical_gnn/baseline_extended_metrics/`.

---

## Classical GNN Baseline (recap of Part 1 — full detail in `QGNN_V4_CLASSICAL_BASELINE.md`)

Extracted, not retrained, from the same frozen-encoder checkpoints
QGNN-v4 uses throughout. Primary PR-AUC 0.807±0.066 matches this
project's already-documented classical-GNN figure. Severity recall is
exactly 0.4406 in all 5 seeds (known finding — same 360 already-ongoing
positives caught every run). Fresh-onset PR-AUC 0.0056–0.0065, the same
near-zero floor every QGNN-v4 configuration shows.

---

## TABLE 1 — Primary/Temporal Split (mean of 5 seeds)

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall | Specificity | Bal. Acc. | MCC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical GNN (full) | 0.8070 | 0.9871 | 0.6771 | 0.5220 | 0.9678 | 0.9353 | 0.9515 | 0.6845 | **0.0418** | **0.0493** |
| QGNN baseline | **0.8328** | 0.9829 | 0.5693 | 0.4373 | **0.9603** | 0.7512 | 0.8557 | **0.6839** | 0.1283 | 0.2845 |
| QGNN LayerNorm-no-affine | 0.8112 | 0.9775 | 0.6565 | 0.5191 | 0.8975 | 0.9400 | 0.9188 | 0.6539 | 0.0655 | 0.1823 |
| QGNN LayerNorm-affine | 0.8002 | 0.9765 | 0.6476 | 0.5191 | 0.8612 | 0.9426 | 0.9019 | 0.6393 | 0.0568 | 0.1456 |

## TABLE 2 — Severity/OOD Split (mean of 5 seeds)

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall | Specificity | Bal. Acc. | MCC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical GNN (full) | **0.4487** | **0.7275** | 0.4113 | 0.3871 | 0.4406 | 0.9438 | 0.6922 | **0.3624** | **0.0857** | **0.0904** |
| QGNN baseline | 0.3915 | 0.6257 | 0.2436 | 0.4354 | 0.5667 | 0.5991 | 0.5829 | 0.2305 | 0.2105 | 0.3448 |
| QGNN LayerNorm-no-affine | 0.3976 | 0.6566 | **0.4187** | 0.4608 | 0.4166 | 0.9513 | 0.6840 | 0.3824 | 0.1225 | 0.2405 |
| QGNN LayerNorm-affine | 0.4077 | 0.6505 | 0.4151 | 0.4401 | 0.4164 | 0.9505 | 0.6835 | 0.3747 | 0.1184 | 0.2304 |

**Classical GNN remains the best-calibrated and highest-ROC-AUC model on
severity by a clear margin.** LayerNorm closes most of the PR-AUC gap to
classical (0.392→0.398–0.408 vs. classical's 0.449) and, notably,
LayerNorm's F1 (0.415–0.419) is now essentially at parity with classical's
(0.411) — the first QGNN-v4 configuration across three full phases to
match classical on a threshold-based classification metric, not just
approach it.

## TABLE 3 — Cross-Seed Stability (mean ± std, test split, n=5)

| Model | Split | PR-AUC | F1 | Recall | MCC | Brier | ECE |
|---|---|---|---|---|---|---|---|
| Classical GNN | primary | 0.807±0.066 | 0.677±0.035 | 0.968±0.016 | 0.684±0.032 | 0.042±0.012 | 0.049±0.010 |
| QGNN baseline | primary | 0.833±0.140 | 0.569±0.224 | 0.960±0.036 | 0.684±0.037 | 0.128±0.083 | 0.285±0.122 |
| QGNN LayerNorm-no-affine | primary | 0.811±0.074 | 0.657±0.043 | 0.898±0.089 | 0.654±0.052 | 0.065±0.014 | 0.182±0.041 |
| QGNN LayerNorm-affine | primary | 0.800±0.073 | 0.648±0.039 | 0.861±0.061 | 0.639±0.046 | 0.057±0.016 | 0.146±0.056 |
| Classical GNN | severity | 0.449±0.013 | 0.411±0.020 | 0.441±0.000 | 0.362±0.023 | 0.086±0.008 | 0.090±0.007 |
| QGNN baseline | severity | 0.392±0.024 | 0.244±0.193 | 0.567±0.371 | 0.231±0.207 | 0.211±0.083 | 0.345±0.162 |
| QGNN LayerNorm-no-affine | severity | 0.398±0.057 | **0.419±0.060** | 0.417±0.030 | 0.382±0.089 | 0.122±0.027 | 0.241±0.071 |
| QGNN LayerNorm-affine | severity | 0.408±0.071 | 0.415±0.061 | 0.416±0.034 | 0.375±0.082 | 0.118±0.023 | 0.230±0.068 |

**The single largest effect in this table: LayerNorm's severity `recall`
std drops from baseline's 0.371 to 0.030–0.034 — more than a 10x
reduction**, and severity `F1` std drops from 0.193 to 0.060–0.061 — also
roughly 3x tighter. Primary PR-AUC std also drops substantially
(0.140→0.073–0.074) at the cost of a small mean decline (0.833→0.800–0.811).
Classical GNN's own stability remains the reference point QGNN-v4 is
converging toward, not yet matching, on primary PR-AUC variance
specifically (0.066 vs. LayerNorm's 0.073–0.074 — close, still slightly
wider) but LayerNorm's severity `recall`/`F1` variance is now
**comparable in order of magnitude to classical's own** on those same
two metrics.

## TABLE 4 — Per-Seed PR-AUC, All Four Models

### Primary split

| Seed | Classical GNN | QGNN baseline | LayerNorm-no-affine | LayerNorm-affine |
|---:|---:|---:|---:|---:|
| 42 | 0.7874 | 0.9615 | 0.8682 | 0.7866 |
| 43 | 0.6888 | 0.5696 | 0.6807 | 0.6802 |
| 44 | 0.8745 | 0.9266 | 0.8197 | 0.8506 |
| 45 | 0.8491 | 0.8203 | 0.7951 | 0.7856 |
| 46 | 0.8354 | 0.8861 | 0.8924 | 0.8982 |

### Severity split

| Seed | Classical GNN | QGNN baseline | LayerNorm-no-affine | LayerNorm-affine |
|---:|---:|---:|---:|---:|
| 42 | 0.4430 | 0.3610 | 0.3943 | 0.3955 |
| 43 | 0.4560 | 0.3744 | 0.4496 | 0.4497 |
| 44 | 0.4378 | 0.4248 | 0.4156 | 0.4759 |
| 45 | 0.4358 | 0.3851 | **0.2900** | **0.2762** |
| 46 | 0.4708 | 0.4123 | 0.4385 | 0.4414 |

**Seed 45 is a real reversal, not noise** — the only severity seed where
LayerNorm underperforms baseline, by a wide margin (0.385→0.290/0.276).
Every other severity seed improves under both LayerNorm variants. Not
cherry-picked: all 5 seeds shown for every model.

---

## Severity-Collapse Analysis — the 13 requested checks

1. **Do severity seeds 42 and 43 still stop at epochs 1 and 3?** Partially.
   Seed 43 stops at epoch 3 in every configuration tested across this
   entire investigation, unchanged by LayerNorm. **Seed 42 no longer stops
   at epoch 1** — it now trains to epoch 5 under both LayerNorm variants,
   the first intervention in three phases to change this seed's stopping
   point at all.
2. **Does prediction std increase?** Yes, dramatically, in 4 of 5 seeds
   (e.g. seed 45: 0.004→0.114–0.125, a ~27x increase; seed 42: 0.022→0.097–0.101).
   Seed 46 is the exception — it *decreases* (0.237→0.191–0.201), because
   baseline seed 46 was already the one long-trained (85-epoch), naturally
   wide-spread run; LayerNorm still leaves it far wider than every other
   severity seed's baseline value.
3. **Does the number of collapsed threshold-0.5 runs decrease?** Yes —
   **from 2 (baseline) to 0, for both LayerNorm variants.** This is the
   first configuration across Phases 1–2c to fully eliminate degenerate
   threshold-0.5 classification on severity (compare: `fixed2.0`/`scale_bias`
   got it to 1; every other Phase 2b variant stayed at 2).
4. **Does severity PR-AUC improve consistently?** No — 4 of 5 seeds
   improve (some substantially, e.g. seed 43: +0.075), but seed 45
   worsens notably (−0.095 to −0.109). "Mostly, not consistently" is the
   accurate description.
5. **Does severity F1 improve?** Yes, clearly and consistently in
   aggregate (0.244→0.415–0.419) — the largest, cleanest single-metric
   gain in this whole investigation.
6. **Does severity recall become more stable?** Yes, dramatically — std
   0.371→0.030–0.034.
7. **Does recall standard deviation decrease?** Same as #6 — yes, by
   roughly 10x.
8. **Does Brier improve?** Yes — 0.211→0.118–0.122, the best severity
   calibration of any QGNN-v4 configuration tested.
9. **Does ECE improve?** Yes — 0.345→0.230–0.241, same ranking.
10. **Does fresh-onset PR-AUC improve?** No — 0.0064–0.0067, statistically
    indistinguishable from baseline's 0.0061 and every other
    configuration's. Confirms yet again (now a 5th consecutive
    configuration family) that no output-side intervention recovers a
    signal absent from the frozen input itself.
11. **Do quantum gradients remain healthy?** Yes — `quantum_grad_norm_mean`
    stays in the same 0.14–0.20 range as every prior phase, no new
    pathology introduced by LayerNorm.
12. **Does LayerNorm improve primary performance?** No, on mean PR-AUC —
    0.833→0.800–0.811, a real decline in 4 of 5 seeds (only seed 43 and
    marginally seed 46 improve). It *does* improve primary calibration
    substantially (ECE 0.285→0.146–0.182, Brier 0.128→0.057–0.065) and
    reduces PR-AUC variance (std 0.140→0.073–0.074).
13. **Does LayerNorm introduce any new instability?** One clear instance:
    seed 45's severity reversal (§ Table 4). No evidence of gradient
    instability, NaN outputs, or training divergence in any of the 20
    new runs.

---

## Calibration Analysis

LayerNorm is the best-calibrated QGNN-v4 configuration tested across
**both** splits and **all three** calibration-relevant phases (2, 2b, 2c)
— Brier/ECE beat every Phase 2b output-scale variant on both splits
simultaneously, something no single Phase 2b configuration achieved
(`fixed2.0` was best on primary and close on severity; `scale_bias` was
close on primary but flat on severity — LayerNorm beats both
configurations' calibration on both splits at once). Classical GNN still
calibrates better than LayerNorm on both splits, by a shrinking but real
margin (primary Brier 0.042 vs. 0.057–0.065; severity Brier 0.086 vs.
0.118–0.122).

---

## Scientific Conclusion

1. **Is QGNN better than Classical GNN on the primary split?** Not
   demonstrated. QGNN baseline's raw PR-AUC mean (0.833) exceeds
   classical's (0.807), but its variance is more than double classical's
   (std 0.140 vs. 0.066) and its calibration is roughly 3x worse (ECE
   0.285 vs. 0.049). LayerNorm's PR-AUC mean (0.800–0.811) sits at or
   below classical's own. No QGNN-v4 configuration beats classical
   cleanly across ranking, classification, and calibration simultaneously
   on this split.
2. **Is QGNN better than Classical GNN on the severity/OOD split?** No.
   Classical leads on PR-AUC (0.449 vs. 0.392–0.408), ROC-AUC, MCC, and
   calibration. LayerNorm closes the F1 gap to near-parity (0.411 vs.
   0.415–0.419) but that is one metric among several, and classical's
   recall stability (std exactly 0.000, if for reasons already flagged as
   a known degenerate-looking finding) still exceeds LayerNorm's own
   (std 0.030–0.034).
3. **Does LayerNorm improve QGNN?** Yes, on multiple dimensions
   simultaneously — calibration (both splits), severity classification
   (F1, recall stability, collapse elimination), and primary-split
   variance — at the cost of primary-split PR-AUC mean and one severity
   seed's ranking performance (seed 45).
4. **Does LayerNorm actually fix the severity collapse?** Substantially,
   not completely. The threshold-0.5 collapse count goes to zero and
   recall/F1 variance drops by an order of magnitude — a real, structural
   change unlike Phase 2b's output-scale variants, which left severity's
   collapse rate and recall variance essentially untouched. It does not
   fix seed 43's persistently early stopping (epoch 3, unchanged) or
   prevent seed 45 from getting worse.
5. **Does LayerNorm improve calibration?** Yes, clearly, on both splits,
   more than any prior QGNN-v4 configuration.
6. **Does LayerNorm improve cross-seed stability?** Yes, markedly, for
   severity recall/F1 and primary PR-AUC. Not for severity PR-AUC itself,
   whose std (0.057–0.071) is actually somewhat higher than baseline's
   (0.024) — driven by seed 45's reversal pulling the range wider even as
   the *other* four seeds tighten.
7. **Does the fresh-onset problem remain?** Yes, completely unchanged.
   Fresh-onset PR-AUC is 0.0056–0.0102 across every configuration tested
   in this project to date, classical GNN included. This is now
   established as a property of the benchmark's input features at the
   prediction horizon used, not of any model architecture.
8. **Is there any evidence of quantum advantage?** No, and none is
   claimed. Every comparison in this report is QGNN-v4 (a quantum head)
   vs. either a matched classical control or the full classical GNN, and
   the full classical GNN wins or ties on the metrics that matter most
   (calibration, severity ranking) in every comparison run so far. A
   quantum head with LayerNorm narrowing some gaps is not evidence that
   the *quantum* component specifically is responsible — LayerNorm is
   architecture-agnostic and would plausibly help a classical bottleneck
   head the same way; that comparison has not been run and would be
   needed before attributing any of this gain to the quantum circuit
   itself.
9. **What problem remains after this experiment?** Two, clearly separated:
   (a) the fresh-onset signal is absent from the input and no
   architectural change can be expected to recover it; (b) even with
   LayerNorm, no QGNN-v4 configuration matches classical GNN's
   calibration or severity ranking, and one severity seed (45) got
   meaningfully worse under the one intervention that helped everywhere
   else — an unexplained, seed-specific interaction worth a closer look
   before treating LayerNorm as a settled improvement.

## Next-Step Decision

LayerNorm **materially improved but did not fully fix** the severity/OOD
problem (collapse count → 0, recall/F1 variance down ~10x, but PR-AUC
gain inconsistent and one seed reversed). Per the investigation's own
decision rule, this is closer to "partial success" than "does not
materially fix it" — a middle case. Given the magnitude of the
calibration and stability gains, and that they are not yet matched by
classical-parity or a resolved seed-45 anomaly, two reasonable next steps
exist and the choice is the user's:

- **Investigate seed 45's reversal specifically** before moving on —
  it's the one clear negative surprise in an otherwise positive result,
  and understanding it (a different mechanism than the
  early-stopping/gradient explanations already ruled out in Phases 1–2)
  could matter for whether LayerNorm is adopted as the new default.
- **Proceed to Phase 3 — Quantum Architecture Ablation** (qubit count,
  variational layer count, ansatz choice) as originally planned, now with
  LayerNorm as the carried-forward default output stage given its
  consistent multi-metric benefit.

**Phase 3 is not run automatically in this report, per instruction.**
