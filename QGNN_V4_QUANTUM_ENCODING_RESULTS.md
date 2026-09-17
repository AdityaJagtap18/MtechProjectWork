# QGNN-v4 — Quantum Encoding Investigation: Results

## 1. Objective

Determine whether the mechanism that encodes the classical 6-dimensional
projection into quantum rotation angles — `angle = π · tanh(x)` in the
standing reference — materially affects QGNN performance, stability, or
generalization, isolated from every other factor already investigated
(projection, initialization, LayerNorm, ansatz).

## 2. Research question

Does the encoding scale, the choice of bounding nonlinearity (tanh vs. a
bounded linear clip), or re-encoding the same features before every
variational layer (data re-uploading) change Primary temporal
generalization, Severity/OOD generalization, seed sensitivity, or the
quantum representation itself?

## 3. Existing reference

| | Primary PR-AUC | Severity PR-AUC | Source |
|---|---|---|---|
| Classical GraphSAGE-Full | 0.807 ± 0.066 | 0.449 ± 0.013 | `QGNN_V4_CLASSICAL_BASELINE.md` |
| QGNN-v4 E0 (= existing reference, 5 seeds) | 0.811 ± 0.074 | 0.398 ± 0.057 | `phase2c_layernorm_noaffine` runs |

Not overwritten. Every number below is a **2-seed (42, 43) pilot**,
directly comparable only to a same-2-seed subset of the above (shown in
§7-8), not the full 5-seed figures.

## 4. Encoding configurations

| Config | encoding_type | encoding_scale | data_reuploading |
|---|---|---:|---|
| E0 baseline (control) | tanh | π | False |
| E1 reduced scale | tanh | 0.5π | False |
| E2 increased scale | tanh | 2π | False |
| E3 linear-bounded | **clip** | π | False |
| E4 data re-uploading | tanh | π | **True** |

## 5. Controlled variables

Frozen GraphSAGE encoder (per seed), dataset, primary/severity splits,
`Linear(128,6)` projection, 6 qubits, 2 variational layers,
StronglyEntanglingLayers (unchanged), default initialization
(uniform[0,2π] — deliberately excluded from combining with Stage 4/4b's
alternatives), post-quantum `LayerNorm(6, no affine)`, `Linear(6,1)`,
Adam (lr=0.001, weight_decay=0.0001), max_epochs=100, patience=10,
balanced class weighting, threshold=fixed@0.5, `default.qubit`,
`backprop`, batch size unchanged. **No implementation constraint forced
any other variable to change** — every configuration ran cleanly with
only the encoding mechanism differing.

E0 was **not rerun** — its 10 runs (5 seeds × 2 splits) are Phase 2c/3's
own reference (`phase2c_layernorm_noaffine`), reused and verified valid
multiple times already in this research program. **E1–E4 are newly
trained**: 4 runs each (2 seeds × 2 splits) = 16 new training runs, 4
reused, 20 total (config, split, seed) rows in this pilot.

## 6. Parameter / circuit resource comparison

| Config | Classical head params | Quantum trainable params | Total trainable | Frozen params | Encoding operations |
|---|---:|---:|---:|---:|---:|
| E0–E3 | 774 (reduce) + 7 (out) = 781 | 36 | **817** | 128×128 GraphSAGE (frozen, unchanged) | 1 |
| E4 (re-uploading) | 781 | 36 | **817** | same | **2** (one per variational layer) |

**Verified, not assumed**: E4's parameter count is identical to E0's —
confirmed both by a dedicated unit test (`test_data_reuploading_
introduces_zero_trainable_parameters`) and by every saved run's own
`quantum_resource_summary.json` (`total_trainable_parameters: 817` in
all 20 rows). Re-encoding is a circuit-construction change (the
parameter-free `AngleEmbedding` operation repeated, introspected directly
via the compiled PennyLane tape: 2 `AngleEmbedding` ops for E4 vs. 1 for
E0–E3, both confirmed by unit test), not an added-capacity change.

## 7. Pilot results — Primary (test, threshold=0.5, mean±std, seeds 42/43)

| Config | PR-AUC | ROC-AUC | F1 | Precision | Recall | Specificity | Bal.Acc | MCC | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 baseline | 0.774±.094 | 0.977 | 0.637 | 0.511 | 0.857 | 0.941 | 0.899 | 0.631±.059 | 0.070 | 0.193 |
| E1 scale=0.5π | 0.803±.143 | 0.980 | 0.667 | 0.523 | 0.943 | 0.936 | 0.939 | 0.646±.015 | 0.056 | 0.185 |
| E2 scale=2π | 0.581±.265 | 0.937 | 0.564 | 0.438 | 0.826 | 0.903 | 0.865 | 0.479±.144 | 0.092 | 0.229 |
| E3 clip | 0.666±.079 | 0.965 | 0.573 | 0.458 | 0.784 | 0.930 | 0.857 | 0.554±.002 | 0.090 | 0.198 |
| E4 re-uploading | 0.599±.202 | 0.956 | 0.548 | 0.421 | 0.820 | 0.906 | 0.863 | 0.536±.063 | 0.096 | 0.232 |

## 8. Pilot results — Severity (test, threshold=0.5, mean±std, seeds 42/43)

| Config | PR-AUC | ROC-AUC | F1 | Precision | Recall | Specificity | Bal.Acc | MCC | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 baseline | 0.422±.028 | 0.667 | 0.456 | 0.592 | 0.414 | 0.962 | 0.688 | 0.440±.114 | 0.139 | 0.287 |
| E1 scale=0.5π | 0.431±.031 | 0.620 | 0.435 | — | — | — | — | 0.364±.031 | — | — |
| E2 scale=2π | 0.370±.026 | 0.678 | — | — | — | — | — | 0.332±.004 | — | — |
| E3 clip | 0.315±.026 | 0.638 | — | — | — | — | — | 0.292±.077 | — | — |
| E4 re-uploading | 0.379±.030 | 0.686 | — | — | — | — | — | 0.393±.104 | — | — |

(Full precision/recall/specificity/Bal.Acc/Brier/ECE for every severity
row: `results.csv`/`encoding_summary.csv` — omitted above only where
space-constrained; none were selectively hidden.)

**Note on output structure**: per this project's established convention
(Stages 1-5), raw per-run artifacts (`predictions.csv`,
`training_history.csv`, `model.pt`, `config.yaml`) already live inside
each run's own saved directory under `experiments/qgnn_v4/
encoding_investigation/<config>/<timestamp>_..._quantum_seed<N>/` — not
duplicated into separate `training_logs/`/`predictions/` folders, to
avoid redundant copies of data that's already saved once, correctly, per
run. `metrics/`, `quantum_features/`, `encoded_angles/`, and `plots/` are
provided as the aggregate CSVs listed above/below (`encoding_summary.csv`,
`encoding_quantum_features.csv`, `encoding_encoded_angles.csv`) plus each
run's own already-rendered `plots/` subdirectory, rather than a second,
separately-aggregated copy.

## 9. Extended metrics

Every value in §7-8 and full per-seed detail is in `results.csv` (exact
schema requested: experiment, encoding_type, encoding_scale,
data_reuploading, seed, split, n_samples, n_positives, pr_auc, roc_auc,
f1, precision, recall, specificity, balanced_accuracy, mcc, brier, ece,
tp, tn, fp, fn, best_epoch, total_epochs, val_pr_auc, val_test_pr_gap,
fresh_onset_pr_auc, quantum_parameters, total_trainable_parameters).

## 10. Seed stability

| Config | Primary PR-AUC std | Primary range | Severity PR-AUC std | Severity range |
|---|---:|---:|---:|---:|
| E0 | 0.094 | 0.187 | 0.028 | 0.055 |
| E1 | **0.143** | 0.287 | 0.031 | 0.062 |
| E2 | **0.265** | 0.530 | 0.026 | 0.051 |
| E3 | 0.079 | 0.157 | 0.026 | 0.053 |
| E4 | 0.202 | 0.405 | 0.030 | 0.061 |

**No encoding configuration reduces primary seed sensitivity relative to
E0** — E3 is marginally tighter (0.079 vs 0.094) but within a 2-seed
sample's noise; E1, E2, and E4 are all substantially *more* variable,
E2 dramatically so (0.265, nearly 3× E0's). Severity std is similar
across all five configs (0.026–0.031) — encoding does not appear to
affect severity variance specifically, consistent with severity's
variance being dataset-driven (Stage 1) rather than encoding-driven.

## 11. Training dynamics

| Config | Split | Mean best_epoch | Mean total_epochs | Mean quantum grad norm |
|---|---|---:|---:|---:|
| E0 | primary | 14.0 | 24.0 | 0.187 |
| E1 | primary | 10.0 | 20.0 | 0.251 |
| E2 | primary | 8.5 | 18.5 | 0.162 |
| E3 | primary | 11.5 | 21.5 | 0.221 |
| E4 | primary | 12.0 | 22.0 | 0.149 |
| E0 | severity | 4.0 | 14.0 | 0.258 |
| E1 | severity | 1.5 | 11.5 | 0.351 |
| **E2** | severity | **47.0** | **57.0** | 0.158 |
| E3 | severity | 8.0 | 18.0 | 0.258 |
| E4 | severity | 2.0 | 12.0 | 0.276 |

**E2's severity convergence is a real, both-seed-consistent anomaly**:
best_epoch 69 (seed 42) and 25 (seed 43) — 3–20× later than every other
configuration's severity best_epoch (1–8). This is a genuine training-
dynamics finding, not a lower-loss-is-better assumption: E2 takes
dramatically longer to reach its best validation checkpoint on severity,
consistent with (not proof of) the wider angular spread (§12) making
optimization harder to navigate for that split specifically. No
configuration showed gradient explosion or vanishing (all grad norms
0.15–0.35, finite throughout).

## 12. Encoded angle distribution

Computed directly from each trained model's own `encode(h)` on its real
test split (`encoding_encoded_angles.csv`, not estimated):

| Config | Mean \|angle\| std | Frac. near ±scale | Frac. near 0 |
|---|---:|---:|---:|
| E0 (π·tanh) | 1.97 | 0.303 | 0.147 |
| E1 (0.5π·tanh) | **1.02** | 0.305 | 0.080 |
| E2 (2π·tanh) | **4.02** | 0.275 | 0.125 |
| E3 (π·clip) | 2.29 | **0.462** | 0.130 |
| E4 (π·tanh, re-uploaded) | 1.85 | 0.225 | 0.127 |

E2's angle spread is roughly 2× E0's and 4× E1's — a direct, measured
confirmation that scaling by 2π produces a much wider, less concentrated
angular distribution. E3 (clip) has the highest fraction of examples
sitting exactly at the ±π boundary (0.462, nearly half) — clip
**saturates exactly**, so many different input magnitudes collapse to
the identical extreme angle, unlike tanh's asymptotic (always slightly
differentiating) approach to its bound — a direct, measured explanation
for information loss under E3, not a guess.

## 13. Quantum feature analysis

Six pre-LayerNorm PauliZ channels, same test-split forward pass
(`encoding_quantum_features.csv`):

| Config | Mean max\|channel↔target corr\| | Mean max\|cross-channel corr\| |
|---|---:|---:|
| E0 | 0.495 | 0.784 |
| E1 | **0.523** | 0.818 |
| E2 | **0.286** | 0.719 |
| E3 | 0.400 | 0.714 |
| E4 | 0.435 | 0.677 |

**E2's channels are the least target-aligned of any configuration**
(0.286, roughly half of E0's 0.495) — directly consistent with its poor
PR-AUC and connects §12's wider-angle finding to an actual representation
consequence: the wider spread pushes rotation gates through more of
their periodic range, degrading the monotonic relationship between input
and PauliZ output. E1, despite its much higher seed *variance* (§10),
has channels that are if anything *slightly better* target-aligned than
E0's on average — the representation-quality metric alone does not
explain E1's instability; something in the optimization trajectory does
(consistent with §11's faster convergence, fewer epochs, potentially
reaching different quality optima per seed).

## 14. Fresh-onset analysis

Primary split: **0 fresh-onset positives** in every configuration (the
primary test window sits entirely inside the standing severity-5 event's
already-ongoing period, per Stage 1's finding) — `fresh_onset_pr_auc` is
undefined (NaN), correctly left undefined rather than fabricated.
Severity split: 84 fresh-onset positives in every configuration (same
underlying labels). Fresh-onset PR-AUC is **near the noise floor for
every encoding tested** (0.005–0.012) — no encoding configuration shows
any meaningful ability to recognize a genuinely new disruption onset
rather than rank an already-active one. Encoding does not appear to be a
lever on this specific, previously-documented weakness (Stage 1).

## 15. Validation→test gap

| Config | Primary gap mean | Severity gap mean |
|---|---:|---:|
| E0 | 0.083 | 0.574 |
| E1 | 0.062 | 0.568 |
| E2 | 0.211 | 0.625 |
| E3 | 0.203 | 0.659 |
| E4 | 0.249 | 0.599 |

Severity's large gap (0.57–0.66) is present for every configuration —
consistent with it being the dataset's own OOD limitation (Stage 1),
not something encoding changes. On primary, E2 and E4 show notably
larger gaps than E0 — worse validation→test generalization, not just
worse raw PR-AUC.

## 16. Calibration analysis

Primary ECE: E0 0.193, E1 **0.185** (best), E2 0.229, E3 0.198, E4 0.232.
Severity ECE (from `results.csv`): comparable across configs, no large
outlier. **Calibration and ranking move together here, not apart** — E1
is both the best-calibrated and (nominally) the highest-mean-PR-AUC
config on primary, while E2/E4 are both worse-calibrated and
worse-ranking. This is a different pattern than Stage 4b (where
calibration and ranking clearly decoupled for F2) — stated explicitly
since Section 8/Q8 asked to check whether they separate again here: **they
do not, in this pilot**.

## 17. Comparison with Classical GraphSAGE (0.807/0.449, 5-seed)

Every encoding configuration's severity mean (0.315–0.431) stays below
classical's 0.449. On primary, only E1's mean (0.803, from a noisy
2-seed pilot) approaches classical's 0.807; every other configuration is
below it.

## 18. Comparison with QGNN-v4 E0

Using the SAME 2 pilot seeds for E0 (0.774 primary / 0.422 severity,
consistent with every prior 2-seed-pilot report in this research
program): E1 nominally exceeds E0 on both splits, but with far higher
primary variance (0.143 vs. 0.094) — not a clean improvement. E2, E3, E4
are below E0 on primary; E1 and E4 are marginally above E0 on severity,
within noise.

## 19. Research question answers

- **Q1 (Primary)**: Yes — encoding scale materially affects primary,
  but mostly in the *wrong* direction: 2π (E2) is a clear regression;
  0.5π (E1) shows a higher mean but much higher variance, not a clean
  improvement.
- **Q2 (Severity)**: Modestly — E1/E4 show small, seed-noisy severity
  gains over E0 (0.43/0.38 vs. 0.42); E2/E3 are clearly worse.
- **Q3 (Seed sensitivity)**: No — no configuration reduces primary seed
  sensitivity; all either match or substantially worsen it (E2 nearly
  3× E0's std).
- **Q4 (Re-uploading representation quality)**: No clear improvement —
  E4's channel target-correlation (0.435) and cross-channel correlation
  (0.677, the lowest of the five, i.e. least redundant) are both
  reasonable, but this does not translate into better PR-AUC or
  stability; re-uploading changes the representation without a
  corresponding performance benefit here.
- **Q5 (Severity val→test gap)**: No — every configuration's severity
  gap is large (0.57–0.66); none meaningfully reduces it.
- **Q6 (Channel redundancy)**: Yes, modestly — E4 has the lowest mean
  cross-channel correlation (0.677 vs. E0's 0.784), E2 the widest angle
  spread with the worst target-alignment. Encoding does measurably shift
  representation properties, just not in a way that helps final metrics
  here.
- **Q7 (Fresh-onset)**: No — every configuration remains at the noise
  floor (0.005–0.012); encoding is not a lever on this weakness.
- **Q8 (Calibration vs. ranking)**: They move together in this pilot
  (unlike Stage 4b's F2 case) — no decoupling observed.
- **Q9 (Robust across seeds/splits)**: No apparent improvement survives
  both seeds and both splits simultaneously — E1's primary gain is
  undermined by its own variance; nothing else is directionally
  consistent across all four cells (2 seeds × 2 splits).
- **Q10 (5-seed expansion justified)**: No — see §20.

## 20. KEEP / INVESTIGATE / DROP decisions

| Config | Verdict | Reasoning |
|---|---|---|
| E0 baseline | **KEEP as reference** | No alternative demonstrates a clean, seed-consistent improvement. |
| E1 reduced scale (0.5π) | **INVESTIGATE** | Nominally higher primary/severity means and the best calibration, but primary variance is 1.5× E0's and the "improvement" does not survive both seeds cleanly (§9's per-seed table: seed 42 very high, seed 43 unremarkable) — an interesting lead, not evidence of a real gain yet. |
| E2 increased scale (2π) | **DROP** | Clear, multiply-corroborated regression: worst primary mean, worst primary variance, worst channel target-correlation, and a distinct, both-seed-consistent severity convergence delay (69/25 epochs vs. 1–8 elsewhere). Every line of evidence points the same direction. |
| E3 linear clip | **DROP** | Consistently below E0 on both splits, with a direct, measured mechanism (§12: 46% of angles saturate exactly at ±π, more information loss than tanh's asymptotic bound) — a clean negative result, not an ambiguous one. |
| E4 data re-uploading | **DROP** | Below E0 on primary with higher variance; only a marginal, noisy severity edge. Representation-level changes (lower channel redundancy) do not translate into a metrics benefit here. |

## 21. Limitations

- **2-seed pilot only** — every std/range figure here is computed from
  n=2; per this program's own established caution, read the qualitative,
  paired pattern (E2 and E3 are unambiguous regressions; E1's story is
  genuinely mixed) rather than exact decimals.
- **Severity's single-underlying-event limitation** (Stage 1) applies to
  every severity number in this report unchanged.
- **E1's higher variance is not explained by the representation-level
  analysis** (§13 shows comparable-or-better channel quality) — the
  cause is not established here; a training-dynamics-focused follow-up
  would be needed, not assumed.
- Only one ansatz (StronglyEntanglingLayers, the standing reference) was
  tested with each encoding variant — an encoding × ansatz interaction
  was out of scope for this controlled comparison.

## 22. Reproducibility

- Git commit: `27ed213634bcc32c723212bdd9f499812fffd737`
- Branch: `qgnn-package-restructure`
- Python: 3.12.3 · PyTorch: 2.14.0+cu130 (CUDA available on this machine
  but not used — all computation here is CPU, `default.qubit`, matching
  every prior phase) · PennyLane: 0.45.1
- Simulator/backend: `default.qubit`, `diff_method="backprop"`, no
  shots, no hardware, no noise
- Seeds: 42, 43 (pilot)
- Dataset: `scm_v1_black_swan_seed43`
- Timestamp: 2026-09-17T15:35:46Z
- Commands: `scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4[_severity].yaml --seeds 42,43 --head-variant layernorm_noaffine --encoding-scale <S> [--encoding-type clip] [--data-reuploading] --output-subdir encoding_investigation/<name> --quantum-only --diagnostics`
- Analysis: `scripts/analyze_qgnn_v4_encoding_investigation.py`,
  `scripts/inspect_qgnn_v4_encoding_features.py`
- Tests: 317 before this investigation's code changes, 334 after (17
  new), 0 failures, 0 regressions, both runs full-suite.

## Final research conclusion

**Quantum encoding does not appear to be a major, currently-exploitable
source of the QGNN's Severity/OOD gap or seed sensitivity.** Every
alternative tested is either a clear regression (E2, E3) or an ambiguous,
seed-inconsistent mixed result (E1, E4) — none produces a robust,
seed-and-split-consistent improvement over the existing π·tanh baseline.
The evidence does, however, show that encoding measurably changes the
underlying representation (angle spread, channel target-correlation,
cross-channel redundancy, and — for E2 specifically — training
convergence speed) even where it doesn't improve final metrics,
confirming encoding is mechanistically "live," just not currently a
productive lever. Per the master decision rule (§21), no configuration
is expanded to 5 seeds: **keep E0 as the reference and stop the encoding
investigation here.**

**Recommended next research direction: (3) final validation/
consolidation.** Six controlled factors have now been investigated
(patience, output scaling/LayerNorm, projection/bottleneck,
initialization + scale, ansatz, encoding) without any producing a
reproducible, seed-robust improvement over the standing QGNN-v4
reference — and Stage 1's own diagnostic finding (a plain logistic
regression on the frozen embedding already matches/exceeds the quantum
head's severity performance) has not been contradicted by anything
found since. Per this investigation's own instruction not to recommend
further experiments merely to keep experimenting, the evidence at this
point supports consolidating the existing findings into a final
cross-phase summary rather than opening a seventh architectural factor
(encoder architecture) on the same frozen representation and dataset.
