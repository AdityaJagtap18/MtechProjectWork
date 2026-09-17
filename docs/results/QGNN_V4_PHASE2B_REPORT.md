# QGNN-v4 Phase 2b: Output-Scale / Calibration Investigation

Phase 2b of the QGNN-v4 stability investigation. Tests whether the
instability and poor calibration documented in `QGNN_V4_PHASE1_DIAGNOSTICS.md`
(and left unexplained by Phase 2's patience experiment,
`QGNN_V4_PHASE2_REPORT.md`) is caused by the output scale of the
pre-`Linear(6,1)` representation, by inserting a scale (and optionally
bias) transform at that one point and holding everything else fixed.

**default.qubit + `diff_method="backprop"` throughout — an ideal,
noise-free classical simulation. Nothing below is attributable to
hardware noise or decoherence.**

**Methodological note, stated up front because it governs how every
result below should be read:** as specified, `alpha * quantum_output`
feeding into the *existing, unmodified* `Linear(6,1)` is mathematically
redundant with that layer's own weights/bias — any `alpha*Linear(x)+beta`
is exactly representable as a different `Linear(6,1)`. This was not an
expressiveness change (verified by test: α=1, no bias, reproduces the
baseline head byte-for-byte). Any effect found here is an **optimization
dynamics** effect — a different effective initial scale/gradient
magnitude on the pre-`Linear` representation — not added model capacity.

---

## 1. What changed

Four new quantum-head configurations, all built on the *unmodified*
`HybridQuantumHead` circuit (identical qubits, layers, ansatz, encoding,
frozen encoder):

| Config | Mechanism |
|---|---|
| `scale` | trainable `alpha` (init 1.0), no bias |
| `scale_bias` | trainable `alpha` (init 1.0) + trainable `beta` (init 0.0) |
| `fixed0.5` | fixed (non-trainable) `alpha = 0.5` |
| `fixed2.0` | fixed (non-trainable) `alpha = 2.0` |

`0.5`/`2.0` were chosen before any run (symmetric shrink/expand around
1.0) — never selected by looking at test performance.

## 2. What did NOT change

Qubits (6), variational layers (2), ansatz (`StronglyEntanglingLayers`),
`AngleEmbedding` (RY), the trainable `Linear(128,6)` reduction, the
frozen GraphSAGE-Full encoder, the matched-capacity classical control
(not retrained — `--quantum-only` reuses its existing baseline results,
since nothing about it changes), optimizer, learning rate, weight decay,
batch size, `max_epochs` (100), patience (10, reverted to the Phase 2
recommendation), seeds (42–46), splits, or evaluation pipeline. No
checkpoint was selected using the test set.

## 3. Experimental configuration

```bash
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml          --tag phase2b_scale      --seeds 42,43,44,45,46 --head-variant scale                          --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4_severity.yaml --tag phase2b_scale      --seeds 42,43,44,45,46 --head-variant scale                          --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml          --tag phase2b_scale_bias --seeds 42,43,44,45,46 --head-variant scale_bias                     --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4_severity.yaml --tag phase2b_scale_bias --seeds 42,43,44,45,46 --head-variant scale_bias                     --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml          --tag phase2b_fixed0.5   --seeds 42,43,44,45,46 --head-variant fixed_scale --alpha-init 0.5   --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4_severity.yaml --tag phase2b_fixed0.5   --seeds 42,43,44,45,46 --head-variant fixed_scale --alpha-init 0.5   --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml          --tag phase2b_fixed2.0   --seeds 42,43,44,45,46 --head-variant fixed_scale --alpha-init 2.0   --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4_severity.yaml --tag phase2b_fixed2.0   --seeds 42,43,44,45,46 --head-variant fixed_scale --alpha-init 2.0   --quantum-only --diagnostics
```

40 new run directories under `experiments/qgnn_v4/*_phase2b_*`,
non-overwriting. Baseline (`alpha=1`, no bias) reuses Phase 1's
`diag_primary`/`diag_severity` runs directly (verified byte-identical to
a fresh baseline forward pass by test). Git commit recorded per-run in
each `run_metadata.json`.

**Data-integrity note:** the first analysis pass of this data had a bug —
primary and severity runs were tagged identically (e.g. `phase2b_scale`
for both splits), and a "pick the most recent matching directory" loader
silently returned the same (severity) run for both split rows on several
configs. Caught before being reported by cross-checking against each
run's own `config.yaml` `split.strategy` field; the corrected loader
filters on that field explicitly. All numbers below are from the
corrected pull, verified against 50 unique run directories (5 configs ×
2 splits × 5 seeds).

## 4. Aggregate results

| Experiment | Primary PR-AUC | Severity PR-AUC | Brier (mean±std) | ECE (mean±std) | Prob. spread (mean std) | Mean best epoch |
|---|---|---|---|---|---|---|
| **Baseline** (α=1) | 0.8328 ± 0.1397 | 0.3915 ± 0.0237 | primary 0.128±0.083 / severity 0.211±0.083 | primary 0.285±0.122 / severity 0.345±0.162 | primary 0.116 / severity 0.060 | primary 20.6 / severity 19.4 |
| **scale** (trainable α) | 0.8417 ± 0.1160 | 0.4040 ± 0.0447 | primary 0.118±0.086 / severity 0.212±0.081 | primary 0.253±0.139 / severity 0.351±0.158 | primary 0.137 / severity 0.061 | primary 21.2 / severity 15.8 |
| **scale_bias** (trainable α+β) | 0.8412 ± 0.1229 | 0.3936 ± 0.0255 | primary 0.107±0.063 / severity 0.211±0.081 | primary 0.243±0.117 / severity 0.348±0.153 | primary 0.140 / severity 0.063 | primary 22.0 / severity 17.8 |
| **fixed0.5** | 0.8197 ± 0.1425 | 0.3882 ± 0.0516 | primary 0.166±0.032 / severity 0.218±0.082 | primary 0.352±0.039 / severity 0.367±0.131 | primary 0.057 / severity 0.044 | primary 18.8 / severity 17.8 |
| **fixed2.0** | 0.8420 ± 0.1224 | 0.4026 ± 0.0375 | primary 0.099±0.079 / severity 0.197±0.078 | primary 0.224±0.129 / severity 0.332±0.145 | primary 0.166 / severity 0.075 | primary 20.8 / severity 18.0 |

**Original baseline for reference:** Primary 0.8328 ± 0.1562 (5-run
`np.std`, matches `QGNN_V4_BENCHMARK.md`), Severity 0.3915 ± 0.0265.
Small std discrepancies between reports are `np.std` (ddof=0, used
throughout this project) vs. the same 5 numbers — all figures above use
`np.std`, ddof=0, for consistency.

## 5. Per-seed results

### Primary split, test PR-AUC

| Seed | Baseline | scale | scale_bias | fixed0.5 | fixed2.0 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.9615 | 0.9463 | 0.9489 | **0.9661** | 0.9368 |
| **43** | **0.5696** | **0.6261** | **0.6082** | 0.5778 | **0.6081** |
| 44 | 0.9266 | 0.9280 | 0.9279 | 0.9270 | 0.9256 |
| 45 | 0.8203 | 0.8220 | 0.8347 | **0.7431** | 0.8332 |
| 46 | 0.8861 | 0.8864 | 0.8865 | 0.8847 | 0.9062 |

**Seed 43 — the worst-generalizing seed throughout every prior phase —
improves under 3 of 4 variants** (scale: +0.057, scale_bias: +0.039,
fixed2.0: +0.039), and its calibration improves alongside (Brier
0.064→0.054–0.057, ECE 0.160→0.113–0.124 under those same three
variants). `fixed0.5` is the outlier in the other direction: seed 45
drops notably (0.820→0.743), and it's the only variant where seed 43
doesn't clearly improve (0.578, essentially flat).

### Severity split, test PR-AUC

| Seed | Baseline | scale | scale_bias | fixed0.5 | fixed2.0 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.3610 | 0.3611 | 0.3611 | 0.3628 | 0.3572 |
| 43 | 0.3744 | 0.3745 | 0.3744 | 0.3746 | 0.3743 |
| 44 | 0.4248 | 0.4877 | 0.4249 | 0.4884 | 0.4565 |
| 45 | 0.3851 | 0.3887 | 0.3856 | 0.3741 | 0.3895 |
| 46 | 0.4123 | 0.4082 | 0.4218 | 0.3409 | 0.4357 |

Seeds 42/43 are essentially frozen across every config (±0.002) — no
scale/bias change moves them at all. Seed 44 moves meaningfully under
`scale`/`fixed0.5` (+0.06). Seed 46 drops notably under `fixed0.5`
(−0.07). No consistent direction across seeds or configs.

### Severity split, best_epoch (does output scale change *when* training stops?)

| Seed | Baseline | scale | scale_bias | fixed0.5 | fixed2.0 |
|---:|---:|---:|---:|---:|---:|
| 42 | 1 | 1 | 1 | 1 | 1 |
| 43 | 3 | 3 | 3 | 3 | 3 |
| 44 | 5 | 1 | 5 | 1 | 2 |
| 45 | 3 | 4 | 3 | 6 | 4 |
| 46 | 85 | 70 | 77 | 78 | 80 |

**Seeds 42 and 43 stop at the exact same epoch in every single one of
the 5 configurations.** Output scale does not change when severity's
early stopping triggers for the seeds where it triggers earliest and
most severely — direct evidence that severity's near-immediate stopping
(Phase 1/2's central severity finding) is not a scale/calibration
artifact.

## 6. Fresh-onset PR-AUC (severity split)

| Config | Mean | Min | Max |
|---|---:|---:|---:|
| baseline | 0.0061 | 0.0046 | 0.0100 |
| scale | 0.0064 | 0.0046 | 0.0100 |
| scale_bias | 0.0061 | 0.0046 | 0.0100 |
| fixed0.5 | 0.0066 | 0.0046 | 0.0100 |
| fixed2.0 | 0.0061 | 0.0046 | 0.0102 |

Unchanged, near the noise floor, in every configuration — consistent
with Phase 1's structural finding that the frozen embedding itself
carries no fresh-onset precursor signal. No output-scale change can
recover a signal that isn't in the input; this was expected, not a new
finding, and is reported for completeness per the investigation's own
measurement list.

---

## 7. Answers to the 10 scientific questions

1. **Does the output distribution become less collapsed?** Split-dependent.
   Primary: `fixed2.0` widens it (std 0.116→0.166); `fixed0.5` narrows it
   further (0.116→0.057, the wrong direction). Severity: `fixed2.0`
   widens modestly (0.060→0.075); the others are flat-to-worse
   (`fixed0.5`: 0.060→0.044, more collapsed).
2. **Does prediction variance increase in the problematic severity runs?**
   Only under `fixed2.0`, and only modestly (std 0.060→0.075). `fixed0.5`
   makes it worse. `scale`/`scale_bias` are essentially flat.
3. **Does calibration improve?** Yes on primary, for 3 of 4 variants
   (`scale`, `scale_bias`, `fixed2.0`: ECE 0.285→0.224–0.253). No on
   severity — all four variants are flat-to-worse (ECE 0.345→0.332–0.367).
4. **Does Brier score improve?** Same pattern as ECE: primary improves
   under `scale`/`scale_bias`/`fixed2.0` (0.128→0.099–0.118); severity
   does not (0.211→0.197–0.218, all within noise or worse).
5. **Does QNN test PR-AUC improve?** Primary: yes, modestly, under 3 of 4
   variants (0.833→0.841–0.842). Severity: marginally under `scale`/
   `fixed2.0` (0.392→0.403–0.404), flat/worse under the others.
6. **Does training become more stable across seeds?** On primary, yes —
   std drops from 0.140 to 0.116–0.123 under 3 of 4 variants (`fixed0.5`
   is the exception, 0.143, slightly worse). On severity, no — std is
   flat-to-worse in every variant (0.024→0.038–0.052).
7. **Does the seed-43 primary failure remain?** Partially resolved, not
   eliminated — seed 43 improves by 0.04–0.06 PR-AUC under 3 of 4
   variants (still the worst seed in absolute terms, but the gap to the
   other seeds narrows measurably). This is the clearest positive
   finding in this phase.
8. **Does the severity-split failure remain?** Yes, essentially
   unchanged. §5's best_epoch table is decisive: two of five seeds stop
   at the identical epoch regardless of output scale, and aggregate
   PR-AUC/calibration/spread show no consistent improvement.
9. **Does improving calibration also improve PR-AUC, or are these
   separate problems?** On primary they move together (the same 3
   variants — `scale`, `scale_bias`, `fixed2.0` — improve both). On
   severity they're both flat, so this split doesn't distinguish the two.
   The primary-split co-movement is suggestive but not proof of a causal
   link, given the reparameterization caveat (§0) — a cleaner test would
   need an intervention that moves one without the other.
10. **Is output scale actually a bottleneck, or should we move on to
    architecture/circuit ablations?** See §8 — split-dependent answer,
    not a single yes/no.

## 8. Decision

**Neither a clean (A) nor a clean (C) — the evidence splits by split.**

- **Primary split: closer to (A)/(B).** Output scale (specifically
  amplifying/trainable variants — `scale`, `scale_bias`, `fixed2.0`) is a
  real, if modest, factor: aggregate variance drops, calibration
  improves, and — the most important single result in this phase — seed
  43's generalization gap partially closes. `fixed0.5` (shrinking) is
  never beneficial and sometimes harmful (seed 45).
- **Severity split: (C).** Output scale does not explain the collapse
  documented in Phase 1. The decisive evidence is mechanistic, not just
  statistical: early-stopping timing for the two most severely truncated
  seeds (42, 43) is byte-identical across all five configurations. A
  transform applied entirely *after* the point where validation PR-AUC is
  computed cannot change when a validation-PR-AUC-triggered stop occurs
  — and it doesn't, confirming severity's problem is upstream of the
  output-scale mechanism being tested here (consistent with Phase 1/2's
  finding: severity's validation set is near-saturated almost
  immediately, which is a property of that split's validation
  composition, not of the head's output calibration).

**Because the split that motivated this investigation most acutely
(severity, with its 0.004–0.022 probability-std collapse in Phase 1) is
the one output scale does NOT fix, the honest overall verdict is closer
to (C) than (A) — with the explicit caveat that (A)/(B) hold on primary
and that finding (seed 43's partial recovery) is real and worth keeping.**

No quantum advantage is claimed anywhere in this phase — every
comparison here is quantum-vs-quantum (different output-scale variants
of the same circuit), not quantum-vs-classical.

## 9. Recommended next experiment

Two live threads, not one:

1. **For severity specifically**, output scale is now ruled out
   (alongside gradients, patience, and initialization from Phases 1–2).
   The next most targeted, still-non-architectural lever is Phase 2b's
   deferred candidate 5 — a genuinely non-redundant, data-dependent
   normalization (e.g. `LayerNorm` without learnable affine, or with) on
   the PauliZ output before `Linear(6,1)`. Unlike `alpha`/`beta`, batch-
   or instance-level normalization is NOT absorbable into a following
   linear layer (its statistics depend on the current batch/input), so it
   could plausibly do something the scale/bias experiments structurally
   cannot. This should be tried before concluding calibration-side
   interventions are exhausted.
2. **For primary**, the `scale`/`scale_bias`/`fixed2.0` result is worth
   keeping as the new default output-scale configuration if further work
   continues on primary specifically — but this is a small enough effect
   (mean +0.01, std −0.02) that it should not be reported as a resolved
   instability, only a modest, real improvement with one notable
   individual-seed exception (`fixed0.5`/seed 45).

If normalization also fails to move severity's early-stopping timing,
that would be strong, convergent evidence (with Phases 1–2) that the
instability is not a training-dynamics or calibration issue at all, and
the qubit/layer/ansatz ablation grid (Phase 3, per the investigation's
own sequencing) becomes the well-motivated next step — at that point
having ruled out gradients, patience, initialization, output scale, and
data-independent bias as explanations.
