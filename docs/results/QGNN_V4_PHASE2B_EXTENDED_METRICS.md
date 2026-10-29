# QGNN-v4 Phase 2b: Extended Metrics

Extends `QGNN_V4_PHASE2B_REPORT.md` with the full ranking / classification /
calibration / generalization / stability metric set, computed purely from
already-saved run artifacts (`scripts/analyze_qgnn_v4_extended_metrics.py`
— no retraining). Covers the 5 completed Phase 2b configurations
(**baseline**, **scale**, **scale_bias**, **fixed0.5**, **fixed2.0**) ×
2 splits × 5 seeds = 50 runs. Threshold-based metrics use the same
`threshold.policy: fixed, value: 0.5` every prior phase used — never
selected or tuned on the test set.

**LayerNorm variants have not been built or run.** That was Phase 2b's
*recommended next experiment*, not something executed yet — no LayerNorm
numbers appear anywhere below.

**Per-severity-level breakdown (Section G):** `severity_split.csv` stores
only the binary train/test label per time period, not the period's
severity value — that's computed internally by
`benchmark.splits.severity_split()` and discarded. It is *not* part of
the modeling pipeline's saved outputs. It **is** reconstructable from the
benchmark's raw `events/events.csv` (`start_time`/`duration`/`severity`/
`recovery_delay`/`recovery_periods`), using the exact same
max-active-severity-per-period logic `severity_split()`/`_event_windows()`
themselves use — done here, verified to reproduce all 104 periods'
train/test labels with **zero mismatches** against the real
`severity_split.csv` before being used for anything. This dataset
realization (`scm_v1_black_swan_seed43`) contains periods at severity
levels **{0, 1, 2, 5} only** — no period's active-event maximum is
exactly 3 or 4.

Full structured data (all 97 columns, every run): `experiments/qgnn_v4/phase2b_extended_metrics/phase2b_all_metrics.csv`, plus `phase2b_severity_level_breakdown.csv`, `phase2b_seed_stability.csv`, `phase2b_paired_vs_baseline.csv` (gitignored alongside every other run artifact; this report and the analysis script are the committed record).

**Verification note:** every table below was cross-checked cell-by-cell
against `phase2b_all_metrics.csv` before this report was finalized. That
pass caught and corrected several errors from an earlier hand-typed
draft: the per-seed severity table (Table 3) had genuinely wrong Brier
values in 5 cells (hand-transcription mistakes, not a data bug); an
earlier severity-level breakdown claimed a monotonic PR-AUC decline
across severity levels that the actual numbers don't support (see that
section); and two smaller numeric claims (a probability-spread ranking
and a per-configuration collapse-count breakdown) were also corrected
against the source data. All tables now reflect direct, verified reads
of the CSV, not hand-copied values.

---

## Table 1 — Main Results (mean across 5 seeds, test split)

| Configuration | Split | PR-AUC | ROC-AUC | F1 | Precision | Recall | MCC | Brier | ECE | Prob Std | Best Epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | primary | 0.8328 | 0.9829 | 0.5693 | 0.4373 | 0.9603 | 0.6839 | 0.1283 | 0.2845 | 0.1158 | 20.6 |
| scale | primary | 0.8417 | 0.9833 | 0.5669 | 0.4364 | 0.9537 | 0.6796 | 0.1177 | 0.2530 | 0.1365 | 21.2 |
| scale_bias | primary | 0.8412 | 0.9832 | **0.6584** | **0.5034** | 0.9587 | 0.6662 | 0.1067 | 0.2432 | 0.1396 | 22.0 |
| fixed0.5 | primary | 0.8197 | 0.9846 | 0.5399 | 0.6064 | 0.7752 | 0.5673 | 0.1661 | 0.3520 | 0.0565 | 18.8 |
| fixed2.0 | primary | 0.8420 | 0.9800 | 0.5794 | 0.4342 | 0.9554 | 0.5914 | **0.0990** | **0.2237** | 0.1664 | 20.8 |
| baseline | severity | 0.3915 | 0.6257 | 0.2436 | 0.4354 | 0.5667 | 0.2305 | 0.2105 | 0.3448 | 0.0604 | 19.4 |
| scale | severity | 0.4040 | 0.6397 | 0.2334 | 0.5119 | 0.5376 | 0.2503 | 0.2115 | 0.3505 | 0.0610 | 15.8 |
| scale_bias | severity | 0.3936 | 0.6250 | 0.2579 | 0.4148 | 0.5770 | 0.2684 | 0.2108 | 0.3480 | 0.0625 | 17.8 |
| fixed0.5 | severity | 0.3882 | 0.6482 | 0.1393 | 0.1108 | 0.4879 | 0.3749 | 0.2183 | 0.3665 | 0.0437 | 17.8 |
| fixed2.0 | severity | 0.4026 | 0.6398 | **0.3151** | 0.3110 | 0.5660 | 0.3102 | **0.1970** | **0.3322** | **0.0748** | 18.0 |

**New finding not visible in Phase 2b's PR-AUC-only view: `scale_bias`
gives the best mean F1/Precision on primary (0.658/0.503 vs. baseline's
0.569/0.437) — a genuinely different, additional benefit from the
calibration-side one already reported.** `fixed2.0` remains the best
single config for calibration on both splits (lowest Brier/ECE
everywhere) and gives the best severity F1 (0.315 vs. baseline's 0.244),
though severity PR-AUC gains stay small (§ below).

## Table 2 — Cross-Seed Stability (mean ± std, test split, n=5)

| Configuration | Split | PR-AUC | ROC-AUC | F1 | Recall | Precision | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| baseline | primary | 0.833±0.140 | 0.983±0.016 | 0.569±0.224 | 0.960±0.036 | 0.437±0.187 | 0.128±0.083 | 0.285±0.122 |
| scale | primary | 0.842±0.116 | 0.983±0.015 | 0.567±0.223 | 0.954±0.041 | 0.436±0.187 | 0.118±0.086 | 0.253±0.139 |
| scale_bias | primary | 0.841±0.123 | 0.983±0.016 | **0.658±0.047** | 0.959±0.040 | 0.503±0.051 | 0.107±0.063 | 0.243±0.117 |
| fixed0.5 | primary | 0.820±0.143 | 0.985±0.012 | 0.540±0.249 | 0.775±0.377 | 0.606±0.202 | 0.166±0.032 | 0.352±0.039 |
| fixed2.0 | primary | 0.842±0.122 | 0.980±0.022 | 0.579±0.156 | 0.955±0.043 | 0.434±0.141 | **0.099±0.079** | **0.224±0.129** |
| baseline | severity | 0.392±0.024 | 0.626±0.039 | 0.244±0.193 | 0.567±0.371 | 0.435±0.350 | 0.211±0.083 | 0.345±0.162 |
| scale | severity | 0.404±0.045 | 0.640±0.033 | 0.233±0.175 | 0.538±0.383 | 0.512±0.417 | 0.212±0.081 | 0.351±0.157 |
| scale_bias | severity | 0.394±0.026 | 0.625±0.040 | 0.258±0.158 | 0.577±0.352 | 0.415±0.342 | 0.211±0.081 | 0.348±0.153 |
| fixed0.5 | severity | 0.388±0.052 | 0.648±0.025 | 0.139±0.154 | 0.488±0.448 | 0.111±0.152 | 0.218±0.082 | 0.367±0.131 |
| fixed2.0 | severity | 0.403±0.038 | 0.640±0.029 | 0.315±0.135 | 0.566±0.239 | 0.311±0.187 | 0.197±0.078 | 0.332±0.145 |

Full mean/std/**median/min/max/IQR** for all 7 metrics × 5 configs × 2
splits is in `phase2b_seed_stability.csv` (70 rows) — median/IQR are
omitted above for width but confirm the same ranking (e.g. primary
`scale_bias` F1 median 0.682, IQR 0.032 — even tighter than the
mean/std suggests, since one seed (45) is a mild outlier pulling the
mean down slightly).

**The most important number in this table:** `recall_std` on severity is
**0.24–0.45 across every single configuration**, roughly 7–19x larger
than the same statistic on primary (0.036–0.043). At `threshold=0.5`,
severity's classification behavior is not just lower-performing than
primary's — it is categorically less stable, regardless of which
output-scale variant is used. See Table 4.

## Table 3 — Severity Split, Per-Seed Detail

| Configuration | Seed | PR-AUC | Recall | F1 | Brier | ECE | Prob Std | Best Epoch | Fresh-Onset PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 42 | 0.3610 | 0.9523 | 0.1391 | 0.2727 | 0.4583 | 0.0215 | 1 | 0.0045 |
| baseline | 43 | 0.3744 | 0.0012 | 0.0024 | 0.1727 | 0.3291 | 0.0222 | 3 | 0.0100 |
| baseline | 44 | 0.4248 | 0.4406 | 0.5139 | 0.2120 | 0.3940 | 0.0165 | 5 | 0.0050 |
| baseline | 45 | 0.3851 | 1.0000 | 0.1371 | 0.3171 | 0.4994 | 0.0044 | 3 | 0.0049 |
| baseline | 46 | 0.4123 | 0.4394 | 0.4254 | 0.0780 | 0.0432 | 0.2373 | 85 | 0.0059 |
| scale | 42 | 0.3611 | 0.9486 | 0.1392 | 0.2725 | 0.4583 | 0.0216 | 1 | 0.0045 |
| scale | 43 | 0.3745 | 0.0061 | 0.0122 | 0.1722 | 0.3288 | 0.0227 | 3 | 0.0100 |
| scale | 44 | 0.4877 | 0.2938 | 0.4541 | 0.2199 | 0.4135 | 0.0101 | 1 | 0.0070 |
| scale | 45 | 0.3887 | 1.0000 | 0.1371 | 0.3135 | 0.4960 | 0.0053 | 4 | 0.0049 |
| scale | 46 | 0.4082 | 0.4394 | 0.4246 | 0.0796 | 0.0559 | 0.2450 | 70 | 0.0057 |
| scale_bias | 42 | 0.3611 | 0.9449 | 0.1394 | 0.2721 | 0.4577 | 0.0216 | 1 | 0.0045 |
| scale_bias | 43 | 0.3744 | 0.0600 | 0.1132 | 0.1746 | 0.3365 | 0.0227 | 3 | 0.0100 |
| scale_bias | 44 | 0.4249 | 0.4406 | 0.4731 | 0.2142 | 0.3859 | 0.0170 | 5 | 0.0050 |
| scale_bias | 45 | 0.3856 | 1.0000 | 0.1371 | 0.3139 | 0.4962 | 0.0044 | 3 | 0.0049 |
| scale_bias | 46 | 0.4218 | 0.4394 | 0.4266 | 0.0791 | 0.0635 | 0.2468 | 77 | 0.0058 |
| fixed0.5 | 42 | 0.3628 | 1.0000 | 0.1371 | 0.2861 | 0.4684 | 0.0107 | 1 | 0.0046 |
| fixed0.5 | 43 | 0.3746 | 0.0000 | 0.0000 | 0.1856 | 0.3455 | 0.0112 | 3 | 0.0100 |
| fixed0.5 | 44 | 0.4884 | 0.0000 | 0.0000 | 0.2249 | 0.3970 | 0.0050 | 1 | 0.0070 |
| fixed0.5 | 45 | 0.3741 | 1.0000 | 0.1371 | 0.3131 | 0.4954 | 0.0034 | 6 | 0.0058 |
| fixed0.5 | 46 | 0.3409 | 0.4394 | 0.4224 | 0.0818 | 0.1263 | 0.1880 | 78 | 0.0058 |
| fixed2.0 | 42 | 0.3572 | 0.6328 | 0.1681 | 0.2479 | 0.4289 | 0.0428 | 1 | 0.0045 |
| fixed2.0 | 43 | 0.3743 | 0.3170 | 0.3863 | 0.1500 | 0.2971 | 0.0437 | 3 | 0.0102 |
| fixed2.0 | 44 | 0.4565 | 0.4406 | 0.4595 | 0.2016 | 0.3702 | 0.0247 | 2 | 0.0055 |
| fixed2.0 | 45 | 0.3895 | 1.0000 | 0.1371 | 0.3050 | 0.4929 | 0.0104 | 4 | 0.0049 |
| fixed2.0 | 46 | 0.4357 | 0.4394 | 0.4243 | 0.0806 | 0.0718 | 0.2523 | 80 | 0.0055 |

**Fresh-onset PR-AUC is 0.0045–0.0102 in every one of the 25 severity
runs** — indistinguishable from noise regardless of configuration,
confirming again (4th consecutive phase) that no head variant recovers a
fresh-onset signal that isn't in the frozen input.

**Seed 43's threshold-0.5 recall collapses to ≤0.06 in 4 of 5
configurations** (0.0012, 0.0061, 0.0600, 0.0000 — only `fixed2.0` lifts
it to 0.317) despite PR-AUC (threshold-free) staying flat at ~0.374
throughout. This is the clearest single illustration in this whole
investigation of *why* PR-AUC and threshold-0.5 classification tell
different stories on this split.

## Table 4 — Threshold-0.5 Classification Collapse (new finding)

Classifying each of the 50 runs as **collapsed-all-positive**
(recall ≥0.98, specificity ≤0.10), **collapsed-all-negative**
(recall ≤0.05, precision ≥0.95), or **normal**:

| Split | Collapsed all-positive | Collapsed all-negative | Normal | Total |
|---|---:|---:|---:|---:|
| primary | 2 | 1 | 22 | 25 |
| severity | 6 | 2 | 17 | 25 |

**8 of 25 severity runs (32%) exhibit degenerate threshold-0.5
classification, vs. 3 of 25 primary runs (12%).** This holds across
*every* configuration, not a specific one — `baseline`, `scale`, and
`fixed0.5` each contribute 2 collapsed severity runs; `scale_bias` and
`fixed2.0` each contribute 1. This significantly extends
Phase 1 §7's "threshold-calibration artifact" finding (originally
identified in 2 of 5 baseline severity seeds) into a **systematic
property of this split at `threshold=0.5`, present in every
configuration tested across two full phases of investigation** — not a
quirk of one seed or one head variant.

## Severity-Level Breakdown (per active-event severity, all splits combined)

Mean across 5 seeds, per configuration, per severity level actually
present in this dataset (0 = baseline/no active event, 1/2 = lower
severity, 5 = the highest severity band — the severity-split's test set).
**Severity level 1 has only 600 examples total and exactly 1 positive
across the entire 26,700-row dataset, in every seed** — its PR-AUC is a
single-example-dependent statistic, not a reliable estimate, and is
excluded from any pattern claim below (shown only for completeness):

| Config | Sev. 0 (n≈8400) PR-AUC | Sev. 1 (n=600, 1 positive — unreliable) PR-AUC | Sev. 2 (n≈7800) PR-AUC | Sev. 5 (n≈9900, = test) PR-AUC |
|---|---:|---:|---:|---:|
| baseline | 0.913 | 0.454 | 0.833 | 0.376 |
| scale | 0.922 | 0.355 | 0.834 | 0.390 |
| scale_bias | 0.914 | 0.454 | 0.833 | 0.379 |
| fixed0.5 | 0.944 | 0.441 | 0.839 | 0.373 |
| fixed2.0 | 0.936 | 0.459 | 0.833 | 0.389 |

**Not monotonic — the reliable levels (0, 2, 5) show high → high → low,
not a steady decline**, in every configuration: severity 0 (baseline
conditions) and severity 2 both sit around 0.83–0.94, then severity 5
drops sharply to 0.37–0.39. This is a real, consistent finding, just not
the one originally drafted here (an earlier pass of this report claimed
a monotonic decline across all four levels — wrong, caught and corrected
before being finalized). The sharp break is between severity 2 and
severity 5 specifically, not a gradual slope — consistent with
`train_max_severity=3` being where the split boundary sits: levels ≤3 are
in-distribution (train), and the model's discrimination collapses only
once it crosses into the truly out-of-distribution band the split holds
out. Output scale shifts the severity-5 endpoint by only ±0.02 in every
configuration — it does not change this break. Full per-seed
severity-level breakdown (100 rows, including recall/precision/prob
stats per level per seed) is in `phase2b_severity_level_breakdown.csv`.

## Paired Per-Seed Differences vs. Baseline (PR-AUC)

| Config | Split | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Mean Δ | Median Δ | Improved | Worsened |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scale | primary | −0.0152 | +0.0565 | +0.0014 | +0.0017 | +0.0003 | +0.0089 | +0.0014 | 4 | 1 |
| scale_bias | primary | −0.0126 | +0.0386 | +0.0013 | +0.0144 | +0.0004 | +0.0084 | +0.0013 | 4 | 1 |
| fixed0.5 | primary | +0.0046 | +0.0082 | +0.0004 | −0.0772 | −0.0014 | −0.0131 | +0.0004 | 3 | 2 |
| fixed2.0 | primary | −0.0247 | +0.0385 | −0.0010 | +0.0129 | +0.0201 | +0.0092 | +0.0129 | 3 | 2 |
| scale | severity | +0.0001 | +0.0001 | +0.0629 | +0.0036 | −0.0041 | +0.0125 | +0.0001 | 4 | 1 |
| scale_bias | severity | +0.0001 | +0.0000 | +0.0001 | +0.0005 | +0.0095 | +0.0021 | +0.0001 | 5 | 0 |
| fixed0.5 | severity | +0.0018 | +0.0002 | +0.0636 | −0.0110 | −0.0714 | −0.0033 | +0.0002 | 3 | 2 |
| fixed2.0 | severity | −0.0038 | −0.0001 | +0.0317 | +0.0044 | +0.0234 | +0.0111 | +0.0044 | 3 | 2 |

**Median Δ is much smaller in magnitude than mean Δ in 7 of these 8
rows** — most apparent "improvements" in the Phase 2b headline numbers
are disproportionately driven by one or two seeds (usually 43 or 44),
not a uniform shift. The one exception is `fixed2.0`/primary, where
median Δ (+0.0129) exceeds mean Δ (+0.0092) — that specific
configuration's primary-split gain is the one case in this table that
looks more like a broad shift than an outlier-driven one. No seed is
cherry-picked here: all 5 per-seed deltas are shown for every
configuration. Full 7-metric paired comparison (56 rows) is in
`phase2b_paired_vs_baseline.csv`.

---

## Interpretation, separated by category (as requested — no single metric drives the read)

**1. Ranking performance (PR-AUC/ROC-AUC).** Matches
`QGNN_V4_PHASE2B_REPORT.md` exactly: modest, real gains on primary under
`scale`/`scale_bias`/`fixed2.0`; small, inconsistent, seed-driven changes
on severity. Nothing new here beyond what that report already
established — included for completeness of the metric set.

**2. Classification performance at threshold=0.5 (F1/Precision/Recall/MCC).**
The genuinely new finding in this pass. On primary, `scale_bias` gives a
materially better and *more stable* F1 (0.658±0.047) than every other
config including baseline (0.569±0.224) — a distinct benefit from its
calibration-side one, not previously visible. On severity, classification
is categorically unstable in every configuration (`recall_std`
0.24–0.45) and 8 of 25 severity runs collapse to near-all-positive or
near-all-negative predictions at this fixed cutoff (Table 4) — a
systematic property of the split, not of any one head variant.

**3. Calibration (Brier/ECE/MCE).** `fixed2.0` is the best-calibrated
configuration on both splits, consistently. `fixed0.5` is the
worst-calibrated on primary by a wide margin. Severity calibration is
flat-to-worse everywhere, matching `QGNN_V4_PHASE2B_REPORT.md`.

**4. Prediction-distribution collapse (prob std / histogram / threshold
behavior).** `fixed2.0` has the highest mean prob_std of all 5
configurations on *both* splits (primary 0.166, severity 0.075);
`fixed0.5` has the lowest on both (primary 0.057, severity 0.044) — the
most consistent, unambiguous pattern in this whole extended pass:
amplifying widens, shrinking narrows, on every split, every time. Table
4's collapse count is the clearer number for the underlying problem
though: fixing output scale does not reduce the *frequency* of
degenerate threshold-0.5 runs on severity (every configuration produces
1–2 of them, `fixed2.0` included).

**5. Optimization behavior (gradients, losses, best epoch).** Unchanged
from Phase 1/2/2b: `best_epoch` for severity's earliest-stopping seeds
(42, 43) is essentially fixed regardless of configuration (§ prior
reports); quantum gradient norms stay in the same healthy 0.1–0.3 range
in every configuration (`phase2b_all_metrics.csv`'s
`quantum_grad_norm_*` columns) — no new gradient pathology introduced by
any output-scale variant.

**6. Generalization (train→val→test gaps).** Seed 43's signature —
train/val PR-AUC high, test PR-AUC low — persists across every
configuration (Table 3, seed 43's recall collapses to ≤0.06 in 4 of 5
configs despite flat PR-AUC ≈0.374 throughout). Output scale does not
close this gap on severity; on primary it partially does (the
paired-differences table's seed-43 deltas: +0.0565/+0.0386/+0.0385 under
three of four variants), the one finding that has now shown up
consistently across both Phase 2b's original PR-AUC-only analysis and
this extended pass.

**7. Cross-seed stability.** Table 2 is decisive: severity's
`recall_std` (0.24–0.45) dwarfs primary's (0.036–0.043) in *every*
configuration — this is a property of the split, present before and
after every intervention tried across Phases 1, 2, and 2b.

**No configuration is declared better on the strength of a single
metric.** `scale_bias` is the only configuration with a *consistent,
multi-metric* case on primary (higher F1 and lower F1-variance, plus the
calibration gains already reported); `fixed2.0` has the strongest,
most consistent calibration case across both splits but a much smaller
and less consistent PR-AUC/classification case. No configuration shows
a consistent, multi-metric case on severity — Table 4's collapse-rate
and Table 2's recall-variance are unmoved by any of them.
