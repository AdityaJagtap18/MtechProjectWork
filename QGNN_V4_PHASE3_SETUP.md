# QGNN-v4 Phase 3: Quantum Architecture Ablation — Setup

Establishes the frozen protocol, the Phase-3 reference configuration,
and the full experimental matrix before any new training runs. Per
Phase 3's own instruction, the current 6-qubit/2-layer
`StronglyEntanglingLayers` + `LayerNorm(no-affine)` configuration is
**shared as the reference rather than duplicated** — it already exists
in full (5 seeds × 2 splits) from Phase 2c. **7 new configurations, not
10, actually need to run.**

**Explicit control choice, not a finding:** LayerNorm-no-affine is used
as the Phase-3 default output stage because Phase 2c found it gives the
best balance of stability/primary-performance/calibration among the
LayerNorm variants tested. This is a documented experimental control for
Phase 3, not evidence that LayerNorm-no-affine is the optimal choice in
any absolute sense — Phase 2c's own report already states LayerNorm
trades primary PR-AUC mean for stability and calibration, a real cost,
not a free win.

---

## 1. What's new (code, this phase)

`src/scm_dataset/modeling/quantum/circuit.py` gained two hand-built
ansätze (additive — `strongly_entangling`/`basic_entangler` unchanged):

- **`hardware_efficient_ring`** (Ansatz 2): `RY` then `RZ` per qubit per
  layer (2 trainable params/qubit/layer) + a CNOT **ring** (wrap-around
  included).
- **`reduced_entanglement`** (Ansatz 3): `RY` only per qubit per layer
  (1 param/qubit/layer) + a CNOT **chain** (linear, no wrap-around) —
  structurally identical to `qgnn.py`'s own v1–v3 ansatz, though built
  independently here on the frozen-embedding-direct pipeline, not a
  reuse of that module.

`HybridQuantumHeadLayerNorm` already supported arbitrary `n_qubits`/
`n_layers`/`ansatz` (Phase 2c never hardcoded these) — verified by 15 new
tests that qubit counts {4,6,8}, layer counts {1,2,3,4}, and all four
ansatz choices build and run correctly, and that the two new ansätze's
parameter counts match their formulas exactly (`n_layers×n_qubits×2` for
the ring, `n_layers×n_qubits×1` for the chain). 283 tests passing, zero
regressions.

**A real methodological confound, worth stating before any results come
in:** the qubit-count ablation necessarily also changes the *classical*
`Linear(128, n_qubits)` reduction layer's size, since both scale with
`n_qubits` together (Table 1 below shows this: going from 4→8 qubits
changes total head parameters by 544, of which only 24 is quantum). Any
qubit-count effect on performance is therefore a mix of "more/fewer
qubits" and "a bigger/smaller classical bottleneck," not a pure quantum
capacity effect — Section 7's capacity-vs-architecture split needs to
account for this explicitly when Phase 3's results are analyzed, not
just report the two numbers side by side.

## 2. Table 1 — Architecture (all 10 configurations)

| Architecture | Qubits | Layers | Ansatz | Quantum Params | Head Params | Status |
|---|---:|---:|---|---:|---:|---|
| Reference (shared) | 6 | 2 | strongly_entangling | 36 | 817 | **Already have** (Phase 2c) |
| Qubit-A | 4 | 2 | strongly_entangling | 24 | 545 | New |
| Qubit-C | 8 | 2 | strongly_entangling | 48 | 1089 | New |
| Depth-A | 6 | 1 | strongly_entangling | 18 | 799 | New |
| Depth-C | 6 | 3 | strongly_entangling | 54 | 835 | New |
| Depth-D | 6 | 4 | strongly_entangling | 72 | 853 | New |
| Ansatz-2 | 6 | 2 | hardware_efficient_ring | 24 | 805 | New |
| Ansatz-3 | 6 | 2 | reduced_entanglement | 12 | 793 | New |

(Depth-B = 6q/2L/strongly_entangling is the same row as Reference;
Ansatz-1 = 6q/2L/strongly_entangling is also the same row — both shared,
not duplicated, per instruction.)

Classical GNN reference (extracted, not retrained, `QGNN_V4_CLASSICAL_BASELINE.md`):
Primary PR-AUC 0.807±0.066, Severity PR-AUC 0.449±0.013 — every Phase 3
configuration will be compared against this same fixed reference, never
re-extracted per architecture.

## 3. Phase-3 Baseline — Verified

The reference configuration already exists (Phase 2c's
`layernorm_noaffine`, 5 seeds × 2 splits, `--diagnostics` throughout):

| Split | PR-AUC (mean ± std) |
|---|---|
| Primary | 0.8112 ± 0.0737 |
| Severity | 0.3976 ± 0.0571 |

Matches `QGNN_V4_PHASE2C_REPORT.md`/`QGNN_V4_PHASE2D_REPORT.md` exactly —
confirmed directly from `phase2b_all_metrics.csv`, no new run needed.

## 4. Frozen Protocol (unchanged from every prior phase)

Frozen GraphSAGE encoder, dataset, splits, preprocessing, optimizer
(Adam), learning rate (0.001), weight decay (0.0001), batch size,
`max_epochs=100`, `patience=10`, class weighting (balanced), threshold
(`fixed, value=0.5`), seeds `42,43,44,45,46`, both splits, `--diagnostics`
on every run, `--quantum-only` (classical control not retrained per
architecture, per instruction). No per-architecture tuning of any kind.

## 5. Experimental Matrix — 7 New Configurations × 2 Splits × 5 Seeds = 70 Runs

```bash
# ── Ablation A: Qubit count ──────────────────────────────────────────
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml          --tag phase3_q4 --seeds 42,43,44,45,46 --head-variant layernorm_noaffine --n-qubits 4 --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4_severity.yaml --tag phase3_q4 --seeds 42,43,44,45,46 --head-variant layernorm_noaffine --n-qubits 4 --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml          --tag phase3_q8 --seeds 42,43,44,45,46 --head-variant layernorm_noaffine --n-qubits 8 --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4_severity.yaml --tag phase3_q8 --seeds 42,43,44,45,46 --head-variant layernorm_noaffine --n-qubits 8 --quantum-only --diagnostics

# ── Ablation B: Variational depth ────────────────────────────────────
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml          --tag phase3_d1 --seeds 42,43,44,45,46 --head-variant layernorm_noaffine --n-layers 1 --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4_severity.yaml --tag phase3_d1 --seeds 42,43,44,45,46 --head-variant layernorm_noaffine --n-layers 1 --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml          --tag phase3_d3 --seeds 42,43,44,45,46 --head-variant layernorm_noaffine --n-layers 3 --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4_severity.yaml --tag phase3_d3 --seeds 42,43,44,45,46 --head-variant layernorm_noaffine --n-layers 3 --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml          --tag phase3_d4 --seeds 42,43,44,45,46 --head-variant layernorm_noaffine --n-layers 4 --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4_severity.yaml --tag phase3_d4 --seeds 42,43,44,45,46 --head-variant layernorm_noaffine --n-layers 4 --quantum-only --diagnostics

# ── Ablation C: Ansatz ───────────────────────────────────────────────
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml          --tag phase3_ansatz_ring --seeds 42,43,44,45,46 --head-variant layernorm_noaffine --ansatz hardware_efficient_ring --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4_severity.yaml --tag phase3_ansatz_ring --seeds 42,43,44,45,46 --head-variant layernorm_noaffine --ansatz hardware_efficient_ring --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml          --tag phase3_ansatz_reduced --seeds 42,43,44,45,46 --head-variant layernorm_noaffine --ansatz reduced_entanglement --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4_severity.yaml --tag phase3_ansatz_reduced --seeds 42,43,44,45,46 --head-variant layernorm_noaffine --ansatz reduced_entanglement --quantum-only --diagnostics
```

14 commands, 70 runs total. Rough timing estimate from Phase 2c's own
pace (21 runs in ~5.5 minutes observed): **~20–30 minutes total**, though
the 8-qubit and depth-4 configurations simulate a larger circuit and may
run somewhat slower per epoch — not independently verified, an estimate
only, reported as such rather than a promise.

No seed reduction, no per-architecture tuning, no threshold selection —
matches the frozen protocol exactly. If you'd rather run these in
smaller batches (e.g. one ablation group at a time) given the count,
that's a reasonable way to pace it; report back after however many
you've completed and I'll analyze what's in rather than waiting for all
70.

## 6. What happens after the runs land

For each new configuration I'll extend the same verified pipeline
already built:

- `analyze_qgnn_v4_extended_metrics.py` (Tables 2–5: full ranking/
  classification/calibration/stability metrics, per-seed and aggregate)
- `inspect_seed45_quantum_features.py`'s approach, generalized (Table 6,
  Section 9: per-channel correlation-with-label for every architecture,
  specifically checking whether seed 45 develops another dominant
  channel under each new qubit count/depth/ansatz)
- Capacity-vs-architecture separation (§7 of the original request) using
  Table 1's already-computed parameter breakdown

No results are interpreted or reported until the actual runs exist —
this document only establishes what's ready to run and verifies what
doesn't need to be rerun.
