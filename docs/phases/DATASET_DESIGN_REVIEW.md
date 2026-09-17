# Dataset Design Review

**Status:** Reconnaissance complete, per §56 of `SCM_DATASET_GENERATION_PLAN.md`.
**Scope:** Inspect the Stanford SupplySim repo and the NIST sample purchasing
dataset, map real fields to the six-node ontology, propose distributions for
what's missing. **No generator code has been written.** This document is a
checkpoint for approval before Phase 1 (§42) begins.

---

## 1. Repository state

The working directory contained only `SCM_DATASET_GENERATION_PLAN.md` — no
existing code, no git repo. This is a greenfield start; Phase 1 (schema) has
not been touched.

---

## 2. Stanford SupplySim — what to reuse, what to redesign

Cloned `snap-stanford/supply-chains` (shallow) and read
`TGB/modules/synthetic_data.py` (739 lines, the actual simulator) plus the
README, `register_data/preprocess_tesla.py`, and the released synthetic CSVs.

**License finding (blocking for literal reuse):** the repo has **no LICENSE
file**, and `gh api repos/snap-stanford/supply-chains` reports `license: null`
— i.e. no open-source license is granted, so it is all-rights-reserved by
default. Per plan §43.7/§52, we must **not copy their code verbatim**. Below,
"reuse" means reimplementing the same *algorithmic idea* independently, citing
the paper (arXiv:2407.18772, AAAI 2025), not lifting functions.

### 2.1 Mechanisms worth reimplementing

| SupplySim mechanism | What it does | Where it maps in our plan |
|---|---|---|
| Spatial-embedding graph generation (`make_product_graph`, `make_supplier_product_graph`) | Places products/firms in a random 2D embedding per layer; each new node connects to its `k` *nearest* predecessors by embedding distance, not uniformly at random | Directly answers plan §9 ("do not generate edges uniformly at random"). Reusable for Material→Plant and Supplier→Material edge generation, tiered by BOM/region/industry instead of raw 2D space. |
| Preferential attachment for buyer↔supplier assignment (`make_supplier_buyer_graph`) | `P(choose supplier s) ∝ (num_existing_buyers[s] + 1)` | Directly gives §9's "critical suppliers" / supplier-concentration behavior and power-law degree. Reusable for Procurement→Supplier assignment. |
| Exogenous supply shock + geometric recovery (`generate_exog_schedule_with_shocks`) | With probability `shock_prob`, supply drops to `shock_supply`; otherwise `supply(t) = min(default, supply(t-1) * recovery_rate)` until back to baseline | Near-exact template for our Recovery Model (§21). We generalize `recovery_rate` and `shock_prob` to be severity-conditioned (§16) instead of a single global constant. |
| Agent-based discrete-time loop with hard conservation constraints (`generate_transactions`, `simulate_actions_for_firm`) | Each timestep: firms fulfill orders FIFO only if `inventory >= amount * bill_of_materials`; `inventory[t+1] = inventory[t] + deliveries[t] - consumption[t]`; unmet orders roll over as pending | This **is** plan §10/§11 already implemented and tested. Strongest reuse candidate for `operations/inventory.py`, `operations/production.py`. |
| Same-supplier persistence parameter `gamma` | New orders go to the historical supplier with prob `gamma`, else re-sampled uniformly among substitutes | Useful lever for single-source vs multi-source material behavior (§9, §48 ablation). |
| Weekday/weekend demand seasonality + Poisson noise + random drift | `demand(t) = base * {2x weekday, 0.5x weekend}` (or inverse), sampled `Poisson(demand)` | Reusable pattern for §10 normal-operations demand generation. |
| TGB negative-sampling / historical-vs-perturbed eval scaffolding | Standard link-prediction eval framework with historical + perturbed negatives | Relevant later for GNN/QGNN benchmark splits (§26), not for the generator itself. |

### 2.2 Explicitly NOT reused — independently designed in our plan

- **Node typing.** SupplySim has exactly one node type (`firm`, playing both
  supplier and buyer roles) plus a separate product-part DAG. It has **no**
  Plant, Region, or Procurement-order entities, and no geography at all. Our
  six-node heterogeneous ontology (§5) is a genuine extension — nothing to
  port here beyond the general "keep it heterogeneous" spirit (§6).
- **Event taxonomy.** SupplySim has one disruption mechanism (exogenous
  supply shock). It has no Natural Disaster / Geopolitical / Cyberattack /
  Logistics families, no severity scale, no cascade propagation across node
  types, and no ground-truth risk labels. §14–§22 (Black-Swan Event Engine,
  Cascade Simulator, Ground-Truth Labels) are new work, only loosely inspired
  by the shock-recovery idea above.
- **Region/geopolitical modeling.** Not present in SupplySim at all — must be
  designed from scratch (§5.6, §13's "region risk should influence affected
  suppliers").
- **Multi-table benchmark export + validation reporting** (§24, §35) — the
  Stanford repo outputs a single transactions CSV for TGB link-prediction;
  our required output structure (metadata/graph/operations/events/labels/
  validation) is new engineering.

---

## 3. NIST Sample Purchasing / Supply Chain Data — structure

Downloaded `https://data.nist.gov/od/ds/mds2-2183/Sample_Data_Sets.zip`
(20 KB) and extracted it. It contains **5 independent toy scenarios**, each
with 3 tables (`suppliers.csv`, `products.csv`, `projects.csv`):

`01 Basic Sample`, `02 Interconnected Sample`, `GPS Manufacturer`,
`Medical Software`, `Small Government Entity`.

### 3.1 Important finding: this is a schema exemplar, not a statistical source

Row counts top out at **56 rows** per file across all 5 scenarios (most are
2–30 rows). Company names are template/Lorem-Ipsum-style placeholders (e.g.
*"Sem Eget Massa Industries"*, *"Ante Dictum LLC"*), and most optional fields
(address, contact, email) are frequently blank. This is a demo dataset
shipped to illustrate the shape of a Supplier-Relationship-Management (SRM)
export, **not** a real transaction log.

**Consequence for §12 of the plan:** NIST cannot supply the kind of
statistical calibration (order-quantity distributions, degree distributions,
KS-tests against synthetic data) that §12 envisions — sample sizes are far
too small and the content is synthetic placeholder text. Its real value is
**structural**: it confirms plausible field vocabularies and relationship
patterns for a supplier/part/program schema. I'm flagging this now rather
than quietly treating NIST numbers as if they were calibration-grade, per
the plan's own warning in §3.2 ("do not assume it contains every attribute...
missing attributes should be treated as calibration targets").

### 3.2 Exact fields observed, by table

**`suppliers.csv`** (richest in `GPS Manufacturer` / `Medical Software` /
`Small Government Entity` variants): `ID, Name, Street Address, City, State/
Region, Zip, Website, Contact Name, Contact Email, Contact Phone` (SGE variant
adds `SMART Supplier ID, FEIN, Toll Free/Local/Cell Phone`).

**`products.csv`** — varies a lot by scenario:
- Basic/Interconnected: `ID, Name, Supplier ID, Project ID`
- GPS Manufacturer (richest): `ID, BOM Level, Name, Project ID, Revision,
  Phase, Description, Unit of Measure, Procurement Type (MTS/OTS), Part
  Owner, Creation Date, Manufacturer, Manufacturer Part Number/Name (x2),
  Vendor, Vendor Part Number/Name (x2), Supplier ID`
- Medical Software: `ID, CHPL ID, Supplier ID, Type, Name, Version, Project ID`
- Small Government Entity: `ID, Name, Supplier ID, Expire Date, Agency,
  Political Subdivision Availability, Project ID`

**`projects.csv`** (identical shape across all 5): `ID, Level, Name`, where
`Level` is a dot-numbered hierarchy path (`1`, `1.1`, `1.1.1`, ...) — i.e. an
organization → program → sub-assembly tree.

### 3.3 A genuinely useful structural finding

In `SGE_products.csv`, the `Supplier ID` field sometimes contains a
**semicolon-delimited list of multiple supplier IDs for one product**
(e.g. `AC_LLC;SODALES_AT;TEMPUS_COR;PER_INCEPT`). This is real-world evidence
that multi-sourcing is commonly represented as a one-to-many field in SRM
exports — it directly supports plan §9's multi-source requirement and gives
a concrete (if informal) precedent for how the relationship is recorded in
practice, independent of statistical calibration.

---

## 4. Mapping NIST fields → six-node ontology

| Ontology node | NIST source | Notes |
|---|---|---|
| **Supplier** | `suppliers.csv`: `ID`, `Name`, `City/State/Zip` | Only identity + coarse location are real; all risk/capacity/financial attributes (§5.1) are absent — must be simulated. |
| **Material** | `products.csv` (NIST's "Product" is actually a BOM line item / component, not a finished good — evidenced by `BOM Level`, `Procurement Type`, part numbers) | `Procurement Type` (MTS = make-to-stock vs OTS = off-the-shelf) is a real, adoptable categorical field — worth adding as an optional `material` attribute informing `substitutability`. `BOM Level` gives a (very small-sample) precedent for tiered depth. |
| **Product** | `projects.csv`, top-of-hierarchy rows (`Level = 1`, e.g. "My Organization" / "Navit Inc.") | The `Level` tree is closer to a BOM/program hierarchy than a finished-product catalog; use it only as a template for **branching-factor shape**, not as literal Product records. |
| **Plant** | *none* | NIST has no facility/site entity whatsoever. Fully simulated. |
| **Region** | `suppliers.csv` `City/State/Zip` (frequently blank) | Only raw place names, no risk indices. Region risk scores (geopolitical/disaster/infrastructure/trade/cyber/transport) have no NIST source — must come from an external public risk index (e.g. WGI, INFORM Risk Index) or be documented as fully synthetic assumptions. |
| **Procurement Order** | *none* | NIST tables are static relationship tables (who supplies what for which program), not an order/transaction log — no quantities, no order values, no dates besides a single product `Creation Date`. Fully simulated via the SupplySim-style agent loop (§2.1 above), not from NIST values. |

---

## 5. Fields that must be simulated, with proposed distributions/dependencies

Following plan §13 ("use dependency functions, not independent sampling").
These are starting proposals for Phase 2–3, not final — flagging for
approval.

| Field(s) | Proposed family | Dependency |
|---|---|---|
| `capacity`, `production_capacity` | Log-normal | Scaled by `tier` (upstream tiers → larger, fewer, more concentrated capacity) |
| `capacity_utilization` | Beta(α,β) in [0,1] | Negatively coupled to spare capacity: `utilization ~ Beta` conditioned so `reliability` falls as utilization → 1 |
| `reliability`, `quality_score`, `financial_health` | Beta in [0,1] | Jointly drawn via a Gaussian copula (plan §12 suggests copulas for correlated variables) so weak financial health correlates with lower reliability |
| `lead_time_mean` | Gamma | Shape/scale by `material_category` and `region.transport_reliability` |
| `lead_time_variability` | Log-normal, scaled off `lead_time_mean` | Higher mean lead time → proportionally higher variance |
| `geopolitical_exposure`, `disaster_exposure`, `cyber_exposure` (supplier) | Inherited from assigned `region.*_risk` + idiosyncratic noise | `exposure = f(region_risk, industry)`, not independent draws — directly implements §13's example |
| `supplier_count`, `concentration` (material) | Derived from generated topology, not sampled | Computed post-hoc from the preferential-attachment edge generation (§2.1), not an independent random field |
| `region.*_risk` (5 indices) | Beta in [0,1], spatially/categorically clustered by a small number of "region archetypes" | Avoids i.i.d. region risk; regions in the same archetype cluster share elevated risk, matching §13's "region with high geopolitical exposure should influence affected suppliers" |
| Procurement `order_quantity`, `order_value`, `order_frequency` | Emerge from the agent-based simulation loop (§2.1), not sampled directly | This is the SupplySim mechanism — order quantities are a *consequence* of demand/inventory/capacity dynamics, which is what makes them internally consistent (satisfies §11.5) |
| `supplier_risk` composite (§13) | Weighted function of the above, explicitly **not** copied into the ML label | As mandated by §13/§43.2 |

---

## 6. Literature reference (plan §57)

`doi.org/10.1145/3801228.3801250` resolves to ACM DL (`dl.acm.org`), which
returned HTTP 403 to automated fetch, and it did not surface in web search
(the DOI prefix `3801228.*` matches *Proceedings of the 2026 5th
International Conference on Big Data, Information and Computer Network*
(BDICN '26), based on neighboring DOIs in the same prefix range, but I could
not independently confirm this specific paper's title/abstract). I could not
verify this citation beyond the plan's own one-line description. Closely
related, independently verifiable recent work for the same framing (scarce
black-swan samples, dynamic graph attention for SC risk) includes
"DynSupplyNet" and hypergraph dynamic-graph-attention (HG-DRA) approaches
surfaced during search — worth a closer look during literature review, but
that's Phase 7+ work, not blocking the generator.

---

## 7. Decisions

1. **NIST's calibration role stays downgraded — proceed without blocking on
   a second real dataset.** NIST is documented as a *schema/relationship-
   pattern reference* only (§3.1). Rather than delay Phase 1 hunting for a
   large enough real transaction dataset, the generator proceeds with
   constraint-based synthetic generation (topology + operations + events, per
   §9–§21) as the primary path. In Phase 4, make a time-boxed attempt to find
   a supplementary dataset for *topology-only* validation (degree
   distribution, clustering — plan §3.3's stated use case); if nothing
   suitable turns up, this is documented as an explicit limitation per §53
   rather than silently assumed away. Rationale: the plan's own philosophy
   (§54) is structure-first/operations-second/events-third, with real data as
   a constraint on top, not a prerequisite to start — and the earlier
   attempt already showed that public transaction-level SCM data at usable
   scale is not trivially available.

2. **SupplySim reuse = reimplement the algorithmic idea in original code,
   not import the package.** Confirmed, given the repo has no license
   (`license: null` via `gh api`). Each reused mechanism (§2.1 table) gets
   its own from-scratch implementation under `src/scm_dataset/`, with a code
   comment/docstring citing arXiv:2407.18772 where the idea originates, and
   no vendored files from `snap-stanford/supply-chains`.

3. **Region risk indices seeded from real public data, not fully synthetic
   archetypes.** Verified both candidates are actually usable:
   - **World Bank Worldwide Governance Indicators (WGI)** — CC-BY-4.0,
     public, CSV/Excel via `databank.worldbank.org` — seeds
     `region.geopolitical_risk` and `region.trade_risk` (political
     stability, rule of law, regulatory quality dimensions).
   - **INFORM Risk Index (EU JRC)** — CC-BY-4.0, public, Excel via
     `drmkc.jrc.ec.europa.eu`, DOI `10.2905/JRC.E5A3SHG` — seeds
     `region.natural_disaster_risk` and `region.infrastructure_risk`.
   `cyber_risk` and `transport_reliability` still have no clean public
   country-level source found; these stay synthetic (archetype-based,
   correlated with the other four per §13), documented as such. Both real
   sources get a provenance record per the §3.3 YAML template
   (`source_name/source_url/access_date/license/...`) when ingested in
   Phase 4. This directly strengthens the "realism-constrained" framing
   that's central to the paper's positioning (§2), at low implementation
   cost (static per-country scores, no transaction-level ETL needed).

4. **`Procurement Type` (MTS/OTS) adopted as an extra `material`
   attribute.** The plan labels §5's attribute lists "Suggested attributes,"
   not a closed contract, and this field is real-world-grounded, cheap to
   carry, and directly informs `substitutability` without conflicting with
   any existing field. Added to `schema/nodes.py` material schema as an
   optional categorical attribute, sourced from this reconnaissance (§3.2)
   rather than invented.

Proceeding to Phase 1 (schema) next.
