# Artifact Status — `docs/assets/exp/geometry-pca/`

> This file annotates the evidentiary status of persisted JSON artifacts against
> the project's final, post-Phase-4 empirical conclusions (2026-06-11).
> The source of truth is `docs/02_EXPERIMENTS_AND_RESULTS.md` — these
> annotations resolve conflicts between artifact-level verdicts and the final,
> cross-examined record.

---

| Artifact | Artifact Verdict | Final Verdict | Status | Note |
|----------|-----------------|---------------|--------|------|
| `za_gate_results.json` | `PASS`, `best_variant: rot` | **FAIL** (overturned 2026-06-11) | ⚠️ STALE | The PASS was an artifact of the low editorial-keypoint z_g baseline (0.540). When z_g was corrected to face-crop resolution (0.688), normals subtract identity signal (ΔAUC −0.039). See ledger Phase 2b face-crop overturn. |
| `phase3b_transfer_results.json` | `PASS`, `fraction_identity_retained: 1.71` | **UNINFORMATIVE** | ⚠️ STALE | The gate technically passed (AUC > 0.51) but missed the random-projection null: random 50-d DINO projections (0.712 ± 0.007) beat the bridge (0.704). The `fraction_identity_retained: 1.71` is nonsensical. See ledger Phase 3b review control. |
| `phase3_bridge_results.json` | R² values only (no verdict) | **FAIL** on C1–C10 criterion | ✅ ACCURATE | The R² measurements are correct. The original "permutation null" (shuffled Y post-prediction) was vacuous but was later fixed in script 32. See ledger Phase 3 systematic review. |
| `zd_gate_results.json` | `FAIL` | **FAIL** | ✅ ACCURATE | Verdict matches final. Minor: `reaon_method` typo (should be `reason_method`). The trace-J metric used here was later found to be blind to complementarity — but the FAIL verdict was independently confirmed by verification AUC. |
| `zd_verification_auc.json` | `FAIL` | **FAIL** | ✅ ACCURATE | Decisive instrument. Matches the face-crop re-run confirmation. |
| `za_systematic_review.json` | Systematic review (2026-06-10) | — | ℹ️ HISTORICAL | Documents the review that accepted the initial PASS. Contains the seed-noise and variance analyses that were valid for the editorial-keypoint measurement context. The PASS itself was overturned when the measurement substrate changed. |
| `metrics.json` / `metrics_production.json` / `metrics_posenorm.json` | Phase 1/1-R metrics | — | ℹ️ HISTORICAL | Phase 1-R encoder performance snapshots. Still valid for the shipped encoder. |
| `gate_sweep_results.json` | z_scale sweep (10 identities, 136 images) | z_scale=1.0 shipped | ℹ️ VALID BUT THIN | The production z_scale=1.0 decision rests on a 10-identity sweep with no uncertainty estimates. The sweep artifact is corrupt and was deliberately not regenerated. The sweep was never re-run on the full 100-identity corpus. This is acknowledged in the ledger. |
| `za_gate_meta.json` / `za_fit_summary.json` | Metadata and fit summaries | — | ✅ ACCURATE | PCA fit statistics and gate metadata. Factual, not verdicts. `za_fit_summary.json` re-verified 2026-11-18 byte-identical to the gitignored `experiments/geometry_pca/output/` copy (sha256 `e7f2337d…`). |
| `zd_complementarity_diagnostic.json` | Cross-examination of trace-J bug | — | ✅ ACCURATE | Documents the 4-metric cross-examination that independently confirmed the z_d FAIL. |
| `zd_fit_summary.json` | z_d PCA fit summaries (modes A / A_prime / C) | — | ✅ ACCURATE | n=69,839; retained variance 0.815 / 0.844 / 0.914. Fit statistics, not a verdict. |

---

## Phase 5b / refit-cleaned evidence (tracked 2026-11-18 — issue #5)

Copied verbatim from the gitignored `experiments/geometry_pca/output/` tree
(byte-identical; sha256 verified at copy time). These close the evidence gap
recorded in the 2026-10-01 write-up staging brief §4: the headline R@1 = 0.8538
previously lived only under a gitignored path.

| Artifact | Content | Status | Note |
|----------|---------|--------|------|
| `phase5b_gt_lda_refit_20260720.json` | GT-LDA64 cross-shoot retrieval on the refitted basis: **R@1 = 0.8538 / R@5 = 0.9305 / R@10 = 0.9507** (variant A Euclidean; 11,110 queries, 30,000 index, 321 personas; chance R@10 = 0.031) | ✅ ACCURATE | Ledger entry `[PRE-REGISTERED] Reproducibility Sprint: LDA Basis Refit with Cleaned Dataset`, gate G2 PASS. Producing script `experiments/geometry_pca/scripts/46b_phase5b_gt_lda_refit.py` (tracked). |
| `phase5b_corrected_masked.json` | Corrected masked text-Prior retrieval vs random-null and caption-shuffle controls + persona bootstrap | ✅ ACCURATE | Prior R@10 = 0.0724 vs null 0.0530; bootstrap k10 Δ = +0.0139, p = 0.0625 — Prior indistinguishable from the null. Ledger: Phase 5a-exp1 corrected-gate story. |
| `phase5b_bootstrap.json` | Persona-level bootstrap (2,000 resamples) of Prior vs null at k=1/5/10 | ✅ ACCURATE | All three deltas statistically null (p = 0.58 / 0.70 / 0.41). Supports the "gap lives in the Prior" conclusion. |
| `phase5b_space_diagnostic.json` | Retrieval-space diagnostic on the pre-refit basis (4 metric variants, A R@1 = 0.8423) | ℹ️ HISTORICAL | Pre-refit ceiling snapshot; superseded by the refit JSON above for the headline number. |
| `phase5b_ffhq_retrieval.json` | FFHQ-side Prior/ceiling probe (GA recall@10 = 0 for prior and null; LDA-1 mean shift within noise) | ℹ️ HISTORICAL | Consistent with one-image-per-identity FFHQ not being an identity-retrieval target. |
| `auraface_lda_ffhq_extremes.json` | FFHQ image IDs at the extremes of LDA axes 1–6 (e.g. LDA1 pos: 68788, 56527, 01970) | ✅ ACCURATE | **Figure provenance**: any public LDA-axes contact sheet must be re-rendered from these FFHQ IDs only (staging brief §5). No proprietary imagery. |

**`za_fit_summary.json`** was already tracked and is byte-identical to the
`output/` copy — no new artifact required; the pairing the 2026-10-01 gap list
asked for (`za_fit_summary.json` + `zd_fit_summary.json`) is now complete here.

---

## Missing Artifacts

The following results have **no persisted JSON artifact** under `docs/assets/`:

| Result | Date | Record |
|--------|------|--------|
| Phase 4 masked patch pooling (AUC 0.797) | 2026-06-11 | Only in ledger prose and tables. Expected artifact: `data/phase4_patch_pooling.json` — check if it exists under the NAS `data/` symlink. |
| Phase 2b face-crop overturn (z_a ΔAUC −0.039) | 2026-06-11 | Only in ledger prose and tables. Expected artifact not specified. |
| z_g face-crop correction (0.540 → 0.671) | 2026-06-11 | Only in ledger prose. |

These are the project's most decision-critical results and the least artifacted. If the raw data exists, it should be copied into the assets tree with a descriptive filename.

---

**Last updated:** 2026-11-18 (issue #5 — Phase 5b evidence closure)
