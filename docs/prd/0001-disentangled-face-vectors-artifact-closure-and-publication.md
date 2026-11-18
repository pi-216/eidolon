---
prd: "0001"
title: "Phase 1–4 Evidence Closure & Scientist-Facing Technical Write-Up for lawrenz.com"
status: ruled
ruled: "2026-11-18"
intent: "moves:artefacts-published"
---

# PRD-0001 — Phase 1–4 Evidence Closure & Scientist-Facing Technical Write-Up

## 1. Problem & Context

1. **Science state (`Phase 1–4` settled; `Phase 5` DiT fusion delegated to `prx-tg`):**
   - `eidolon` has completed its representation and disentanglement program:
     - **50-d Pose/Expression Geometry (`z_g`):** DWPose 68 iBUG keypoints → EPnP 3D frontalization → GPA → PCA (`99.987%` variance), empirically orthogonal to identity (`ridge R^2 ≈ -0.03`), identity-blind (`Fisher J = 0.0847`, `8.31x` noise floor vs `197.5x` for AuraFace-LDA — a `24x` separation), with the `norm > 25 = degenerate` belief falsified (`exp/zg-validity`).
     - **64-d Identity Carrier (`AuraFace-LDA`):** Supervised LDA refit on the cleaned multi-shoot corpus (`324` personas, `166k` images), raising cross-shoot `R@1` from `0.842` to **`0.8538`** (reconstruction ceiling `AUC = 0.9998`), with both `FFHQ` (`69,960` files) and `hegre_corpus` (`31,711` files) reprojected onto verified basis fingerprint **`120e1c5a1dc4f423`**.
     - **Four Falsified Fast Paths (First-Class Negative Results):** Monocular depth (`z_d`, `ΔAUC = -0.023`), surface normals (`z_a`, initial PASS overturned to `ΔAUC = -0.039`), DINOv3→slider bridge (`0.704 < 0.712` random-projection null), and FLAME 3DMM shape (`AUC = 0.585 vs 0.69` keypoint baseline).
2. **Why `artefacts published` is still `0` (`docs/briefings/2026-10-01_eidolon-arxiv-writeup-staging.md` §4):**
   - The 2026-10-01 staging outline identified that `experiments/geometry_pca/output/phase5b_gt_lda_refit_20260720.json` (`R@1 = 0.8538`), `phase5b_corrected_masked.json`, `phase5b_bootstrap.json`, `phase5b_space_diagnostic.json`, and `auraface_lda_ffhq_extremes.json` sit on disk under gitignored `experiments/geometry_pca/output/` rather than tracked `docs/assets/exp/geometry-pca/`.
   - The full scientist-facing technical write-up (`docs/briefings/2026-11-18_disentangled-face-conditioning-stack.md`) has not yet been drafted from the staging outline (`docs/briefings/2026-10-01_eidolon-arxiv-writeup-staging.md`), blocking `pi-216/lawrenz.com#5`.

## 2. Pre-Registered Gate

- **G1 (Evidence Closure):** All headline JSON artifacts from `experiments/geometry_pca/output/` cited by the write-up (`phase5b_gt_lda_refit_20260720.json`, `phase5b_corrected_masked.json`, `phase5b_bootstrap.json`, `phase5b_space_diagnostic.json`, `auraface_lda_ffhq_extremes.json`, `za_fit_summary.json`, `zd_fit_summary.json`) are committed under `docs/assets/exp/geometry-pca/` and verified by `tests/tools/test_governance_sync.py`.
- **G2 (Complete Scientist-Facing Write-Up & Public-Safety Audit):** `docs/briefings/2026-11-18_disentangled-face-conditioning-stack.md` is written in full across all 8 sections of the staging outline, every number cites a tracked path in `docs/assets/`, zero occurrences of `"Hegre"` or `"eidolon"` appear in the public prose, and every face figure is FFHQ-only with explicit FFHQ image IDs.
- **G3 (CI Test-Debt Burn-Down — Issue `#3`):** The 7 excluded test files in `ci/known-failing.txt` are triaged and repaired so `ci/known-failing.txt` shrinks to zero on neutral runners.

## 3. Atomic Ticket Breakdown

- **T1 (`Task` — Close Phase 1–5b Tracked Evidence Gap & Author Full Scientist-Facing Write-Up in `docs/briefings/`):**
  - Copy the tracked JSON evidence files from `experiments/geometry_pca/output/` (`phase5b_gt_lda_refit_20260720.json`, `phase5b_corrected_masked.json`, `phase5b_bootstrap.json`, `phase5b_space_diagnostic.json`, `auraface_lda_ffhq_extremes.json`, `za_fit_summary.json`, `zd_fit_summary.json`) into `docs/assets/exp/geometry-pca/` and update `docs/assets/exp/geometry-pca/ARTIFACT_STATUS.md`.
  - Write the complete scientist-facing technical report `docs/briefings/2026-11-18_disentangled-face-conditioning-stack.md` following the 8-section outline in `docs/briefings/2026-10-01_eidolon-arxiv-writeup-staging.md` (covering the 50-d `z_g` + 64-d `AuraFace-LDA` stack, the `120e1c5a1dc4f423` basis-contamination discovery and FFHQ reprojection, the `z_g` high-norm validity proof and `24x` identity-blindness separation, the 4 falsified volumetric/bridge fast paths, and the 6 methodological gate lessons), ready for `pi-216/lawrenz.com#5`.
  - Archive the 35 no-op `docs/assets/lda-reprojection-audit/` log bloat notes into a single compact summary table so repo context stays clean.
  - Verify `./bin/verify` exits 0 (`VERIFY OK`).
- **T2 (`Task` — Burn Down `ci/known-failing.txt` Test Debt — Issue `#3`):**
  - `Depends on: #<T1>`
  - Fix the `review.db` fixture-root collision in `tests/tools/test_enrichment.py` and the bit-rotted imports/assertions in `tests/tools/test_hegre_dataset_cli.py`, `tests/tools/test_review_ui.py`, and `tools/hegre_dataset/tests/test_averages.py` so they run and pass in `bin/verify`.
