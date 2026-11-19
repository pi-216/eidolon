# A Disentangled Face Conditioning Stack: a Pose-Invariant Geometry Control and a Compact Identity Carrier — and the Negative Results That Shaped Them

**Date:** 2026-11-18 · **Status:** `[DRAFT — staged for founder review; nothing published]`
**Public tag:** `face-vectors`
**Scope:** this document describes *representations and the experiments that validated them*. No generative renderer has been trained on this stack yet; the paper says so explicitly (§7).

---

## Abstract

We describe a face as two empirically-orthogonal vectors: a 50-dimensional pose/expression geometry control (`z_g`), built from 68 facial keypoints via EPnP 3D frontalization and PCA, and a 64-dimensional identity carrier (AuraFace-LDA), an ArcFace-lineage embedding compressed by supervised LDA. The two are orthogonal by measurement, not assumption: a ridge regression from `z_g` to AuraFace on held-out FFHQ pairs scores **R² = −0.033** — worse than predicting the mean. The geometry space is identity-blind: its Fisher separability **J = 0.0847** sits only **8.31×** above its own noise floor `(C−1)/(N−C)`, while the identity space sits **197.5×** above the floor — a **24× separation** that is the quantitative basis for the disentangled design. The identity carrier is discriminative across photographic shoots (**cross-shoot R@1 = 0.8538**) and near-lossless for verification (reconstruction ceiling **AUC = 0.9998**). We then report, as a first-class contribution, five falsified "fast paths" — monocular depth, surface normals (an initial PASS we overturned ourselves), a DINO→slider bridge, FLAME 3DMM shape parameters, and a text→`z_g` prior — and the gate-design discipline (verification AUC, random-projection nulls, do-nothing nulls, noise-floor calibration) that caught two of our own bad verdicts.

---

## 1. Introduction

A face photograph factorizes into *who* and *how posed*. A controllable generative system should carry those two facts in separate slots, so that pose can be edited without drifting identity and vice versa. The naive architecture for that separation is a **fast path**: take an off-the-shelf monocular model (depth, normals, a 3D morphable model, or a vision foundation embedding), regress a compact "structural identity" vector from it, and condition the generator on that.

This paper reports that the fast path is false, and what survived the falsification.

The surviving stack is two vectors and one control stream:

| Signal | Carrier | Dim | Key mechanism |
|---|---|---|---|
| Pose/expression geometry control | DWPose 68 iBUG keypoints → EPnP 3D frontalization → GPA → PCA → whiten | 50 | pose-invariance by construction; k=50 retains **99.987%** of variance |
| Identity | AuraFace 512-d → PC1 + yaw removal → supervised LDA | 64 | cross-shoot **R@1 = 0.8538**; verification ceiling **AUC 0.9998** |
| Shape morphology (complementary) | Sapiens2 dense keypoints | — | linearly AuraFace-orthogonal (**ridge R² = −0.11**) |

The firewall between the two slots is *measured*: ridge regression from `z_g` to AuraFace on 8k held-out FFHQ pairs gives **R² = −0.033**, i.e. the geometry basis linearly explains nothing of the identity embedding, and the planned "project `z_g` out of AuraFace" step was dropped as pointless — there is nothing linear to remove (§4).

The spine of this document is the falsification program: five popular routes to structured identity geometry, each killed by a pre-registered gate, two of them after an initially positive verdict that our own review overturned (§5, §6).

---

## 2. Data and Basis Hygiene

### 2.1 FFHQ (public)

The public dataset is **FFHQ** (NVIDIA, **CC BY-NC-SA 4.0**), 70,000 images at 1024², preprocessed with our public `stratum-ffhq` pipeline. Every figure in this document is FFHQ-derived; no other imagery appears anywhere. FFHQ carries one structural limitation we state up front: **~1 image per identity**, which makes it unusable for cross-shoot identity evaluation (an "identity match" in FFHQ is a near-duplicate detection task, not a generalization task). FFHQ is therefore used for encoder fitting, basis hygiene, and figures — never as the identity-validation set.

### 2.2 Proprietary multi-shoot validation dataset

Identity validation uses **the proprietary multi-shoot dataset, used exclusively for cross-shoot validation**. It contains ~324 personas with ~166k human-approved images (median ~200 images/persona) spanning many separate photographic shoots, with strict per-image human review. It is **never used for encoder fitting** — encoders are fit on FFHQ and *measured* on this set, so cross-shoot numbers cannot be inflated by seeing the same person in training. No image from this dataset appears in any figure or will appear in any publication.

"Cross-shoot" is the load-bearing constraint: verification/retrieval pairs of the same person are **required to come from different shoots**, which removes lighting, styling, and background leakage from every headline number.

### 2.3 The mixed-basis contamination, and the fingerprint guard

In September 2026 we discovered that the identity stream had silently forked. The LDA basis had been refitted on a cleaned, human-approved corpus (2.1× more personas, 3× more images), and the training corpus rebuilt around it — but the **FFHQ identity embeddings were never reprojected onto the refit basis**. Files older than the refit proved bit-exact pre-refit projections: `‖stored − project_old(raw)‖ = 0.00000000` while `‖stored − project_new(raw)‖ ≈ 153.5`.

The consequence: any adapter trained on a mixture of FFHQ and validation-corpus samples was feeding its 64-d identity slot **two incompatible encodings** — FFHQ embeddings at norm **0.350** (σ 0.035) against corpus embeddings at norm **1.000000000** (L2-normalized). The corruption was invisible to loss curves. A separate corpus audit found the same failure class from the other side: **5,306 of 37,011 corpus directories (14.3%) carried the previous basis**, because per-sample directories that a rebuild did not re-select simply survive — carrying stale persona averages.

The repair and the guard:

1. **Reprojection** of all 69,960 FFHQ identity embeddings onto the refit basis, L2-normalized (provably cosine-preserving: between-image cosine 0.9945 ± 0.0012 before/after), 69,960/69,960 written, 0 errors (`scripts/reproject_lda_ffhq.py`).
2. **A basis fingerprint**: a deterministic hash of the basis matrices, **`120e1c5a1dc4f423`**, stamped into every consumed data directory and recorded in every corpus manifest. Loaders call `assert_basis()` and **fail closed** on an unstamped directory or a stamp that disagrees (`basis_fingerprint.py`, guarded by `tests/tools/test_basis_fingerprint.py`, including a negative control proving the guard can fire). The negative-control gate was written *before* the run: a guard that cannot fail does not count.

Both mixed-basis findings are recorded in the permanent ledger before any downstream result is quoted, and the affected downstream arms were declared confounded rather than quietly salvaged.

---

## 3. Method

### 3.1 The geometry control `z_g` (50-d)

Pipeline (shipped frozen): DWPose whole-body keypoints → slice 68 iBUG face points → mean-confidence prefilter → **3D frontalization**: estimate head rotation via orthographic EPnP against the canonical 300W 68-point template (z_scale = 1.0), rotate the lifted point set to canonical frontal, reproject → light 2D GPA → PCA (k=50) → whiten.

Why 3D frontalization rather than 2D GPA: 2D Procrustes neutralizes translation, scale, and in-plane rotation but is **blind to out-of-plane yaw/pitch**, so PCA correctly shoved pose into its top components. An early encoder passed a gate on clean C1 = yaw / C2 = pitch traversals — and we revoked our own PASS: the encoder was pose-*entangled*, answering the wrong question. An identity-adjacent geometry control must be pose-invariant by construction.

The shipped encoder is fit on the **full 69,851** qualifying FFHQ faces (107 s), retaining **99.987%** of variance at k=50. A synthetic-pose probe (one identity, yawed ±30°) measures the fix: mean per-component std of `z_g` across the yaw sweep drops **1.45 → 0.29** (4.9×) versus the 2D-GPA encoder, with variance retention unchanged. Component semantics at ±3σ (see §"Figure provenance"): C1 face width/aspect morphology, C2 upper-face/brow structure, C4 mouth opening with eyes and nose strictly static.

`z_g` is *not* an identity carrier — that is the point. It is a control surface: traverse any of its 50 axes to change pose, expression, and proportion without moving identity (§4.1 measures how well that holds).

### 3.2 The identity carrier: AuraFace-LDA (64-d)

AuraFace (ArcFace-lineage face-recognition embedding, 512-d) → remove a domain-artifact PC1 axis and a head-pose/yaw direction → **supervised LDA on personas** → 64 dims. The LDA basis was refitted on the cleaned multi-shoot corpus: **324 personas, 166,195 approved images** (pooled with FFHQ: 236,155 vectors; 259 train personas / 137,102 train images).

The representation is near-lossless for verification: skipping the learned prior entirely and verifying **ground-truth LDA-64 coordinates reconstructed against raw AuraFace** scores **AUC = 0.9998** (raw-vs-raw sanity 0.9989; LDA-32 0.9765; LDA-16 0.8658). Whatever identity work fails, it fails in the conditioning/prior, not in the 64-d carrier.

The prior itself has not yet reached that ceiling: text→LDA retrieval reaches **R@10 = 0.0724** against a random-projection null of 0.0530 and chance 0.031, with a persona-level bootstrap p = 0.0625 at k=10 — statistically indistinguishable from the null. We report this as the open gap (§7), not bury it: the ceiling experiment proves the target representation is sound and localizes the failure to text conditioning.

### 3.3 The firewall, measured

Ridge regression `z_g` → AuraFace, 8k held-out FFHQ pairs: **R² = −0.033** (worse than the mean predictor). The streams are linearly independent by measurement; the generative design keeps them in decoupled cross-attention streams with asymmetric classifier-free-guidance dropout as the non-linear backstop.

---

## 4. Results — What Survived

### 4.1 `z_g` is identity-blind — and its high-norm tail is real pose

Two pre-registered arms established that the geometry slot carries almost no identity.

**Identity content (curated corpus, 31,711 samples / 321 personas, confirmatory arm):**

| stream | Fisher J | J / noise floor (C−1)/(N−C) |
|---|---|---|
| per-image AuraFace-LDA (positive control) | **2.0137** | **197.5×** |
| `z_g` | **0.0847** | **8.31×** |

The ratio is **23.8×** — the quantitative separation that licenses the disentangled architecture. Three honest caveats are part of the record: an earlier quoted J = 0.059 was **withdrawn** (its producing script exists on no branch, so by our own evidence rule it was not evidence); raw J is **size-confounded** — the noise floor `(C−1)/(N−C)` is 2.2× higher on the smaller curated corpus, so raw J *rises for free* when N falls (size-corrected, `z_g`'s relative signal **fell 12.60× → 8.31×**); and the pre-registered verdict was **PARTIAL**, not CONFIRM — the morphology-axis count (11) missed the pre-registered ≤10 bar by one axis, and a pre-registered gate is never moved after seeing the count. Artifacts: `docs/assets/exp/zg-identity-blindness/fisher_metrics.json`; code `experiments/zg_identity_blindness/src/run_fisher.py`; pinned at commit `29aba37d6e0fc499eb3f011e6d28ccb1cbd4c5a7` (tag `arm/zg-identity-blindness`).

**High-norm tail (validity arm, confirmatory, gate locked before measurement):** a long-standing belief held that `z_g` norm > 25 flagged DWPose landmark failure. Sixty high-tail samples against 60 controls: **0 missing keypoints and 0 sub-0.3-confidence keypoints in either group**; alignment residuals are clean. The strata differ in **pose atypicality** (the tail is atypical, often head-inverted poses with *accurate* landmark alignment — e.g. eye–mouth vertical ratio 2.695 vs 1.173 controls), not landmark quality. The filter belief was **KILLed**; `z_g` norm became a pose-atypicality index. Artifacts: `docs/assets/exp/zg-validity/g2_diagnostics.json`, `review_crosstab.json`, skeleton sheets `g1_skeleton_{high,ctrl}.jpg`; pinned at commit `c0ed696d82e9544fe63e38d5594384db144a6cc7` (tag `arm/zg-validity`). (The skeleton sheets are rendered from validation-set imagery and are internal-only; the public figures are the FFHQ traversals in §Figure provenance.)

### 4.2 AuraFace-LDA retrieves identity across shoots

Cross-shoot retrieval on the refitted basis (query = held-out shoot; index = remaining shoots; 11,110 queries, 30,000 index, 321 personas; chance R@10 = 3.1%):

| metric space | R@1 | R@5 | R@10 |
|---|---|---|---|
| **GT-LDA64, Euclidean (shipped)** | **0.8538** | 0.9305 | 0.9507 |
| GT-LDA64, cosine | 0.8504 | 0.9307 | 0.9528 |
| GT-LDA64, z-scored | 0.8484 | 0.9272 | 0.9475 |
| reconstruction round-trip | 0.8134 | 0.9001 | 0.9249 |

The cleaning + refit raised the ceiling **0.842 → 0.8538**; the previously published post cites 0.842 (pre-refit basis) and will carry a correction note. Evidence: `docs/assets/exp/geometry-pca/phase5b_gt_lda_refit_20260720.json` (tracked as of 2026-11-18; previously a gitignored file), produced by the tracked script `experiments/geometry_pca/scripts/46b_phase5b_gt_lda_refit.py`. Companion artifacts now tracked alongside it: the Prior-vs-null corrected retrieval probe (`phase5b_corrected_masked.json`), the persona bootstrap (`phase5b_bootstrap.json`), and the pre-refit space diagnostic (`phase5b_space_diagnostic.json`).

### 4.3 A complementary morphology stream: Sapiens2 dense keypoints

Sapiens2 keypoints *measure* rather than hallucinate: they withhold ~22% of keypoints per image under occlusion (33% on profiles) with genuine per-keypoint confidence, while DWPose plants all 68 points regardless of visibility. Within-person configuration scatter is **0.0076 vs 0.0243** (3× more stable). As identity features they beat DWPose shape decisively: at 100 personas × 15 cross-shoot images, AUC 0.745 [0.725, 0.763] vs 0.650 [0.633, 0.667], **Δ = +0.096 [+0.079, +0.113]**, persona-level bootstrap.

But two findings keep them a *third* stream rather than a `z_g` replacement. First, **no identity-blind transient block exists in Sapiens2 shape** — every PCA axis is person-discriminative (min axis J = 0.099; 0 transient axes vs 42 morphology axes); dense faithful landmarks capture identity so well that *how the face moves is itself identity*. `z_g`'s DWPose coarseness **is** its identity-blindness; load-bearing, kept. Second, full Sapiens2 shape → AuraFace ridge **R² = −0.11** — *more* orthogonal than `z_g`'s −0.03: its shape-identity content is linearly independent of appearance-identity, making it a safe complementary morphology/morphing axis (anatomical sliders + viewpoint), guarded downstream by CFG dropout against non-linear leakage. Evidence: `experiments/sapiens2_keypoints/output/study_*.json` (tracked).

### 4.4 Settled identity conditioning for the renderer: masked DINOv3 patches

For a generative renderer the identity slot is conditioned on **flesh-masked DINOv3 patch tokens** (mask-average-pooled evaluation embedding): cross-shoot verification AUC **0.7975** (cross-shoot-only 0.7965) versus 0.7691 for the raw CLS token, identity-level bootstrap Δ = +0.027 [+0.014, +0.045]. Provenance caveat, recorded rather than papered over: this is one of two numbers in this document whose **result JSON was never committed** — the producing script is tracked (`experiments/geometry_pca/scripts/37_dino_patch_face_pooling.py`), the number lives in ledger prose and the annotated artifact status, and re-derivation is scheduled before final publication. (The second is the `z_a` overturn, §5.2.)

---

## 5. Results — The Negative Results (a First-Class Contribution)

| Path | Verdict | Headline number | Tombstone |
|---|---|---|---|
| `z_d` monocular depth (64×64 → face-crop, k=50) | **FAIL** | ΔAUC **−0.023** (full face-crop set) / **−0.034** (seg-clean subset) | `DISCONTINUATION_NOTICE_zd_depth.md` |
| `z_a` surface normals | **FAIL** (initial PASS overturned) | ΔAUC **−0.039** | `DISCONTINUATION_NOTICE_za_normals.md` |
| DINOv3-cls → slider bridge (ridge, both directions) | **FAIL** | bridge 0.704 < **random-projection null 0.712 ± 0.007** | `DISCONTINUATION_NOTICE_dino_bridge.md` |
| FLAME 3DMM shape parameters (β) | **FAIL** | AUC **0.585** vs 0.69 keypoint baseline | ledger tombstone (red-lines list) |
| text → `z_g` learned prior | **FAIL** | per-dim MSE/variance ratio **1.75** — 75% *worse* than predicting the mean | ledger tombstone (red-lines list) |

**5.1 Depth.** The pre-registered gate (J lift ≥ ×1.15) failed at ×1.02; a systematic review then caught that the *trace* Fisher ratio is a weighted average for concatenated vectors and mathematically guarantees failure for any weaker partition — so the verdict was re-established with the decisive instrument (verification AUC): depth *subtracts* identity signal in every mode, at **24× facial resolution** after eliminating both the resolution and domain-shift escape hatches. Decisive artifact: `docs/assets/exp/geometry-pca/zd_verification_auc.json`.

**5.2 Normals — our own overturned PASS.** The initial PASS (+0.028, 10/10 seeds) was real against the baseline of the day (z_g AUC 0.540) and wrong for a reason we found later: that baseline was an *editorial-crop keypoint-resolution artifact* (see §6.4). Re-tested at proper face-crop resolution (z_g baseline 0.688), normals **subtract** identity signal (ΔAUC −0.039). The artifact status file marks the original PASS artifact **STALE** rather than deleting it. Provenance flag: the overturn itself is recorded in ledger prose; its result JSON was never committed and is scheduled for re-derivation before publication.

**5.3 The DINO bridge and the null that kills it.** Regressing physical sliders from DINOv3 cls tokens scored held-out R² 0.690 / 0.385 (per-component criterion failed), and the transfer gate (AUC > 0.51) technically passed at 0.704 — but a **random 50-d Gaussian projection of DINO itself** scores **0.712 ± 0.007**. The bridge is worse than a random shadow of its own input; the gate was structurally too weak because it lacked that null. The same review corrected the review: the original "permutation null" shuffled labels *after* prediction (analytically vacuous); fixed in the re-run. Artifact: `docs/assets/exp/geometry-pca/phase3b_transfer_results.json` (annotated UNINFORMATIVE in the artifact status).

**5.4 3DMM shape.** FLAME β parameters fit per-person score AUC 0.585 as cross-shoot identity features, against 0.69 for plain 2D keypoint geometry — thirty years of parametric 3DMM literature does not buy you identity. (Recorded in ledger summary prose and the red-lines tombstone; the study predates the artifact convention.)

**5.5 Text → `z_g`.** A rectified-flow prior trained to predict `z_g` from T5 captions scores per-dim MSE / per-dim variance = **1.75**, i.e. 75% worse than a constant mean predictor (ratio 1.0 by definition). Informative negative: `z_g`'s 50 axes are fine-grained pose/framing/camera detail that captions do not describe — which is *exactly* why `z_g` belongs to a user-side slider control at inference and not to a text prior.

**The unifying claim.** Monocular volumetric models hallucinate *plausible but generic* human geometry: visually stunning, identity-poor. Depth, normals, and 3DMM shape all land at or below the information already in 2D landmarks. Do **not** retry this class with a better monocular model — the failure is structural, and each route carries a tombstone that states the structural reason, so that no future run re-attempts it.

---

## 6. Six Methodological Lessons (each bought with a real near-miss)

1. **Verification AUC beats trace-J for partition gates.** For a concatenated vector, trace Fisher J is exactly a weighted average of its parts — a weaker partition *always* drags it down even when perfectly complementary. The canonical instrument is now same/different-identity verification AUC (cosine, z-scored, balanced pairs). The trace-J bug is itself documented by an artifact (`zd_complementarity_diagnostic.json`).
2. **Every transfer/learned-mapping gate needs a random-projection null of its input.** Our bridge "PASSED" a pre-registered gate and then lost to a random 50-d Gaussian projection of its own input (0.704 vs 0.712 ± 0.007). A gate without the right null "passes" on structure the test never isolates. (Converse note: for a *raw geometric feature*, a projection null is invalid — Johnson–Lindenstrauss preserves cosine; there the floor is a label-shuffle null. Match the null to the hypothesis.)
3. **Run the do-nothing null before any PASS.** Predict-the-mean defines the baseline for any regression gate (ratio 1.0); label-shuffle and AUC 0.5 for classification gates. The text→`z_g` FAIL is only meaningful *because* the null is 1.0.
4. **The keypoint-resolution crop trap: confidence ≠ precision.** `z_g` verification AUC measured 0.5405 with keypoints extracted from wide editorial frames and **0.6706** with the *same frozen encoder* on tight face crops — +0.13 AUC purely from landmark precision — while reported keypoint *confidence* was **higher** in the worse condition (0.944 vs 0.865). That single artifact inverted a verdict in both directions: it made `z_g` look "very weak" (it is moderate, ~0.67–0.69) and it made the normals PASS look real (it was not; §5.2).
5. **Verify identity test sets visually.** Segmentation collapse on ~10% of tight crops produced near-empty masked depth vectors that were cross-identity near-duplicates (cos 0.994 across different people); a visual skeleton audit (and a 97-row exclusion re-run that *strengthened* the negative result) is now mandatory before accepting a gate. The contamination near-misses are named in the ledger.
6. **Calibrate Fisher J against its noise floor `(C−1)/(N−C)`; small-N splits are optimism.** A synthetic calibration (before reading any corpus result) showed random labels give J ≈ the floor. Cross-corpus J is meaningless without it: our `z_g` J "rose" 0.059 → 0.0847 while the *size-corrected* signal **fell** 12.60× → 8.31×. Likewise a legacy "27 morphology axes" split collapsed to ~6 at full corpus — small-N optimism, retired.

Governance frame: every arm above declares `mode: confirmatory | exploratory` before its first run; only confirmatory arms write PASS/FAIL; a gate locked after peeking is exploratory forever; verdicts are pinned to tagged commits (`arm/zg-validity`, `arm/zg-identity-blindness`) whose trees contain both code and evidence.

---

## 7. Limitations and Open Debts (the honest section)

- **No trained generative renderer yet.** Everything here is representation-level evidence and gates; the DiT conditioning stack built on it is designed and delegated, not trained.
- **The identity carrier's ceiling is not the system's performance.** GT-LDA retrieval R@1 = 0.8538 and ceiling AUC = 0.9998 bound the *representation*; the text→identity prior remains at R@10 = 0.0724, statistically indistinguishable from the random-projection null (p = 0.063 at k=10).
- **The LDA basis carries a population prior** (fit on the validation corpus; FFHQ projects onto it without collapsing but at 53% of the corpus spread).
- **FFHQ is ~1 image per identity**; its reprojection made the identity stream encoding-consistent, not a valid identity *target*.
- **`z_g` encoding reliability at pose-distribution extremes is an open pre-registration** (the validity arm established the tail is real pose, not that the encoding behaves well there).
- **Corpus-quality debt:** extreme-pose / off-frame crops pass human review and affect all streams; one 19-row constant-distance defect remains unexplained.
- **One z_scale decision rests on a thin, corrupt-but-documented sweep artifact**, acknowledged in the ledger rather than hidden.
- **Two prose-only numbers** (`z_a` overturn −0.039; masked-patch AUC 0.797) will be re-derived into committed artifacts before publication (§4.4, §5.2).

---

## 8. Conclusion

Two vectors — a 50-d pose-invariant geometry control and a 64-d identity carrier — separated by a firewall that measures (R² −0.033) rather than asserts, with the geometry space 24× closer to its identity-noise floor than the identity space; and five falsified fast paths that collectively say monocular structural geometry cannot carry identity. We publish the overturned PASSes and the withdrawn numbers alongside the survivors because the gate discipline — nulls, floors, visual audits, tagged-commit evidence — is the result most reusable by other researchers.

---

## Claims & Figure Provenance

Every headline number above traces to a **git-tracked artifact** under `docs/assets/` (or a tracked results JSON) and a ledger entry in `docs/02_EXPERIMENTS_AND_RESULTS.md`. Rows marked ⚠️ are honest provenance gaps, scheduled for artifact re-derivation before any public release.

| Claim | Number | Tracked artifact / code |
|---|---|---|
| Identity cross-shoot retrieval | R@1 0.8538 (R@5 0.9305, R@10 0.9507) | `docs/assets/exp/geometry-pca/phase5b_gt_lda_refit_20260720.json` + `experiments/geometry_pca/scripts/46b_phase5b_gt_lda_refit.py` |
| Prior vs null (k=10) | 0.0724 vs 0.0530, p 0.0625 | `docs/assets/exp/geometry-pca/phase5b_corrected_masked.json`, `phase5b_bootstrap.json` |
| LDA-64 verification ceiling | AUC 0.9998 | ledger Phase 5a ceiling test (⚠️ prose-only table; re-derive at publication) |
| `z_g` identity content | J 0.0847 / 8.31× floor; AuraFace 2.0137 / 197.5× | `docs/assets/exp/zg-identity-blindness/fisher_metrics.json` · tag `arm/zg-identity-blindness` (`29aba37d…`) |
| `z_g` high-norm tail is pose | 0 missing kp, 0 conf<0.3; ratio 2.695 vs 1.173 | `docs/assets/exp/zg-validity/g2_diagnostics.json`, `review_crosstab.json` · tag `arm/zg-validity` (`c0ed696d…`) |
| `z_g` fit / variance | 69,851 faces, k=50, 99.987% | `docs/assets/exp/geometry-pca/metrics_production.json`, `prod_scree.png`, `recon_error.png` |
| Firewall orthogonality | ridge R² −0.033 | ledger Phase 5-prep (⚠️ prose-only; re-derive at publication) |
| Depth FAIL | ΔAUC −0.023 / −0.034 | `docs/assets/exp/geometry-pca/zd_verification_auc.json`, `zd_complementarity_diagnostic.json`, `zd_fit_summary.json` |
| Normals FAIL (overturn) | ΔAUC −0.039 | `za_gate_results.json` (⚠️ STALE, annotated) + ledger overturn prose |
| Bridge FAIL vs null | 0.704 < 0.712 ± 0.007 | `docs/assets/exp/geometry-pca/phase3b_transfer_results.json` (⚠️ verdict stale, annotated) |
| FLAME β FAIL | 0.585 vs 0.69 | ledger summary + red-lines tombstone (⚠️ prose-only) |
| text→z_g FAIL | MSE ratio 1.75 | ledger Phase 5a-exp1 corrected-gate table |
| Masked DINO patches | AUC 0.7975 / 0.7965 | `experiments/geometry_pca/scripts/37_dino_patch_face_pooling.py` (⚠️ result JSON uncommitted) |
| Sapiens2 morphology stream | ΔAUC +0.096 [+0.079,+0.113]; ridge −0.11 | `experiments/sapiens2_keypoints/output/study_results_wide.json` etc. (tracked) |
| Basis guard | fingerprint `120e1c5a1dc4f423`; 69,960 reprojected, 0 errors | `scripts/reproject_lda_ffhq.py`, `basis_fingerprint.py`, `tests/tools/test_basis_fingerprint.py`; norm audit `docs/assets/exp/sapiens2-keypoints-study/evidence-20260923/` |
| LDA refit fit statistics | 324 personas / 166,195 images; 236,155 pooled | `docs/assets/exp/geometry-pca/za_fit_summary.json` (PCA stats, factual), ledger G1 table |

**Figures.** All figures in any public release are FFHQ-derived and contain **no photographic faces**:

| Figure | Tracked asset | Public-safe |
|---|---|---|
| `z_g` ±3σ component traversals (C1–C5) | `docs/assets/exp/geometry-pca/posenorm_traversal_C{1..5}.png`, `prod_traversal_C{1..5}.png` | ✅ landmark-scatter renders from the FFHQ-fit basis; visually verified to contain no photographic faces |
| Scree / variance retention / reconstruction error | `prod_scree.png`, `scree.png`, `recon_error.png` | ✅ plot-only |
| LDA-axes extremes (if a face contact sheet is ever published) | FFHQ image IDs in `docs/assets/exp/geometry-pca/auraface_lda_ffhq_extremes.json` (LDA1 pos 68788, 56527, 01970; neg 55145, 34801, 68584) | ⚠️ must be re-rendered from these FFHQ IDs only, with `<small>` NVIDIA CC BY-NC-SA 4.0 attribution listing exact IDs |
| `z_g` validity skeleton sheets | `docs/assets/exp/zg-validity/g1_skeleton_*.jpg` | ❌ rendered from validation-set imagery — **internal only; never publish** |

No image from the proprietary multi-shoot dataset appears in this document or may appear in its public release. FFHQ is attributed to NVIDIA (CC BY-NC-SA 4.0); `stratum-ffhq` is attributed to its maintainer.

---

## References

1. Permanent experiment ledger: `docs/02_EXPERIMENTS_AND_RESULTS.md` (every verdict, gate, and overturn cited above).
2. Staging brief and claims audit of 2026-10-01 (the artifact-gap list this document closes).
3. FFHQ: Karras et al., *Progressive Growing of GANs* (NVIDIA, CC BY-NC-SA 4.0); `stratum-ffhq` (public preprocessing).
4. DWPose (Yang et al., 2023); Sapiens / Sapiens2 (Meta, 2024–2025); AuraFace (ArcFace-lineage recognition embeddings); DINOv3 (Oquab et al., 2023–2025).
5. EPnP: Lepetit et al., 2009; 300W ibug 68-point canonical face model.
6. FLAME: Li et al., 2017; GIF: Ghosh et al., 2020; IP-Adapter/Face for appearance conditioning context.
