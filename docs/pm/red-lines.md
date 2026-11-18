# Red Lines — eidolon (Founder-Owned)

This file is owned exclusively by the founder (Tim). Agents may read it, never edit it.
Any suggestion or experiment arm that violates a rule below is ruled out forever.

1. **No Re-Attempting Discontinued Volumetric / Bridge Fast Paths:**
   The following paths are permanently discontinued (`docs/DISCONTINUATION_NOTICE_*.md`):
   - `z_d` monocular depth (`DISCONTINUATION_NOTICE_zd_depth.md`)
   - `z_a` surface normals (`DISCONTINUATION_NOTICE_za_normals.md`)
   - `DINOv3 -> slider` linear/MLP bridge (`DISCONTINUATION_NOTICE_dino_bridge.md`)
   - `z_g` norm > 25 filtering (`DISCONTINUATION_NOTICE_zg_norm_filter.md`)
   - FLAME 3DMM shape parameters (`beta`) and `text -> z_g` Prior (`Phase 5a-exp1`).
2. **No Mixed-Basis Conditioning:**
   Every consumed identity corpus (`FFHQ`, `hegre_corpus`) must pass `basis-fingerprint verify` (`120e1c5a1dc4f423`). Loaders must fail closed on mixed or unverified basis fingerprints.
3. **No Uncommitted or Gitignored Evidence Citations:**
   Every headline number in `docs/02_EXPERIMENTS_AND_RESULTS.md` or `docs/briefings/` must trace to a git-tracked artifact (under `docs/assets/`) and an immutable tagged 40-hex commit SHA (`tests/tools/test_governance_sync.py`).
4. **Public-Surface Privacy Rules (`lawrenz.com` / HF Hub):**
   - Public briefings and articles use the public project tag **`face-vectors`**; never write the internal repo name `eidolon` or the proprietary dataset name (`Hegre`) in public prose (refer to it only as *"the proprietary multi-shoot dataset, used exclusively for cross-shoot validation"*).
   - Never include any image from the proprietary multi-shoot dataset in a public figure or commit to `lawrenz.com`. Every face image in a public figure must come from **FFHQ** with `<small>` CC BY-NC-SA 4.0 attribution listing the exact FFHQ image IDs.
5. **No Bespoke Crons or Systemd Timers:**
   The legacy nightly LDA reprojection audit cron is permanently retired. All work is scheduled via `ds4-queue` and bounded GPU runs via `bin/gpu_run` (`gpu-scheduler`).
