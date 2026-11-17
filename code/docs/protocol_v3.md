# Protocol v3: unsupervised face-part discovery at matched resolution

Written 2026-09-30, **before any v3 run**; revised the same day, still before any run: readout on 2000 held-out labelled faces (no small-L results); U = 30 added; our K-part selection rule fixed; K swept over {4, 8, 16, 32} with each method's K chosen on development data; paper layout simplified to one table and one figure; level 16 removed from our node; PCA basis from up to 300 faces of U. 2026-10-01, still before any run: our method replaced by the new node (no PCA, levels 1/2/4/8 with sub-boxes at level 1, leader clustering; see **Ours**), so the level-16 and PCA notes above apply to the superseded 518 version. Any change after a run is listed under *Deviations* at the end, with the reason.

## Question and claim

**Question.** On high-resolution faces, how close does part discovery without gradient descent or pretraining (ours) come to part-discovery methods built on networks pretrained on over a million images, when every method sees the same images at the same resolution and is scored the same way?

**What we may claim** (wording rule, fixed now): with `gain(m) = NME(position only) − NME(m)`, at U = 1000, each method at its own best K (chosen on development data, below), readout fitted on the 2000 labelled faces:
- if `gain(ours) ≥ 0.8 × max over baselines of gain(baseline)`, we may say ours "comes close";
- otherwise we report the numbers and the ratio, without "close".

The ratio and its 95 % interval are reported for every U either way, and the full results for every K.

## Labels: where they are used

- **Discovery uses no labels**, for any method.
- **Labels enter at one step only, identically for every method:** the readout regression that maps each method's discovered parts to landmarks.
- **The readout uses many labels, so that it is not the bottleneck:** it is fitted on 2000 labelled faces, held out from discovery (discovery on one set, regression labels from another, as in SCOPS).
- **Saturation check:** for every method, the readout is also fitted on 500 and 1000 of those faces. If NME barely changes from 1000 to 2000, more labels would not help, and the differences between methods come from their parts. Reported in the appendix.
- **No small-L results.** With few labels a method could lose because the regression lacked data, not because its parts were worse, and the two can't be told apart.

## Data

- **Images:** official NVlabs FFHQ (`ffhq_nvlabs/`, MD5-verified), aligned 1024 px, **reduced to half resolution (512 px) for every method.** The eyes are then about 128 px apart. This is the largest size at which every method runs on one 11 GB GPU.
- **Unlabelled set U:** U ∈ {30, 100, 300, 1000}, draw 0 of `paper_protocol.json`'s training pool, the same faces as the v2 runs.
- **Readout faces:** the 2000 training-pool faces that are not in U = 1000 (draw 0), with their official landmarks. The same 2000 are used at every U. Every method locates its parts on them by inference only. The saturation check uses the first 500 and 1000 of them, in a fixed shuffled order (seed 2).
- **Test:** the 1000-image `test_secondary` subset, each a random crop with the v2 offsets (`test_crop`), reduced to half resolution: 384 × 384 crops at the same scale as the training images.
- **Landmarks:** 5, from the official 68 points: left eye centre (mean of points 36–41), right eye centre (42–47), nose tip (30), left mouth corner (48), right mouth corner (54).
- **IOD:** the distance between the two eye centres.

## Methods

Every method gets the same U images at 512 px and outputs **K part centres per image**, for K ∈ {4, 8, 16, 32}.

**Choosing K (per method, per U, without test data):** each method's K is the one with the lowest *development* NME: the readout regression's 5-fold cross-validated NME on the 2000 readout faces (fit on 4/5, score on 1/5; crops as in the readout). The test crops play no part in the choice. Every method gets the same four candidates, so the same tuning budget. The chosen K is reported for every method and U, and all four K are reported in full.

**Ours (new method, replacing the 518 detector; decided 2026-10-01 before any v3 run).** Prototype: `code/proto_gpu/proto_leader_gpu.py`.
- **Node (frozen 2026-10-05, see Deviations).** Levels 1 / 2 / 4 / 8: 12 (3 x 4, 32 px apart) / 12 (3 x 4, 32 px apart) / 6 (2 x 3, 44 px apart) / 1 boxes of 19 / 38 / 76 / 152 px. Every box is 19 x 19 pixels at its level and is pooled +-4 level px, with a spring cost 0.05 (moving a box its full range costs 0.05). (`proto_leader_gpu.py --level1 single`; was: level 1 as 2 x 2 sub-boxes per feature with two-level pooling.)
- **Box descriptor.** Raw filter values, no PCA: gradient orientation (cos 2t, sin 2t; sigma 1) and signed LoG (sigma 3) at each level; each filter group divided by max(its norm, energy floor), the floor being the 20th percentile of box norms on unlabelled faces.
- **Match.** Per box, best cosine over its shifts minus the spring; per feature, best over the feature's shifts of the mean of its boxes minus the spring; mean over a level's features; mean over levels.
- **Discovery.** sigma-32 LoG keypoints; leader clustering: keypoints face by face (faces and keypoints in random order); shortlist of the 20 leaders highest on the level-8 box (unpooled), exact pooled match on those; a keypoint joins the best leader whose match is at least TAU and that has no member from its face, else it starts a new cluster; leaders are running aligned means. TAU = 99.9th percentile of pooled matches between random keypoints of different faces (5000 pairs). Then one reassignment pass to the clusters of at least 3 members.
- **Part model per cluster.** The aligned mean of its members' boxes (each member's box taken where it matched). Stored next to it, as a control for the figures: the unaligned mean (boxes at their fixed places).
- **Parts (label-free rule; amended 2026-10-05, see Deviations).** Clusters ranked by size x orientation spread (spread: 1 - |sum of the template's double-angle gradient vectors| / sum of their lengths, per level, averaged over levels; ~0 for a single edge or contour, towards 1 for edges in many directions); the top K, skipping duplicates (on at least half of their shared faces, within 48 px of an already kept one). K is matched across methods ({4, 8, 16, 32}). (Was: the largest clusters.)
- **Part centre in an image.** The cluster detector's best match (a dense scan).

**Required figure (user, 2026-10-01).** For the discovered parts: the members, and the part model at every level drawn directly as edges and signed blobs, aligned mean next to unaligned mean. The point: the aligned mean gives crisp outlines of real structures (lids, iris, brow), not blur, so what the model does can be read off the figure.

**DINO co-parts (Amir et al. 2021).** Official code `dino-vit-features/part_cosegmentation.py` with its defaults (ViT-S/8, key facet, layer 11, saliency threshold 0.065, votes 75 %, elbow 0.975, no crop augmentation), `num_parts = K`, `load_size = 512`.
- **Stride 8, not the default 4:** at 512 px, stride 4 (about 16k tokens) does not fit in 11 GB.
- **The parts are fitted on the U images only.** In the paper, the clustering is run on the evaluation images themselves (co-segmentation of each test set). Here the k-means is fitted on the same U unlabelled faces ours discovers from, and readout and test images are assigned afterwards, so that no method sees test images during discovery. Each pixel is assigned to the nearest learned part centre, using the code's own foreground (saliency) rule.
- **Part centre:** the weighted centroid of the part's assigned area (the SCOPS formula).

**DFF (Collins et al. 2018).** Official code; VGG-19 conv5_4 activations, as in the paper; NMF with k = K on the U images; input 512 px (feature map 32 × 32).
- **Test images:** activations projected onto the learned factors, by NMF with the factors fixed.
- **Part centre:** the heat-map-weighted centroid of each factor.
- **No CRF refinement:** the paper uses it only for segmentation masks, and centroids don't need it.

**Position only (floor).** The mean landmark positions over the 2000 readout faces, shifted into crop coordinates by the mean test offset (the v2 rule).

**Missing parts:** if a part has no assigned area in an image, its centre is replaced by that part's mean centre over the regression's training crops. This is applied identically to every method, and the rate is reported.

## Readout regression (identical for every method)

- **Inputs:** the K part centres (2K numbers) plus a bias; **outputs:** the 5 landmarks (10 numbers). Linear, as in SCOPS. At K = 32 that is 65 weights per output against 2000 faces, so still well determined.
- **Fitting:** least squares with a small ridge penalty (λ by 5-fold cross-validation on the readout faces) for numerical stability; it changes little with 2000 faces. The same rule applies to every method.
- **Training crops:** each readout face is cropped twice at random (the same crop generator as the test, seed 1, at half resolution), so the regression sees translated faces the way SCOPS's sees unaligned ones.

## Metrics

- **Primary: NME,** the mean over the 5 landmarks of |prediction − truth| / IOD, in %, averaged over the 1000 test crops.
- **Secondary:** PCK at 0.25 and 0.1 IOD (mean over the landmarks); per-landmark NME.
- **Uncertainty:** 95 % paired bootstrap over test crops (2000 resamples, seed 0).
- **Reported for every method:** images seen (pretraining + U), training on faces (none for all three), wall time per stage on one GTX 1080 Ti.

## What goes in the paper

**Main text: three sentences of setup, one table, one figure.**
- **Setup:** each method discovers K parts from U unlabelled faces at 512 px; a linear regression, fitted on 2000 separate labelled faces, maps each method's parts to the 5 landmarks, and we report NME on 1000 test crops; K is chosen for each method from {4, 8, 16, 32} by cross-validation on the labelled faces.
- **Table:** rows = ours, DINO co-parts, DFF, position only; columns = U = 30, 100, 300, 1000; each cell = test NME with its 95 % interval, the chosen K in small type; plus an images-seen column.
- **Figure:** each method's parts drawn on the same few faces.

**Appendix:** the saturation check (readout on 500 / 1000 / 2000 faces); test NME at every K and U, and NME against K at U = 1000; development vs. test NME; PCK; per-landmark errors; missing-part rates.

**Protocol only, not described in the paper:** the wording rule for "comes close" (top of this file).

## Not in v3

- Methods that train networks on faces (Thewlis, Zhang, Jakab, SCOPS): they need training runs of days, so they're stated as a limitation.
- Classic HOG patch discovery (Singh, Doersch): old code.

## Deviations

- **2026-10-05, node frozen before any v3 run.** Level 1 changed from 2 x 2 sub-boxes per feature (two-level pooling) to one 19 px box per feature pooled +-4, like the other levels. Chosen by a rule fixed before the runs: discovery at U = 300, 1024 px, three versions (no level 1 / single / sub-boxes); take the simplest whose part coverages and top-32 share are all within 5 points of the best (training faces only, `proto_gpu/compare_res.py`). Result (eye / brow / nose / mouth coverage, top-32 share): none 67 / 22 / 55 / 25 / 11 (nose -7: fails); single 67 / 27 / 62 / 26 / 16; sub 62 / 23 / 62 / 30 / 12. -> single. A further test of larger, half-overlapping level-1 / level-2 boxes (65 / 33 level px) lowered every number (eye -3.0, brow -7.3, nose -0.3, mouth -2.3, top-32 -4.9 points) -> not adopted. Resolution (512 vs 1024 px for ours), same comparison on the single version, U = 300: 512 vs 1024 eye -1.0, brow -7.7, nose -2.3, mouth +0.7, top-32 +0.1 points -> by the rule, 512 is not enough (brow). Every method still runs at 512 px (main table, unchanged); ours is also run at 1024 px and reported in the appendix as an extra.
- **2026-10-05, part selection amended before any v3 run.** Ranking clusters by size picked face contours (side of face, hairline, jaw): on U = 300 (single, 1024 px) the 16 largest kept clusters included no eye / brow / nose / mouth cluster, the first eye at rank 17 (`proto_gpu/select_check.py`). Contours cannot be pinned down along their length (aperture problem), so a label-free localisability measure was tested: orientation spread of the cluster template. Rule fixed before running (`proto_gpu/select_rules.py`, training faces' labels used only to score): adopt a new ranking only if it puts more eye / brow / nose / mouth clusters in the top 8 and the top 16 than size. Result, parts in the top 4 / 8 / 16 / 32: size 0 / 0 / 0 / 3; size x spread 1 / 1 / 3 / 9; size among spread >= median 1 / 1 / 2 / 8 -> size x spread. Known costs: elongated parts (brows, lip lines) rank low; contours, which locate the face well, are mostly dropped; high-spread non-parts (texture) are not filtered. K stays matched across methods; "all clusters as parts" was considered and rejected (unmatched K, cost).
- **2026-10-05, further label-free rankings tested; size x spread kept.** After size x spread, more rankings were tested on the U = 300 clusters (training faces' labels used only to score; parts among the top 8 / 16 / 32 on U300_single_r1024): SIFT edge rejection of keypoints before clustering, r = 5: 2 / 4 / 5, r = 3: 2 / 4 / 9 (`--edge_r`); configuration consistency (stable relative offsets to other clusters): 3 / 3 / 7; configured size: 3 / 3 / 5; size x per-box (level 1) orientation spread: 2 / 3 / 3; size x alignment gain: 0 / 0 / 3; grouping clusters by pooled template similarity (average linkage at TAU), groups ranked by faces covered: 0 / 0 / 2 (contour clusters merged, eyes did not) (`select_rules2.py`, `group_rules.py`, `cluster_similarity_pooled.py`). By the rule fixed for the edge-rejection test (beat size x spread at K = 8 and 16), edge rejection r = 5 would have been adopted; the gain was one cluster at each K (within run-to-run noise) and it lost four parts at K = 32. Decision (user): keep size x spread (1 / 3 / 9): simplest, no change to discovery, and the most parts at K = 32. No label-free ranking makes the top 16 mostly face parts; this is reported as a finding. Label-based selection was considered and rejected (unequal candidate pools vs DINO co-parts / DFF at their intended K).
- **2026-10-05, scope reduced before any v3 run** (deadline: paper on arXiv by about 2026-11-08). U in {100, 1000} (was {30, 100, 300, 1000}); compared methods: ours, DINO co-parts, position only (DFF dropped, stated as a limitation); no 1024 px appendix run for ours (the U = 300 coverage comparison is reported instead: 512 px costs a little, mostly on brows); no ablation table (aligned vs unaligned part models shown qualitatively in the gallery); no figure of every method's parts on the same faces. Unchanged: K in {4, 8, 16, 32} chosen by development NME, the regression readout, NME primary with PCK@0.25 / 0.1 and per-landmark NME secondary, the saturation check, the bootstrap intervals. Our runs: `proto_gpu/run_v3_ours.sh` (discovery at 512 px, `run_ours.py` for parts + detection); readout crops `v3/readout_crops.json` (`make_readout_crops.py`, test rule, seed 1).
- **2026-10-05 (late), extra analysis added before its results: label-selected parts for ours.** Our parts chosen with labels from ALL discovered clusters (141 at U = 100, about 600 at U = 1000): every cluster scanned on the first 500 readout faces (crop 0); greedy forward selection of 32 by 5-fold cross-validated NME of the readout regression on those faces (ridge lambda 1); K = 4 / 8 / 16 / 32 are the first K chosen; the chosen 32 then scanned on all readout and test crops and scored exactly as the main results (`proto_gpu/run_v3_ours_sel.sh`, `select_parts.py`, `run_ours.py --scan_all / --parts_json`; method name `ourssel`). Reported as a clearly labelled extra row, not the main comparison: the candidate pools differ (all our clusters vs DINO co-parts' K parts). The selection faces are among the readout faces, so the development NME used to choose K is slightly optimistic for this row; the test crops are untouched until scoring.
- **2026-10-06, main comparison changed to label-selected parts for every method (before any DINO or DFF result; after seeing ours' results).** Reason (user): the label-free ranking measures how well a heuristic picks parts, not how much landmark information a method's discovered parts contain. New main pipeline, identical for every method:
  1. **Pool** = the method's discovered parts. Ours: all clusters of the U run (141 at U = 100, about 600 at U = 1000). DINO co-parts and DFF: all parts of their four standard runs together, K = 4 + 8 + 16 + 32 = 60 candidates (no non-standard settings).
  2. **Selection:** greedy forward selection of 32 parts by 5-fold cross-validated NME of the ridge readout (lambda 1, inputs standardised), on the first 500 readout faces, crop 0 (the same faces and folds for every method; ours is limited to 500 faces by scan cost).
  3. **Readout menu**, chosen per method and U with K (first K of the selection, K in {4, 8, 16, 32}) by 5-fold CV NME on all 2000 readout faces: (a) ridge regression (as before); (b) median vote: per landmark, the m parts with the lowest CV error each vote "part location + median offset", the prediction is the median of the votes (m in {1, 2, 3, 5, 8, 12, 16, 24, 32}, m <= K, chosen inside the CV). Test crops are scored once, at the end.
  **DFF is back in** (user): VGG-19 conv5_4 at 512 px, NMF on the U faces, factors fixed for new crops, part centre = heat-map-weighted centroid; reimplemented in PyTorch (`proto_gpu/run_dff.py`).
  **Pool-matched row** for ours: pool = the 60 largest clusters (same size as DINO / DFF), same selection and readout. The label-free rows (size x spread) move to the appendix.
  Disclosed: before this decision the median vote was tried on ours (U = 100, label-free and label-selected parts) on the test crops (best-12 vote: 16.4 % and 13.7 % NME; regression 19.3 % and 17.9 %); m and the readout are now chosen on readout faces only.
- **2026-10-06, DINO co-parts part centres corrected (after seeing its first scores; the change can only help DINO).** The first run located each part at the centroid of its final hard CRF labels; with K = 32, 70-75 % of parts had no pixels on a test crop (27 % at K = 4), because the saliency step kept 1 of 6-7 first-stage clusters (about 15 % of each image) as foreground, so all K parts were cut from a small region. The protocol and the handoff specify the weighted centroid (SCOPS formula), so `run_dino.py` now computes each part's centre weighted by its CRF probability map (never missing): method `dino`, the main DINO row. The hard-label centroids are kept as `dinohard` (appendix). A sensitivity row `dinoallfg` treats the whole image as foreground (parts fitted on all descriptors, no background label), since faces fill the frame (appendix). Declared before these runs: the main row is `dino`. First-run scores (hard centroids, superseded): test NME 25.7 % (U = 100), 20.6 % (U = 1000).
- **2026-10-06, second pool-matched row added (after seeing the first).** The planned pool-matched row (ours choosing from the 60 *largest* clusters) scored 21.6 % NME at U = 100 (DFF: 14.8 %), plausibly because the largest clusters are mostly contours and hair. A second row is added in which the pool is the 60 clusters ranked highest by size x orientation spread (our label-free part ranking, adopted 2026-10-05), with the same label selection and scan (`run_v3_ours_pool60sxs.sh`, method `ourssel60sxs`). Both pool-matched rows are reported; the size-ranked one was the planned check.
- **2026-10-06, the size-ranked pool-matched row removed from the paper (user decision).** Its pool (the 60 *largest* clusters) used plain size, a ranking the protocol had already rejected on 2026-10-05 in favour of size x orientation spread, so it tested a configuration that was not our method. Results kept here for the record: 21.6 % (U = 100), 24.2 % (U = 1000) test NME. The paper reports one pool-matched row, the 60 clusters ranked highest by size x spread (14.6 % / 13.3 %), whose pool rule follows the method's own label-free ranking; that row was added after the size-ranked one had been scored (see the entry above).

- **2026-10-07, DINO co-parts row = the whole-image-as-foreground variant (`dinoallfg`), chosen after seeing all three DINO variants' test scores; the choice reports the strongest DINO.** Test NME U = 100 / 1000: official saliency (`dino`, soft CRF centroids) 31.0 / 24.6 %; hard labels (`dinohard`) 25.7 / 20.6 %; whole image as foreground 7.5 / 7.5 % (PCK@0.25 0.989 / 0.988, PCK@0.1 0.758 / 0.753; regression, K = 32 by CV). The official saliency step keeps about 15 % of each face, so most parts are missing; faces fill the frame, so treating the whole image as foreground is the setting suited to this data. The official-settings numbers are given in the table caption. All three variants are pooled and selected like `dino` (score_v3.py POOL_METHODS amended to include them before scoring). Unexplained: DINO's median-vote NME is identical across K (e.g. 67.80 at U = 100); regression was chosen for every DINO variant, so no reported number is affected.
