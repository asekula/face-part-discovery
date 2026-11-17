# Interpretable Face Part Discovery Without Gradient Descent

**Paper:** [parts.pdf](parts.pdf) (Alexander Sekula, manuscript, 2026)

Code and evaluation data for the paper. Face parts are discovered from unlabelled FFHQ faces with hand-built multi-level
part detectors (oriented gradients and LoG, max-pooled boxes), leader clustering and aligned running means: no
pretraining and no gradient descent anywhere. The full protocol, including every deviation, is in
[code/docs/protocol_v3.md](code/docs/protocol_v3.md).

## Layout

| Path | What |
|---|---|
| `code/paper_common.py`, `code/paper_protocol.json` | data access (FFHQ images, landmarks), filters, keypoints; the fixed protocol (faces, crops) |
| `code/download_ffhq.py` | downloads the FFHQ images needed (checks MD5s) into `code/ffhq_nvlabs/` |
| `code/proto_gpu/proto_leader_gpu.py` | discovery: keypoints → leader clustering → part detectors (`clusters.pkl`) |
| `code/proto_gpu/run_ours.py`, `scan_gpu.py` | detection: dense scan of part detectors on the readout and test crops |
| `code/proto_gpu/select_parts.py` | label selection: greedy choice of 32 parts by cross-validated NME (same rule for every method) |
| `code/proto_gpu/run_dino.py`, `run_dff.py` | the DINO co-parts and DFF baselines (fit on U, assign to crops) |
| `code/proto_gpu/run_v3_*.sh` | the queues that produced the paper's runs |
| `code/score_v3.py` | readout (ridge / median vote, chosen by 5-fold CV) and test NME / PCK → `code/v3/scores/` |
| `code/v3/readout_crops.json` | the 2000 readout faces' crops |
| `code/v3/parts/` | part locations on every readout and test crop for each method in Table 1 |
| `code/results/v3_U*_r512/parts_labelsel*.json` | the parts label selection chose |
| `figure_scripts/` | Figures 1–5 (need `clusters.pkl`; Fig. 3 needs `code/proto_gpu/fig3_pick.py` first) |
| `CREDITS.md` | author and licence of every FFHQ face shown in the figures |

## Reproduce Table 1 from the included part locations

```bash
pip install -r requirements.txt
# put ffhq-dataset-v2.json (landmarks; from github.com/NVlabs/ffhq-dataset) in code/ffhq_nvlabs/
cd code && python score_v3.py             # writes v3/scores/scores.txt
```

Scoring needs only the landmark metadata. Discovery, detection and the figures also need the images:
`python download_ffhq.py` fetches the faces in `paper_protocol.json` (train pool and test) and checks their MD5s.
Discovery writes to `code/proto_gpu/results/`; the figure scripts read `code/results/v3_U1000_r512/`, so copy the run there.

## Re-run discovery and detection

`clusters.pkl` (about 180 MB per run) is not included. From `code/proto_gpu/`:

```bash
bash run_v3_ours.sh          # discovery at U = 100 and 1000 (512 px), about 1 h on a GTX 1080 Ti for U = 1000
bash run_v3_ours_sel.sh      # scan every cluster on 500 readout faces, label selection, detection
```

## Licences

Code: Apache 2.0 (`LICENSE`). FFHQ (Karras et al., CVPR 2019) is CC BY-NC-SA 4.0 by NVIDIA; its images are under their
Flickr authors' licences (see `CREDITS.md`). The part-location files are derived from FFHQ and are for non-commercial use.
# face-part-discovery
