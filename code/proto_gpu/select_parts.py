"""Label selection of our parts from ALL discovered clusters (extra analysis; the main table stays label-free).
Input: <run>/scan_all_<N>.npz from `run_ours.py --scan_all N` (every cluster's best match on the first N readout faces, crop 0,
512 px crop coordinates). Greedy forward selection: start empty; repeatedly add the cluster whose addition gives the lowest 5-fold
cross-validated NME (folds by face; ridge, lambda 1, inputs standardised) of the readout regression on those N faces; stop at 32.
K = 4 / 8 / 16 / 32 are the first K chosen. Only readout faces' labels are used; test crops are not touched.
Output: <run>/parts_labelsel.json (ordered cluster ids) for `run_ours.py --parts_json`.
Optional third argument: pool size (the N top clusters; pool-matched row); fourth: pool ranking 'size' (default, planned)
or 'sxs' (size x orientation spread, added 2026-10-06 after the 'size' pool scored 21.6 % at U = 100). Output parts_labelsel_pool<N>[_sxs].json.
Run from this folder:  FFHQ_DIR=... python select_parts.py results/v3_U1000_r512 500 [60]"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_common as pc
RUN, NF = sys.argv[1], int(sys.argv[2]); POOL = int(sys.argv[3]) if len(sys.argv) > 3 else 0; t0 = time.time()
Z = np.load(f'{RUN}/scan_all_{NF}.npz'); POS, CL, FACES = Z['pos'].astype(float), Z['clusters'], Z['faces']   # (N, nC, 2)
RANK = sys.argv[4] if len(sys.argv) > 4 else 'size'                     # pool ranking: 'size' (planned) or 'sxs' (size x spread)
if POOL:                                                                  # pool-matched: only the POOL top clusters
    import pickle; CC = pickle.load(open(f'{RUN}/clusters.pkl', 'rb'))['clusters']
    def spread(t):
        out = []
        for s_, T in t.items():
            n = T.shape[-1] // 3; v = T[..., :2 * n].reshape(-1, 2, n)
            out.append(1 - np.linalg.norm(v.sum((0, 2))) / max(np.hypot(v[:, 0], v[:, 1]).sum(), 1e-12))
        return float(np.mean(out))
    key = (lambda i: CC[i]['n']) if RANK == 'size' else (lambda i: CC[i]['n'] * spread(CC[i]['template']))
    keep = sorted(range(len(CL)), key=lambda j: -key(int(CL[j])))[:POOL]; POS, CL = POS[:, keep], CL[keep]
RC = {r['face']: r['crops'][0] for r in json.load(open('../v3/readout_crops.json'))['faces']}
LMI = [list(range(36, 42)), list(range(42, 48)), [30], [48], [54]]
def lm5(f):
    lm = np.array(pc.meta()[str(f)]['image']['face_landmarks'], float)[:, ::-1]; return np.stack([lm[i].mean(0) for i in LMI])
Y = np.stack([(lm5(f) - np.array(RC[int(f)][::-1], float)) / 2 for f in FACES])        # (N, 5, 2), 512 px crop coords
IOD = np.linalg.norm(Y[:, 0] - Y[:, 1], axis=1); FOLD = np.arange(NF) % 5; np.random.default_rng(3).shuffle(FOLD)
def cv_nme(cols):
    X = POS[:, cols].reshape(NF, -1); err = []
    for k in range(5):
        tr, te = FOLD != k, FOLD == k; mx, s = X[tr].mean(0), X[tr].std(0) + 1e-9; my = Y[tr].reshape(-1, 10).mean(0)
        A = (X[tr] - mx) / s; W = np.linalg.solve(A.T @ A + np.eye(A.shape[1]), A.T @ (Y[tr].reshape(-1, 10) - my))
        P = (((X[te] - mx) / s) @ W + my).reshape(-1, 5, 2); err.append((np.linalg.norm(P - Y[te], axis=-1).mean(1) / IOD[te]))
    return float(np.concatenate(err).mean() * 100)
chosen, trace = [], []
while len(chosen) < 32:
    best = min((j for j in range(POS.shape[1]) if j not in chosen), key=lambda j: cv_nme(chosen + [j]))
    chosen.append(best); trace.append(round(cv_nme(chosen), 3))
    print(f'[{time.time() - t0:5.0f}s] {len(chosen):2d} parts: CV NME {trace[-1]:.2f} % (added cluster {int(CL[best])})', flush=True)
OUTF = f'{RUN}/parts_labelsel' + (f'_pool{POOL}' + ('' if RANK == 'size' else f'_{RANK}') if POOL else '') + '.json'
json.dump(dict(parts=[int(CL[j]) for j in chosen], cv_nme_trace=trace, n_faces=NF, pool=POOL or 'all', pool_rank=RANK if POOL else None,
               rule='greedy forward selection, 5-fold CV NME, ridge lambda 1'), open(OUTF, 'w'))
print('saved', OUTF)
