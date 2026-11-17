"""Protocol v3 scoring, identical for every method (protocol_v3.md: Readout regression, Metrics, and the 2026-10-06 deviation:
label-selected parts and a readout menu for every method).
Inputs: v3/parts/<method>_U<U>_K<K>.npz (part centres in 512 px crop coordinates; handoff_competitors.md), v3/readout_crops.json,
paper_protocol.json (test crops, test_secondary order), the official landmarks. Landmarks: 5 of the 68 -- left eye centre (mean of
36-41), right eye centre (42-47), nose tip (30), mouth corners (48, 54) -- in 512 px crop coordinates; IOD = |eye centre distance|.
Parts, per method and U (an ordered list; K = the first K):
  pool methods (dino, dff): pool = their K = 4, 8, 16, 32 parts together (60); greedy forward selection of 32 by 5-fold CV NME of
           a ridge readout (lambda 1, standardised) on the first 500 readout faces, crop 0 -- the same faces, folds and code as
           proto_gpu/select_parts.py, which selected ours.
  preselected methods (ours = label-free size x spread [appendix]; ourssel = label-selected from all clusters; ourssel60 = from the
           60 largest): the K = 32 file, already in order.
Readout menu, chosen per method and U together with K in {4, 8, 16, 32} by 5-fold CV NME on all 2000 readout faces (folds by face):
  regression: ridge, 2K part centres + bias -> 10 numbers, lambda by CV; missing parts -> that part's mean over the training crops.
  vote:       per landmark, the m parts with the lowest median error (part + its median offset to the landmark, on training crops)
              each vote; prediction = the median of the votes (missing parts do not vote); m in {1,2,3,5,8,12,16,24,32}, m <= K,
              chosen inside the CV.
Test (scored once, for the chosen readout and K): NME (%), PCK@0.25 / 0.1, per-landmark NME; 95 % bootstrap intervals over test
crops (2000 resamples, seed 0), and paired against position only. All (readout, K) dev and test numbers go to scores.json.
position only: the mean readout landmarks (aligned 512 px) shifted into crop coordinates by the mean test offset.
saturation: the chosen readout refitted on the first 500 / 1000 / 2000 readout faces in a fixed shuffled order (seed 2).
Run from the code folder:  python score_v3.py [--parts v3/parts] [--methods ...] [--dummy]"""
import argparse, glob, json, os, re, warnings, numpy as np, paper_common as pc
warnings.filterwarnings('ignore', category=RuntimeWarning)          # all-NaN medians / means of mostly-missing parts are handled explicitly
ap = argparse.ArgumentParser(); ap.add_argument('--parts', default='v3/parts'); ap.add_argument('--out', default='v3/scores')
ap.add_argument('--methods', nargs='*'); ap.add_argument('--dummy', action='store_true'); args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
POOL_METHODS, KS, MS = ('dino', 'dinohard', 'dinoallfg', 'dff'), (4, 8, 16, 32), (1, 2, 3, 5, 8, 12, 16, 24, 32)
LMI = [list(range(36, 42)), list(range(42, 48)), [30], [48], [54]]; NAMES = ['left eye', 'right eye', 'nose tip', 'left mouth corner', 'right mouth corner']
def lm5(f):
    lm = np.array(pc.meta()[str(f)]['image']['face_landmarks'], float)[:, ::-1]           # (68, 2) as (y, x), 1024 px
    return np.stack([lm[i].mean(0) for i in LMI])
RC = json.load(open('v3/readout_crops.json'))['faces']; PRO = json.load(open('paper_protocol.json'))['data']
TF = [int(f) for f in PRO['test_secondary']]; OFF = PRO['test_crop']['offsets']
def to_crop(p, x, y): return (p - np.array([y, x], float)) / 2                              # 1024 aligned -> 512 px crop coords
RL = np.stack([[to_crop(lm5(r['face']), x, y) for x, y in r['crops']] for r in RC])          # (2000, 2, 5, 2)
TL = np.stack([to_crop(lm5(f), *OFF[str(f)]) for f in TF])                                   # (1000, 5, 2)
IOD = np.linalg.norm(TL[:, 0] - TL[:, 1], axis=1)
ALIGNED = np.stack([lm5(r['face']) / 2 for r in RC])                                         # (2000, 5, 2), aligned 512 px
MEAN_OFF = np.array([OFF[str(f)][::-1] for f in TF], float).mean(0) / 2                      # mean test offset (y, x), 512 px
FOLD = np.arange(len(RC)) % 5; np.random.default_rng(3).shuffle(FOLD)                       # 5 folds by face (readout)
SAT = np.random.default_rng(2).permutation(len(RC))                                          # saturation order
LAMBDAS = [0.0] + list(10.0 ** np.arange(-3, 5)); ALL = np.arange(len(RC))
def iod_of(L): return np.linalg.norm(L[..., 0, :] - L[..., 1, :], axis=-1)
def nme(P, T): return (np.linalg.norm(P - T, axis=-1) / iod_of(T)[:, None]).mean(-1) * 100  # per crop, %

# ---------------- selection (pool methods): identical to proto_gpu/select_parts.py
def select32(R):
    NF = 500; X0 = R[:NF, 0]; Y = RL[:NF, 0]; IO = iod_of(Y); F5 = np.arange(NF) % 5; np.random.default_rng(3).shuffle(F5)
    def cv(cols):
        err = []
        for k in range(5):
            tr, te = F5 != k, F5 == k; Xc = X0[:, cols]; fill = safe_mean(Xc[tr], Xc)
            X = np.where(np.isnan(Xc), fill, Xc).reshape(NF, -1); mx, s = X[tr].mean(0), X[tr].std(0) + 1e-9; my = Y[tr].reshape(-1, 10).mean(0)
            A = (X[tr] - mx) / s; W = np.linalg.solve(A.T @ A + np.eye(A.shape[1]), A.T @ (Y[tr].reshape(-1, 10) - my))
            err.append(np.linalg.norm((((X[te] - mx) / s) @ W + my).reshape(-1, 5, 2) - Y[te], axis=-1).mean(1) / IO[te])
        return float(np.concatenate(err).mean() * 100)
    chosen = []
    while len(chosen) < 32:
        chosen.append(min((j for j in range(R.shape[2]) if j not in chosen), key=lambda j: cv(chosen + [j])))
    return chosen

# ---------------- readout (a): ridge regression
def fit(X, Y, lam):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; s = Xc.std(0) + 1e-9
    W = np.linalg.solve((Xc / s).T @ (Xc / s) + lam * np.eye(X.shape[1]), (Xc / s).T @ Yc)
    return lambda Z: ((Z - mx) / s) @ W + my
def safe_mean(A, A_all):
    """Per-part mean over training rows, falling back to the mean over all rows, then 0, for parts missing everywhere."""
    with np.errstate(all='ignore'), __import__('warnings').catch_warnings():
        __import__('warnings').simplefilter('ignore', RuntimeWarning); m = np.nanmean(A, 0); g = np.nanmean(A_all.reshape(-1, *A.shape[1:]), 0)
    return np.nan_to_num(np.where(np.isnan(m), g, m))
def design(parts, fill): P = np.where(np.isnan(parts), fill, parts); return P.reshape(*P.shape[:-2], -1)
def reg_fit_predict(R, Te, tr, lam):
    fill = safe_mean(R[tr].reshape(-1, *R.shape[2:]), R); K = R.shape[2]
    f = fit(design(R[tr], fill).reshape(-1, 2 * K), RL[tr].reshape(-1, 10), lam); return f(design(Te, fill)).reshape(-1, 5, 2)
def reg_cv(R, faces):
    def cv(lam):
        return float(np.concatenate([nme(reg_fit_predict(R, R[faces[FOLD[faces] == k]].reshape(-1, *R.shape[2:]), faces[FOLD[faces] != k], lam),
                                         RL[faces[FOLD[faces] == k]].reshape(-1, 5, 2)) for k in range(5)]).mean())
    lam = min(LAMBDAS, key=cv); return dict(lam=lam), cv(lam)

# ---------------- readout (b): median vote
def vote_fit(R, tr):
    P = R[tr].reshape(-1, R.shape[2], 2); L = RL[tr].reshape(-1, 5, 2)
    off = np.nanmedian(L[:, None] - P[:, :, None], axis=0)                                   # (K, 5, 2)
    err = np.nanmedian(np.linalg.norm(P[:, :, None] + off[None] - L[:, None], axis=-1) / iod_of(L)[:, None, None], axis=0)   # (K, 5)
    return off, np.argsort(np.where(np.isnan(err), np.inf, err), axis=0), L.mean(0)
def vote_predict(model, Te, m):
    off, rank, mean = model; out = np.empty((len(Te), 5, 2))
    for l in range(5):
        idx = rank[:m, l]; v = Te[:, idx] + off[idx, l][None]                               # (n, m, 2)
        with np.errstate(all='ignore'): med = np.nanmedian(v, axis=1)
        out[:, l] = np.where(np.isnan(med), mean[l], med)
    return out
def vote_cv(R, faces):
    Ms = [m for m in MS if m <= R.shape[2]]; acc = {m: [] for m in Ms}
    for k in range(5):
        tr, te = faces[FOLD[faces] != k], faces[FOLD[faces] == k]; model = vote_fit(R, tr); Te = R[te].reshape(-1, *R.shape[2:])
        for m in Ms: acc[m].append(nme(vote_predict(model, Te, m), RL[te].reshape(-1, 5, 2)))
    dev = {m: float(np.concatenate(v).mean()) for m, v in acc.items()}; m = min(dev, key=dev.get); return dict(m=m), dev[m]

def predict(kind, prm, R, Te, faces):
    return reg_fit_predict(R, Te, faces, prm['lam']) if kind == 'regression' else vote_predict(vote_fit(R, faces), Te, prm['m'])
def boot(v, ref=None, reps=2000):
    r = np.random.default_rng(0); idx = r.integers(0, len(v), (reps, len(v))); d = v if ref is None else v - ref
    m = d[idx].mean(1); return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
def metrics(P, ref=None):
    e = np.linalg.norm(P - TL, axis=-1) / IOD[:, None]; per = e.mean(1) * 100
    out = dict(NME=boot(per), PCK25=float((e <= .25).mean()), PCK10=float((e <= .1).mean()),
               per_landmark_NME={n: float(e[:, i].mean() * 100) for i, n in enumerate(NAMES)})
    if ref is not None: out['NME_minus_position_only'] = boot(per, ref)
    return out, per

def load(p):
    z = np.load(p); R, T = z['readout'].astype(float), z['test'].astype(float); R[z['readout_missing']] = np.nan; T[z['test_missing']] = np.nan
    return R, T
pos_pred = np.broadcast_to(ALIGNED.mean(0) - MEAN_OFF, TL.shape); POS, pos_per = metrics(pos_pred)
results = {'position only': POS}; runs = {}
for p in sorted(glob.glob(f'{args.parts}/*_U*_K*.npz')):
    m, U, K = re.match(r'(.+)_U(\d+)_K(\d+)\.npz', os.path.basename(p)).groups()
    if args.methods and m not in args.methods: continue
    runs.setdefault((m, int(U)), {})[int(K)] = p
lines = [f'position only: NME {POS["NME"][0]:.2f} % [{POS["NME"][1]:.2f}, {POS["NME"][2]:.2f}], PCK@0.25 {POS["PCK25"]:.3f}, PCK@0.1 {POS["PCK10"]:.3f}']
todo = []
for (m, U), byK in sorted(runs.items()):
    if m in POOL_METHODS:
        if sorted(byK) != list(KS): lines.append(f'{m} U={U}: skipped, needs all of K = {KS} for its pool (has {sorted(byK)})'); continue
        parts = [load(byK[K]) for K in KS]; R = np.concatenate([p[0] for p in parts], 2); T = np.concatenate([p[1] for p in parts], 1)
        order = select32(R); todo.append((m, U, R[:, :, order], T[:, order], f'label-selected 32 of a pool of {R.shape[2]}'))
    elif 32 in byK:
        R, T = load(byK[32]); todo.append((m, U, R, T, 'preselected order (K = 32 file)'))
if args.dummy:
    r = np.random.default_rng(9); todo.append(('dummy (landmarks + 5 px noise, 5 parts)', 0, RL + r.normal(0, 5, RL.shape), TL + r.normal(0, 5, TL.shape), 'dummy'))
for m, U, R, T, how in todo:
    menu = {}
    for K in [k for k in KS if k <= R.shape[2]]:
        for kind, cvf in (('regression', reg_cv), ('vote', vote_cv)):
            prm, dv = cvf(R[:, :, :K], ALL); menu[(kind, K)] = (prm, dv)
    kind, K = min(menu, key=lambda c: menu[c][1]); prm, dv = menu[(kind, K)]
    res, per = metrics(predict(kind, prm, R[:, :, :K], T[:, :K], ALL), pos_per)
    res.update(parts=how, readout=kind, K=K, params=prm, dev_NME=round(dv, 3),
               all_dev_test={f'{c[0]} K={c[1]}': [round(v[1], 2), round(metrics(predict(c[0], v[0], R[:, :, :c[1]], T[:, :c[1]], ALL))[0]['NME'][0], 2)] for c, v in menu.items()},
               saturation={n: metrics(predict(kind, prm, R[:, :, :K], T[:, :K], np.sort(SAT[:n])))[0]['NME'][0] for n in (500, 1000, 2000)})
    results[f'{m} U={U}'] = res
    lines.append(f'{m} U={U} [{how}]: {kind} K={K} {prm} (dev {dv:.2f}), test NME {res["NME"][0]:.2f} % [{res["NME"][1]:.2f}, {res["NME"][2]:.2f}], '
                 f'vs position only {res["NME_minus_position_only"][0]:+.2f} [{res["NME_minus_position_only"][1]:+.2f}, {res["NME_minus_position_only"][2]:+.2f}], '
                 f'PCK@0.25 {res["PCK25"]:.3f}, PCK@0.1 {res["PCK10"]:.3f}; saturation (500/1000/2000) ' + '/'.join(f'{v:.2f}' for v in res['saturation'].values()))
    lines.append('    per landmark NME: ' + ', '.join(f'{n} {v:.2f}' for n, v in res['per_landmark_NME'].items()))
    lines.append('    every (readout, K): dev / test NME  ' + ', '.join(f'{c} {v[0]:.2f}/{v[1]:.2f}' for c, v in res['all_dev_test'].items()))
json.dump(results, open(f'{args.out}/scores.json', 'w'), indent=1); open(f'{args.out}/scores.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
