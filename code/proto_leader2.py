"""Prototype of the new discovery method on a small U (laptop, NumPy/SciPy) -- version 2.

Node (offsets in full-resolution px; every box is 19 x 19 pixels at its own level, stored as raw filter values):
  level 1: 12 features on a 3 x 4 grid (32 px apart), each 2 x 2 sub-boxes of 19 px, 16 px apart  -> 48 boxes
  level 2: 12 boxes of 38 px on the same 3 x 4 grid
  level 4:  6 boxes of 76 px, 2 x 3, 44 px apart
  level 8:  1 box of 152 px
Filters per level: gradient orientation (cos 2t, sin 2t; sigma 1) and signed LoG (sigma 3) -- one minimal box per scale.
Each filter group is divided by max(its norm, an energy floor) (20th percentile of box norms on unlabelled faces), so
low-texture boxes stay weak instead of being blown up. No PCA.
Match (two-level pooling, spring costs): per box, best cosine over its own shifts (+-RB level px); per feature, best over
the feature's shifts (+-RF) of the mean of its pooled boxes; spring cost SPRING * (d / range)^2 at both levels;
mean over a level's features; mean over levels.
Clustering: leader clustering; shortlist = the 20 leaders ranked highest on the level-8 box alone (unpooled), exact pooled
match on those; join the best above tau if the cluster has no member from this face, else start a new cluster; leaders are
running aligned means. tau = 99th percentile of pooled matches between random keypoints of different faces. Then one
reassignment pass to the clusters of >= 3.
Run from code/:  python3 proto_leader2.py [U] [out_dir]"""
import sys, os, json, time, pickle
import numpy as np, scipy.ndimage as ndi
sys.path.insert(0, '.')
import paper_common as pc

U_N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OUT = sys.argv[2] if len(sys.argv) > 2 else 'results/proto_leader_v2'; os.makedirs(OUT, exist_ok=True)
BOX, HB, SPRING, SHORT = 19, 9, 0.05, 20
HOME = (np.array([(dy, dx) for dy in (-8, 0, 8) for dx in (-12, -4, 4, 12)]) * 4).astype(float)
SUB4 = np.array([(-8, -8), (-8, 8), (8, -8), (8, 8)], float)
NODE = {1: dict(feat=HOME, sub=SUB4, RF=4, RB=2),
        2: dict(feat=HOME, sub=np.zeros((1, 2)), RF=0, RB=4),
        4: dict(feat=np.array([(dy, dx) for dy in (-22, 22) for dx in (-44, 0, 44)], float), sub=np.zeros((1, 2)), RF=0, RB=4),
        8: dict(feat=np.zeros((1, 2)), sub=np.zeros((1, 2)), RF=0, RB=4)}
LEVELS = tuple(NODE); PAD = HB + 8
rng = np.random.default_rng(0); t0 = time.time()
def log(*a): print(f'[{time.time() - t0:6.0f}s]', *a, flush=True)

# ---------------- faces, keypoints (the saved sigma-32 keypoints), filter planes
U = [int(f) for f in pc.u_draw(1000, 0)[:U_N]]
KPJ = json.load(open('results/node_detectors/ours_U1000_approx/keypoints_all.json'))['log32']
g1 = pc.grid(1024)[0]
def planes(L, s):
    Ls = L if s == 1 else ndi.zoom(ndi.gaussian_filter(L, .5 * np.sqrt(s * s - 1)), 1 / s, order=1)
    q = ndi.gaussian_filter(Ls, 1.0); gx, gy = ndi.sobel(q, 1), ndi.sobel(q, 0); m, th = np.hypot(gx, gy), np.arctan2(gy, gx)
    P = np.stack([m * np.cos(2 * th), m * np.sin(2 * th), -9.0 * ndi.gaussian_laplace(Ls, 3.0)]).astype(np.float32)
    return np.pad(P, ((0, 0), (PAD, PAD), (PAD, PAD)))
FACEP = {}; KPS = []
for i, fi in enumerate(U):
    L = pc.lightness(pc.rgb(fi)); FACEP[fi] = {s: planes(L, s) for s in LEVELS}
    for r, c in KPJ[str(fi)]: KPS.append((fi, float(g1[r]), float(g1[c])))
    if i % 25 == 24: log(f'filters: {i + 1}/{len(U)} faces')
N = len(KPS); log(f'{N} keypoints on {len(U)} faces')

def box_at(fi, s, y, x):
    P = FACEP[fi][s]; yy, xx = int(round(y)) + PAD, int(round(x)) + PAD
    return P[:, yy - HB:yy + HB + 1, xx - HB:xx + HB + 1]

# energy floor: 20th percentile of each filter group's box norm, per level, over random boxes of unlabelled faces
FLOOR = {}
for s in LEVELS:
    no, nl = [], []
    for fi in U[:min(50, len(U))]:
        H = FACEP[fi][s].shape[1] - 2 * PAD
        for y, x in rng.integers(HB + 1, H - HB - 1, (200, 2)):
            b = box_at(fi, s, y, x); no.append(np.linalg.norm(b[:2])); nl.append(np.linalg.norm(b[2:]))
    FLOOR[s] = (np.percentile(no, 20), np.percentile(nl, 20))
log('energy floor per level (orientation, LoG):', {s: tuple(round(float(v), 3) for v in FLOOR[s]) for s in LEVELS})
def unit(b, s):
    o, l = b[:2], b[2:]; fo, fl = FLOOR[s]
    return np.concatenate([(o / max(np.linalg.norm(o), fo)).ravel(), (l / max(np.linalg.norm(l), fl)).ravel()]) / np.sqrt(2)

def grid_shifts(R): return np.array([(dy, dx) for dy in range(-R, R + 1) for dx in range(-R, R + 1)])
from numpy.lib.stride_tricks import sliding_window_view
def patches(k):
    """Per level: (F, B, W*W, D) unit descriptors of every box at every displacement within +-(RF+RB) level px (vectorised)."""
    fi, y, x = KPS[k]; out = {}
    for s, c in NODE.items():
        R = c['RF'] + c['RB']; P = FACEP[fi][s]; fo, fl = FLOOR[s]; per_f = []
        for f in c['feat']:
            per_b = []
            for b in c['sub']:
                cy, cx = int(round((y + f[0] + b[0]) / s)) + PAD, int(round((x + f[1] + b[1]) / s)) + PAD
                win = P[:, cy - HB - R:cy + HB + R + 1, cx - HB - R:cx + HB + R + 1]           # (3, 19+2R, 19+2R)
                v = sliding_window_view(win, (BOX, BOX), axis=(1, 2))                           # (3, W, W, 19, 19)
                v = v.transpose(1, 2, 0, 3, 4).reshape(-1, 3, BOX * BOX)                          # (W*W, 3, 361)
                o = v[:, :2].reshape(len(v), -1); l = v[:, 2]
                o = o / np.maximum(np.linalg.norm(o, axis=1, keepdims=True), fo); l = l / np.maximum(np.linalg.norm(l, axis=1, keepdims=True), fl)
                per_b.append(np.concatenate([o, l], 1) / np.sqrt(2))
            per_f.append(np.stack(per_b))
        out[s] = np.stack(per_f).astype(np.float32)
    return out
def pooled(Pk, T):
    """Two-level pooled match of keypoint patches Pk with templates T (list of dict s -> (F, B, D)).
    Returns scores (nT,) and per level the best displacement index (nT, F, B) into the (RF+RB) grid."""
    sc = np.zeros(len(T)); best = {}
    for s, c in NODE.items():
        RF, RB = c['RF'], c['RB']; Wd = 2 * (RF + RB) + 1
        Ts = np.stack([t[s] for t in T]); Tn = Ts / np.maximum(np.linalg.norm(Ts, axis=3, keepdims=True), 1e-9)
        cos = np.einsum('tfbd,fbwd->tfbw', Tn, Pk[s]).reshape(len(T), *Ts.shape[1:3], Wd, Wd)
        U_ = 2 * RF + 1; boxbest = np.full((len(T), *Ts.shape[1:3], U_, U_), -9.0); boxarg = np.zeros(boxbest.shape, int)
        for vy in range(-RB, RB + 1):                                                    # box level: best own shift v, spring
            for vx in range(-RB, RB + 1):
                cost = SPRING * (vy * vy + vx * vx) / RB ** 2 if RB else 0.0
                cand = cos[..., RB + vy:RB + vy + U_, RB + vx:RB + vx + U_] - cost
                upd = cand > boxbest; boxbest[upd] = cand[upd]; boxarg[upd] = (vy + RB) * (2 * RB + 1) + (vx + RB)
        fu = boxbest.mean(2)                                                             # feature level: mean of its boxes
        if RF:
            uy, ux = np.meshgrid(np.arange(-RF, RF + 1), np.arange(-RF, RF + 1), indexing='ij'); fu = fu - SPRING * (uy ** 2 + ux ** 2) / RF ** 2
        flat = fu.reshape(len(T), fu.shape[1], -1); ui = flat.argmax(2); sc += flat.max(2).mean(1) / len(LEVELS)
        uy_, ux_ = np.divmod(ui, U_)
        vi = np.take_along_axis(boxarg.reshape(len(T), *Ts.shape[1:3], -1), ui[:, :, None, None].repeat(Ts.shape[2], 2), 3)[..., 0]
        vy_, vx_ = np.divmod(vi, 2 * RB + 1)
        dy = (uy_[:, :, None] - RF) + (vy_ - RB); dx = (ux_[:, :, None] - RF) + (vx_ - RB)
        best[s] = (dy + RF + RB) * Wd + (dx + RF + RB)
    return sc, best
def centre(s): c = NODE[s]; R = c['RF'] + c['RB']; return R * (2 * R + 1) + R

# ---------------- tau from a null
log('null for tau ...')
pairs = []
while len(pairs) < 500:
    a, b = rng.integers(0, N, 2)
    if KPS[a][0] != KPS[b][0]: pairs.append((a, b))
null = []
for a, b in pairs:
    Pa = patches(a); Ta = {s: Pa[s][:, :, centre(s)] for s in LEVELS}; null.append(pooled(patches(b), [Ta])[0][0])
TAU = float(np.percentile(null, 99)); log(f'tau = {TAU:.3f} (99th percentile of {len(null)} random cross-face pooled matches; median {np.median(null):.3f})')

# ---------------- leader clustering + reassignment
def short_vec(Pk): v = Pk[8][:, :, centre(8)].ravel(); return v / max(np.linalg.norm(v), 1e-9)
def run_pass(order, leaders=None):
    fixed = leaders is not None
    if fixed:
        T = [{s: L['sum'][s] / L['n'] for s in LEVELS} for L in leaders]
        Rv = np.stack([t[8].ravel() / max(np.linalg.norm(t[8]), 1e-9) for t in T])
        new = [dict(sum={s: np.zeros_like(leaders[0]['sum'][s]) for s in LEVELS}, n=0, faces=set(), members=[]) for _ in leaders]
    else:
        new = []; Rv = np.zeros((0, 3 * BOX * BOX), np.float32)
    for i, k in enumerate(order):
        fi = KPS[k][0]; Pk = patches(k); joined = False
        if len(Rv):
            cand = np.argsort(-(Rv @ short_vec(Pk)))[:SHORT]
            Tc = [T[j] for j in cand] if fixed else [{s: new[j]['sum'][s] / new[j]['n'] for s in LEVELS} for j in cand]
            sc, best = pooled(Pk, Tc)
            for jj in np.argsort(-sc):
                j = cand[jj]
                if sc[jj] < TAU: break
                if fi in new[j]['faces']: continue
                for s in LEVELS:
                    F, B = Pk[s].shape[:2]; new[j]['sum'][s] += Pk[s][np.arange(F)[:, None], np.arange(B)[None, :], best[s][jj]]
                new[j]['n'] += 1; new[j]['faces'].add(fi)
                new[j]['members'].append((int(k), {s: best[s][jj].tolist() for s in LEVELS}, float(sc[jj])))
                if not fixed:
                    v = (new[j]['sum'][8] / new[j]['n']).ravel(); Rv[j] = v / max(np.linalg.norm(v), 1e-9)
                joined = True; break
        if not joined and not fixed:
            new.append(dict(sum={s: Pk[s][:, :, centre(s)].copy() for s in LEVELS}, n=1, faces={fi},
                            members=[(int(k), {s: np.full(Pk[s].shape[:2], centre(s)).tolist() for s in LEVELS}, 1.0)]))
            Rv = np.vstack([Rv, short_vec(Pk)[None]])
        if i % 500 == 499: log(f'{"reassign" if fixed else "leader"} pass: {i + 1}/{N}, {len(new)} clusters')
    return new
P1 = run_pass(rng.permutation(N)); sizes = sorted((L['n'] for L in P1), reverse=True)
log(f'pass 1: {len(P1)} clusters; 10 largest {sizes[:10]}; singletons {sum(s == 1 for s in sizes)}')
keep = [L for L in P1 if L['n'] >= 3]
P2 = [L for L in run_pass(rng.permutation(N), keep) if L['n'] >= 3]
sizes = sorted((L['n'] for L in P2), reverse=True); log(f'pass 2: {len(P2)} clusters; 10 largest {sizes[:10]}')
pickle.dump(dict(U=U, KPS=KPS, NODE=NODE, BOX=BOX, SPRING=SPRING, TAU=TAU, null=null, FLOOR=FLOOR,
                 clusters=[dict(n=L['n'], members=L['members'], template={s: L['sum'][s] / L['n'] for s in LEVELS}) for L in P2]),
            open(f'{OUT}/clusters.pkl', 'wb'))
log(f'saved {OUT}/clusters.pkl')
