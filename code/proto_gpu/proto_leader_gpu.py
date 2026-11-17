"""GPU (PyTorch) version of the discovery prototype (proto_leader2.py), for U up to 1000 on the desktop.

Same method as proto_leader2.py:
  node    level 1: 12 features (3 x 4, 32 px apart) x 2 x 2 sub-boxes of 19 px 16 px apart, two-level pooling (feature +-4,
          box +-2 px); levels 2 / 4 / 8: 12 (3x4) / 6 (2x3, 44 px apart) / 1 boxes, each pooled +-4 level px. Every box is 19 x 19
          pixels at its level; raw filter values (orientation cos2t, sin2t at sigma 1; signed LoG sigma 3); each filter group
          divided by max(its norm, energy floor = 20th percentile of box norms on unlabelled faces). Spring 0.05 at both levels.
  cluster leader clustering; shortlist = 20 leaders highest on the level-8 box (unpooled), exact pooled match on those; a
          keypoint joins the best leader whose match is >= TAU and that has no member from its face, else starts a new cluster;
          leaders are running aligned means; then one reassignment pass to the clusters of >= 3.
Changes from proto_leader2.py:
  - TAU = 99.9th percentile (was 99th) of pooled matches between random keypoints of different faces, from 5000 pairs
    ("margin rule": the chance bar moves from 1 in 100 to 1 in 1000).
  - keypoints are processed face by face (faces in random order, keypoints within a face in random order), so each face's
    filter maps go to the GPU once.
Run on the desktop from this folder (torch venv):
  FFHQ_DIR=../ffhq_nvlabs python -u proto_leader_gpu.py --U 1000 --out results/U1000"""
import argparse, os, sys, json, time, pickle
import numpy as np, scipy.ndimage as ndi, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_common as pc

ap = argparse.ArgumentParser()
ap.add_argument('--U', type=int, default=1000); ap.add_argument('--out', default='results/U1000')
ap.add_argument('--tau_q', type=float, default=99.9); ap.add_argument('--null_pairs', type=int, default=5000)
ap.add_argument('--keypoints', default='keypoints_all.json')
ap.add_argument('--level1', choices=('sub', 'single', 'none'), default='single',
                help="level 1 as 2 x 2 sub-boxes per feature (sub), one 19 px box per feature pooled +-4 like level 2 (single), or no level 1 (none)")
ap.add_argument('--box1', type=int, default=19, help='box size at level 1, in level-1 px (odd)')
ap.add_argument('--box2', type=int, default=19, help='box size at level 2, in level-2 px (odd)')
ap.add_argument('--edge_r', type=float, default=0, help="SIFT edge rejection of keypoints with this curvature ratio (0 = off)")
ap.add_argument('--res', type=int, default=1024, help='reduce each face to this size, then enlarge back to 1024 (node unchanged)')
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
BOX, HB, SPRING, SHORT = 19, 9, 0.05, 20
HOME = (np.array([(dy, dx) for dy in (-8, 0, 8) for dx in (-12, -4, 4, 12)]) * 4).astype(float)
SUB4 = np.array([(-8, -8), (-8, 8), (8, -8), (8, 8)], float)
NODE = {1: dict(feat=HOME, sub=SUB4, RF=4, RB=2) if args.level1 == 'sub' else dict(feat=HOME, sub=np.zeros((1, 2)), RF=0, RB=4),
        2: dict(feat=HOME, sub=np.zeros((1, 2)), RF=0, RB=4),
        4: dict(feat=np.array([(dy, dx) for dy in (-22, 22) for dx in (-44, 0, 44)], float), sub=np.zeros((1, 2)), RF=0, RB=4),
        8: dict(feat=np.zeros((1, 2)), sub=np.zeros((1, 2)), RF=0, RB=4)}
if args.level1 == 'none': del NODE[1]
LEVELS = tuple(NODE)
BOXL = {s: {1: args.box1, 2: args.box2}.get(s, BOX) for s in LEVELS}             # box size per level, in that level's px
PAD = max(BOXL.values()) // 2 + 8; D = 3 * BOX * BOX                              # D: the level-8 descriptor (shortlist)
rng = np.random.default_rng(0); t0 = time.time()
def log(*a): print(f'[{time.time() - t0:7.0f}s]', *a, flush=True)
log('resolution', args.res, 'level 1', args.level1, 'box1', args.box1, 'box2', args.box2); log('device', dev, torch.cuda.get_device_name(0) if dev == 'cuda' else '')

# ---------------- faces and keypoints
U = [int(f) for f in pc.u_draw(1000, 0)[:args.U]]
KPJ = json.load(open(args.keypoints))['log32']; g1 = pc.grid(1024)[0]
KPS = [(fi, float(g1[r]), float(g1[c])) for fi in U for r, c in KPJ[str(fi)]]
if args.edge_r > 0:                                                               # SIFT's edge rejection (paper_common.edge_ok)
    n0 = len(KPS); keep = []
    for fi in U:
        mine = [k for k in KPS if k[0] == fi]
        if mine: keep += [k for k, ok in zip(mine, pc.edge_ok(pc.lightness(pc.rgb(fi)), [k[1:] for k in mine], r=args.edge_r)) if ok]
    KPS = keep; log(f'edge rejection r = {args.edge_r}: kept {len(KPS)} of {n0} keypoints')
BYFACE = {}
for k, (fi, _, _) in enumerate(KPS): BYFACE.setdefault(fi, []).append(k)
N = len(KPS); log(f'{N} keypoints on {len(U)} faces')
CACHE = {}
def face_planes(fi):
    """{level: GPU tensor (3, H, W)} for face fi (kept for the face currently being processed)."""
    if fi not in CACHE:
        CACHE.clear(); L = pc.lightness(pc.rgb(fi)); out = {}
        if args.res < L.shape[0]:                                                         # detail lost at --res, sizes kept at 1024
            f = L.shape[0] / args.res; L = ndi.zoom(ndi.zoom(ndi.gaussian_filter(L, .5 * np.sqrt(f * f - 1)), 1 / f, order=1), f, order=1)
        for s in LEVELS:
            Ls = L if s == 1 else ndi.zoom(ndi.gaussian_filter(L, .5 * np.sqrt(s * s - 1)), 1 / s, order=1)
            q = ndi.gaussian_filter(Ls, 1.0); gx, gy = ndi.sobel(q, 1), ndi.sobel(q, 0); m, th = np.hypot(gx, gy), np.arctan2(gy, gx)
            P = np.stack([m * np.cos(2 * th), m * np.sin(2 * th), -9.0 * ndi.gaussian_laplace(Ls, 3.0)]).astype(np.float32)
            out[s] = torch.tensor(np.pad(P, ((0, 0), (PAD, PAD), (PAD, PAD))), device=dev)
        CACHE[fi] = out
    return CACHE[fi]

# ---------------- energy floor (unlabelled faces, random boxes)
FLOOR = {s: ([], []) for s in LEVELS}
for fi in U[:50]:
    Pf = face_planes(fi)
    for s in LEVELS:
        H = Pf[s].shape[1] - 2 * PAD
        hb = BOXL[s] // 2
        for y, x in rng.integers(hb + 1, H - hb - 1, (200, 2)):
            b = Pf[s][:, y + PAD - hb:y + PAD + hb + 1, x + PAD - hb:x + PAD + hb + 1]
            FLOOR[s][0].append(float(b[:2].norm())); FLOOR[s][1].append(float(b[2:].norm()))
FLOOR = {s: (float(np.percentile(a, 20)), float(np.percentile(b, 20))) for s, (a, b) in FLOOR.items()}
log('energy floor:', {s: tuple(round(v, 3) for v in FLOOR[s]) for s in LEVELS})

def patches(k):
    """Per level: GPU tensor (F, B, W*W, D) of unit descriptors of every box at every displacement within +-(RF+RB) level px."""
    fi, y, x = KPS[k]; Pf = face_planes(fi); out = {}
    for s, c in NODE.items():
        R = c['RF'] + c['RB']; P = Pf[s]; fo, fl = FLOOR[s]; wins = []; bx = BOXL[s]; hb = bx // 2
        for f in c['feat']:
            for b in c['sub']:
                cy, cx = int(round((y + f[0] + b[0]) / s)) + PAD, int(round((x + f[1] + b[1]) / s)) + PAD
                wins.append(P[:, cy - hb - R:cy + hb + R + 1, cx - hb - R:cx + hb + R + 1])
        Wn = torch.stack(wins)                                                            # (F*B, 3, 19+2R, 19+2R)
        u = Wn.unfold(2, bx, 1).unfold(3, bx, 1)                                          # (F*B, 3, W, W, bx, bx)
        u = u.permute(0, 2, 3, 1, 4, 5).reshape(len(wins), -1, 3, bx * bx)                # (F*B, W*W, 3, bx*bx)
        o = u[:, :, :2].reshape(*u.shape[:2], -1); l = u[:, :, 2]
        o = o / o.norm(dim=2, keepdim=True).clamp_min(fo); l = l / l.norm(dim=2, keepdim=True).clamp_min(fl)
        out[s] = (torch.cat([o, l], 2) / np.sqrt(2)).reshape(len(c['feat']), len(c['sub']), -1, 3 * bx * bx)
    return out
def centre(s): c = NODE[s]; R = c['RF'] + c['RB']; return R * (2 * R + 1) + R
SPRING_V = {}; SPRING_U = {}
for s, c in NODE.items():
    RB, RF = c['RB'], c['RF']
    SPRING_V[s] = [(vy, vx, SPRING * (vy * vy + vx * vx) / RB ** 2 if RB else 0.0) for vy in range(-RB, RB + 1) for vx in range(-RB, RB + 1)]
    uy, ux = torch.meshgrid(torch.arange(-RF, RF + 1), torch.arange(-RF, RF + 1), indexing='ij')
    SPRING_U[s] = (SPRING * (uy ** 2 + ux ** 2).float() / RF ** 2).to(dev) if RF else torch.zeros(1, 1, device=dev)
def pooled(Pk, T):
    """Two-level pooled match of keypoint patches Pk with templates T: dict s -> tensor (nT, F, B, D).
    Returns scores (nT,) and per level the best displacement index (nT, F, B)."""
    nT = next(iter(T.values())).shape[0]; sc = torch.zeros(nT, device=dev); best = {}
    for s, c in NODE.items():
        RF, RB = c['RF'], c['RB']; Wd = 2 * (RF + RB) + 1; U_ = 2 * RF + 1
        Tn = T[s] / T[s].norm(dim=3, keepdim=True).clamp_min(1e-9)
        cos = torch.einsum('tfbd,fbwd->tfbw', Tn, Pk[s]).reshape(nT, *T[s].shape[1:3], Wd, Wd)
        boxbest = torch.full((nT, *T[s].shape[1:3], U_, U_), -9.0, device=dev); boxarg = torch.zeros(boxbest.shape, dtype=torch.long, device=dev)
        for vy, vx, cost in SPRING_V[s]:
            cand = cos[..., RB + vy:RB + vy + U_, RB + vx:RB + vx + U_] - cost
            upd = cand > boxbest; boxbest = torch.where(upd, cand, boxbest); boxarg[upd] = (vy + RB) * (2 * RB + 1) + (vx + RB)
        fu = boxbest.mean(2) - SPRING_U[s]
        flat = fu.reshape(nT, fu.shape[1], -1); mx, ui = flat.max(2); sc += mx.mean(1) / len(LEVELS)
        uy_, ux_ = ui // U_, ui % U_
        vi = torch.gather(boxarg.reshape(nT, *T[s].shape[1:3], -1), 3, ui[:, :, None, None].expand(-1, -1, T[s].shape[2], 1))[..., 0]
        vy_, vx_ = vi // (2 * RB + 1), vi % (2 * RB + 1)
        dy = (uy_[:, :, None] - RF) + (vy_ - RB); dx = (ux_[:, :, None] - RF) + (vx_ - RB)
        best[s] = (dy + RF + RB) * Wd + (dx + RF + RB)
    return sc, best

# ---------------- TAU from the null (random cross-face pairs)
log(f'null from {args.null_pairs} random cross-face pairs ...')
pairs = []
while len(pairs) < args.null_pairs:
    a, b = rng.integers(0, N, 2)
    if KPS[a][0] != KPS[b][0]: pairs.append((int(a), int(b)))
cent = {}
for a in sorted({a for a, _ in pairs}, key=lambda k: KPS[k][0]):
    Pa = patches(a); cent[a] = {s: Pa[s][:, :, centre(s)].clone() for s in LEVELS}
null = []
for b in sorted({b for _, b in pairs}, key=lambda k: KPS[k][0]):
    Pb = patches(b); As = [a for a, bb in pairs if bb == b]
    sc, _ = pooled(Pb, {s: torch.stack([cent[a][s] for a in As]) for s in LEVELS}); null += sc.tolist()
TAU = float(np.percentile(null, args.tau_q))
log(f'TAU = {TAU:.3f} ({args.tau_q} th percentile of {len(null)} null matches; 99th {np.percentile(null, 99):.3f}, median {np.median(null):.3f})')
del cent

# ---------------- leader clustering + reassignment (templates kept on the CPU, 20 candidates moved per keypoint)
def order_faces(seed):
    r = np.random.default_rng(seed); out = []
    for fi in r.permutation(U): out += list(r.permutation(BYFACE[int(fi)]))
    return out
def run_pass(order, leaders=None):
    fixed = leaders is not None
    if fixed:
        TT = [{s: L['sum'][s] / L['n'] for s in LEVELS} for L in leaders]
        Rv = torch.tensor(np.stack([t[8].ravel() / max(np.linalg.norm(t[8]), 1e-9) for t in TT]), device=dev)
        new = [dict(sum={s: np.zeros_like(leaders[0]['sum'][s]) for s in LEVELS}, rsum={s: np.zeros_like(leaders[0]['sum'][s]) for s in LEVELS},
                    n=0, faces=set(), members=[]) for _ in leaders]
    else:
        new = []; Rv = torch.zeros((0, D), device=dev)
    for i, k in enumerate(order):
        fi = KPS[k][0]; Pk = patches(k); joined = False
        v8 = Pk[8][:, :, centre(8)].reshape(-1); v8 = v8 / v8.norm().clamp_min(1e-9)
        if len(Rv):
            cand = torch.topk(Rv @ v8, min(SHORT, len(Rv))).indices.tolist()
            Tc = {s: torch.tensor(np.stack([(TT[j][s] if fixed else new[j]['sum'][s] / new[j]['n']) for j in cand]), device=dev) for s in LEVELS}
            sc, best = pooled(Pk, Tc); sc = sc.cpu().numpy(); best = {s: best[s].cpu().numpy() for s in LEVELS}
            for jj in np.argsort(-sc):
                j = cand[jj]
                if sc[jj] < TAU: break
                if fi in new[j]['faces']: continue
                for s in LEVELS:
                    Fn, Bn = Pk[s].shape[:2]
                    idx = torch.tensor(best[s][jj], device=dev)
                    new[j]['sum'][s] += Pk[s][torch.arange(Fn, device=dev)[:, None], torch.arange(Bn, device=dev)[None, :], idx].cpu().numpy()
                    new[j]['rsum'][s] += Pk[s][:, :, centre(s)].cpu().numpy()             # same member, boxes at their fixed places
                new[j]['n'] += 1; new[j]['faces'].add(fi); new[j]['members'].append((int(k), {s: best[s][jj].tolist() for s in LEVELS}, float(sc[jj])))
                if not fixed:
                    v = new[j]['sum'][8].ravel() / new[j]['n']; Rv[j] = torch.tensor(v / max(np.linalg.norm(v), 1e-9), device=dev)
                joined = True; break
        if not joined and not fixed:
            new.append(dict(sum={s: Pk[s][:, :, centre(s)].cpu().numpy().copy() for s in LEVELS},
                            rsum={s: Pk[s][:, :, centre(s)].cpu().numpy().copy() for s in LEVELS}, n=1, faces={fi},
                            members=[(int(k), {s: np.full(Pk[s].shape[:2], centre(s)).tolist() for s in LEVELS}, 1.0)]))
            Rv = torch.cat([Rv, v8[None]])
        if i % 2000 == 1999: log(f'{"reassign" if fixed else "leader"} pass: {i + 1}/{N}, {len(new)} clusters')
    return new
P1 = run_pass(order_faces(1)); sizes = sorted((L['n'] for L in P1), reverse=True)
log(f'pass 1: {len(P1)} clusters; 10 largest {sizes[:10]}; singletons {sum(s == 1 for s in sizes)}')
keep = [L for L in P1 if L['n'] >= 3]
P2 = [L for L in run_pass(order_faces(2), keep) if L['n'] >= 3]
sizes = sorted((L['n'] for L in P2), reverse=True); log(f'pass 2: {len(P2)} clusters; 10 largest {sizes[:10]}')
pickle.dump(dict(edge_r=args.edge_r, res=args.res, level1=args.level1, U=U, KPS=KPS, NODE=NODE, BOX=BOX, BOXL=BOXL, SPRING=SPRING, TAU=TAU, tau_q=args.tau_q, null=null, FLOOR=FLOOR,
                 clusters=[dict(n=L['n'], members=L['members'], template={s: L['sum'][s] / L['n'] for s in LEVELS},
                                template_unaligned={s: L['rsum'][s] / L['n'] for s in LEVELS}) for L in P2]),
            open(f'{args.out}/clusters.pkl', 'wb'))
log(f'saved {args.out}/clusters.pkl')
