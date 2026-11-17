"""Protocol v3, our method: part selection + detection on the readout and test crops -> v3/parts/<method>_U<U>_K<K>.npz.
Parts: clusters ranked by size x orientation spread, the top 32 kept, skipping duplicates (within 48 px of a kept one on at least
half of their shared faces) -- protocol_v3.md, Parts. K = 4 / 8 / 16 / 32 are the first K of that list (one scan serves all K).
Detection: each part's detector scanned densely over the crop (the method's pooled match, scan_gpu.py), part centre = best match.
Images: the aligned 1024 px face reduced to --res (512 in the main table) and enlarged back (the node's sizes are in 1024 px),
then the 768 x 768 crop (384 at 512). Part centres are written in 512 px crop coordinates (handoff_competitors.md).
Readout crops: ../v3/readout_crops.json; test crops: paper_protocol.json test_crop offsets, test_secondary order.
Saves partial results every 200 faces and resumes from them. Run on the desktop from this folder:
  FFHQ_DIR=... python -u run_ours.py results/U300_single_r512 300 [--res 512] [--method ours]"""
import argparse, os, sys, json, time, pickle, numpy as np, scipy.ndimage as ndi, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_common as pc
ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('U', type=int)
ap.add_argument('--res', type=int, default=512); ap.add_argument('--method', default='ours'); ap.add_argument('--out', default='../v3/parts')
ap.add_argument('--crops', default='../v3/readout_crops.json')
ap.add_argument('--scan_all', type=int, default=0, help='scan EVERY cluster on the first N readout faces (crop 0), save, exit (for label selection)')
ap.add_argument('--parts_json', default='', help='scan these clusters (ordered list) instead of the size x spread ranking'); args = ap.parse_args()
os.makedirs(args.out, exist_ok=True); dev = 'cuda'; t_start = time.time()
def log(*a): print(f'[{time.time() - t_start:7.0f}s]', *a, flush=True)
D = pickle.load(open(f'{args.run}/clusters.pkl', 'rb')); C, KPS, NODE, FLOOR, SPRING = D['clusters'], D['KPS'], D['NODE'], D['FLOOR'], D['SPRING']
BOXL = D.get('BOXL', {s: D['BOX'] for s in NODE}); LEVELS = tuple(NODE)
assert all(NODE[s]['RF'] == 0 and len(NODE[s]['sub']) == 1 for s in LEVELS), 'frozen (single-box) node only'
assert D.get('res', 1024) == args.res, f"run was discovered at {D.get('res', 1024)} px, scanning at {args.res}"

# ---------------- parts: size x orientation spread, duplicates skipped, top 32
def spread(t):
    out = []
    for s, T in t.items():
        n = T.shape[-1] // 3; v = T[..., :2 * n].reshape(-1, 2, n)
        out.append(1 - np.linalg.norm(v.sum((0, 2))) / max(np.hypot(v[:, 0], v[:, 1]).sum(), 1e-12))
    return float(np.mean(out))
SP = np.array([spread(c['template']) for c in C]); N = np.array([c['n'] for c in C])
POS = [{KPS[m[0]][0]: np.array(KPS[m[0]][1:]) for m in c['members']} for c in C]
def dup(i, j):
    sh = POS[i].keys() & POS[j].keys()
    return bool(sh) and np.mean([np.linalg.norm(POS[i][f] - POS[j][f]) <= 48 for f in sh]) >= .5
PARTS = []
if args.scan_all: PARTS = list(range(len(C)))                                   # every cluster (label selection, step 1)
elif args.parts_json: PARTS = [int(i) for i in json.load(open(args.parts_json))['parts']]   # chosen elsewhere (step 3)
else:
    for i in np.argsort(-N * SP, kind='stable'):
        if len(PARTS) == 32: break
        if not any(dup(i, j) for j in PARTS): PARTS.append(int(i))
log(f'{len(C)} clusters; {len(PARTS)} scanned; first 32 (cluster, size, spread):', [(i, int(N[i]), round(SP[i], 2)) for i in PARTS[:32]])

# ---------------- dense scan (as scan_gpu.py, which checks it against the direct pooled match)
TPL = {}
for s in LEVELS:
    bx = BOXL[s]; T = torch.tensor(np.stack([C[i]['template'][s][:, 0] for i in PARTS]), device=dev, dtype=torch.float32)
    TPL[s] = (T[..., :2 * bx * bx].reshape(-1, 2, bx, bx), T[..., 2 * bx * bx:].reshape(-1, 1, bx, bx), T.norm(dim=2).clamp_min(1e-9).reshape(-1))
def planes(L):
    out = {}
    for s in LEVELS:
        Ls = L if s == 1 else ndi.zoom(ndi.gaussian_filter(L, .5 * np.sqrt(s * s - 1)), 1 / s, order=1)
        q = ndi.gaussian_filter(Ls, 1.0); gx, gy = ndi.sobel(q, 1), ndi.sobel(q, 0); m, th = np.hypot(gx, gy), np.arctan2(gy, gx)
        out[s] = torch.tensor(np.stack([m * np.cos(2 * th), m * np.sin(2 * th), -9.0 * ndi.gaussian_laplace(Ls, 3.0)]).astype(np.float32), device=dev)
    return out
@torch.no_grad()
def scan(P, H1, W1):
    return np.concatenate([scan_chunk(P, H1, W1, a, min(a + 32, len(PARTS))) for a in range(0, len(PARTS), 32)])
def scan_chunk(P, H1, W1, a, b):
    total = torch.zeros(b - a, H1, W1, device=dev)
    for s in LEVELS:
        bx = BOXL[s]; hb = bx // 2; RB = NODE[s]['RB']; fo, fl = FLOOR[s]; feat = NODE[s]['feat']; nF = len(feat)
        To, Tl, tn = (t[a * nF:b * nF] for t in TPL[s])
        X = P[s][None]; H, W = X.shape[2:]
        no = (F.avg_pool2d((X[:, :2] ** 2).sum(1, keepdim=True), bx, 1, hb, count_include_pad=True) * bx * bx).sqrt()[0, 0]
        nl = (F.avg_pool2d(X[:, 2:] ** 2, bx, 1, hb, count_include_pad=True) * bx * bx).sqrt()[0, 0]
        cos = (F.conv2d(X[:, :2], To, padding=hb)[0] / no.clamp_min(fo) + F.conv2d(X[:, 2:], Tl, padding=hb)[0] / nl.clamp_min(fl)) / (np.sqrt(2) * tn[:, None, None])
        pad = F.pad(cos, (RB, RB, RB, RB), value=-9.0)                                     # best over shifts minus the spring,
        row = torch.full((cos.shape[0], H + 2 * RB, W), -9.0, device=dev)                  # in two passes (the spring is
        for vx in range(-RB, RB + 1): row = torch.maximum(row, pad[:, :, RB + vx:RB + vx + W] - SPRING * vx * vx / RB ** 2)   # a sum of
        best = torch.full_like(cos, -9.0)                                                   # vertical and horizontal parts:
        for vy in range(-RB, RB + 1): best = torch.maximum(best, row[:, RB + vy:RB + vy + H] - SPRING * vy * vy / RB ** 2)   # same result)
        best = best.reshape(b - a, nF, H, W); lvl = torch.zeros(b - a, H, W, device=dev); bpad = F.pad(best, (64, 64, 64, 64), value=-9.0)
        for f_, (fy, fx) in enumerate(feat):
            oy, ox = int(round(fy / s)), int(round(fx / s)); lvl += bpad[:, f_, 64 + oy:64 + oy + H, 64 + ox:64 + ox + W]
        up = lvl / nF if s == 1 else (lvl / nF).repeat_interleave(s, 1).repeat_interleave(s, 2)
        total += F.pad(up, (0, max(0, W1 - up.shape[2]), 0, max(0, H1 - up.shape[1])), value=-9.0)[:, :H1, :W1] / len(LEVELS)
    flat = total.reshape(b - a, -1).argmax(1).cpu().numpy()
    return np.stack(np.unravel_index(flat, (H1, W1)), 1).astype(np.float32)          # (32, 2) as (y, x), 1024 px crop coords
def face_L(f):
    L = pc.lightness(pc.rgb(f))
    if args.res < 1024:                                                    # reduce to --res, enlarge back (the node's frame)
        r = 1024 / args.res; L = ndi.zoom(ndi.zoom(ndi.gaussian_filter(L, .5 * np.sqrt(r * r - 1)), 1 / r, order=1), r, order=1)
    return L
def detect(L, x, y): return scan(planes(L[y:y + 768, x:x + 768]), 768, 768) / 2           # -> 512 px crop coordinates

RC = json.load(open(args.crops))['faces']
if args.scan_all:                                                               # label selection, step 1: all clusters, N faces
    A = np.stack([detect(face_L(RC[n]['face']), *RC[n]['crops'][0]) for n in range(args.scan_all)])
    np.savez(f'{args.run}/scan_all_{args.scan_all}.npz', pos=A, clusters=np.array(PARTS), faces=np.array([RC[n]['face'] for n in range(args.scan_all)]))
    log(f'saved {args.run}/scan_all_{args.scan_all}.npz: {A.shape}'); sys.exit()
# ---------------- readout (2000 x 2) and test (1000) crops, with partial saves
TF = pc.test_faces('secondary'); OFF = json.load(open('paper_protocol.json'))['data']['test_crop']['offsets']
part_file = f'{args.out}/{args.method}_U{args.U}.partial.npz'
R = np.full((len(RC), 2, len(PARTS), 2), np.nan, np.float32); T = np.full((len(TF), len(PARTS), 2), np.nan, np.float32)
if os.path.exists(part_file): z = np.load(part_file); R, T = z['readout'], z['test']; log('resuming from', part_file)
jobs = [('r', n) for n in range(len(RC))] + [('t', n) for n in range(len(TF))]
for q, (kind, n) in enumerate(jobs):
    if (kind == 'r' and not np.isnan(R[n]).any()) or (kind == 't' and not np.isnan(T[n]).any()): continue
    if kind == 'r':
        L = face_L(RC[n]['face'])
        for c, (x, y) in enumerate(RC[n]['crops']): R[n, c] = detect(L, x, y)
    else:
        x, y = OFF[str(TF[n])]; T[n] = detect(face_L(TF[n]), x, y)
    if q % 200 == 199: np.savez(part_file, readout=R, test=T); log(f'{q + 1}/{len(jobs)} faces')
wall = time.time() - t_start
for K in (4, 8, 16, 32):
    meta = dict(method=args.method, U=args.U, K=K, res=args.res, run=args.run, parts=PARTS[:K],
                selection=f'labels: {args.parts_json}' if args.parts_json else 'size x orientation spread, duplicates skipped',
                wall_time_s=round(wall), code='proto_gpu/run_ours.py')
    np.savez(f'{args.out}/{args.method}_U{args.U}_K{K}.npz', readout=R[:, :, :K], test=T[:, :K],
             readout_missing=np.zeros(R[:, :, :K, 0].shape, bool), test_missing=np.zeros(T[:, :K, 0].shape, bool), meta=json.dumps(meta))
os.remove(part_file); log(f'saved {args.out}/{args.method}_U{args.U}_K{{4,8,16,32}}.npz')
