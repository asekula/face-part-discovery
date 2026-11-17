"""Dense GPU scan of cluster detectors over an image (the readout / test step), with a correctness check and a timing.
Score at every position = the method's pooled match: per level, per box, cosine of the box's descriptor (oriented gradient and
LoG, each group divided by max(its norm, energy floor)) with the template, best over shifts of up to +-RB level px minus the
spring 0.05 (d / RB)^2; mean over the level's boxes; mean over levels. Computed for all positions at once with convolutions;
level maps are brought to the level-1 grid. Frozen node only (single boxes, RF = 0 at every level).
  check: on a member's own face, the scan's value at the member keypoint vs the pooled match computed directly at that point.
  time : K detectors on a 768 x 768 crop (a 384 px test crop at 512 px, in the node's 1024 px frame); extrapolated.
Run on the desktop from this folder:  FFHQ_DIR=... python scan_gpu.py results/U300_single_r1024 [K]"""
import sys, os, time, pickle, numpy as np, scipy.ndimage as ndi, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_common as pc
RES = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 32
D = pickle.load(open(f'{RES}/clusters.pkl', 'rb')); C, KPS, NODE, FLOOR, SPRING = D['clusters'], D['KPS'], D['NODE'], D['FLOOR'], D['SPRING']
BOXL = D.get('BOXL', {s: D['BOX'] for s in NODE}); LEVELS = tuple(NODE); dev = 'cuda'
assert all(NODE[s]['RF'] == 0 and len(NODE[s]['sub']) == 1 for s in LEVELS), 'frozen (single-box) node only'
RES_PX = D.get('res', 1024)

def planes(L):
    """{level: GPU tensor (3, H_s, W_s)} -- the method's filter planes (as proto_leader_gpu.face_planes)."""
    if RES_PX < L.shape[0]:
        f = L.shape[0] / RES_PX; L = ndi.zoom(ndi.zoom(ndi.gaussian_filter(L, .5 * np.sqrt(f * f - 1)), 1 / f, order=1), f, order=1)
    out = {}
    for s in LEVELS:
        Ls = L if s == 1 else ndi.zoom(ndi.gaussian_filter(L, .5 * np.sqrt(s * s - 1)), 1 / s, order=1)
        q = ndi.gaussian_filter(Ls, 1.0); gx, gy = ndi.sobel(q, 1), ndi.sobel(q, 0); m, th = np.hypot(gx, gy), np.arctan2(gy, gx)
        out[s] = torch.tensor(np.stack([m * np.cos(2 * th), m * np.sin(2 * th), -9.0 * ndi.gaussian_laplace(Ls, 3.0)]).astype(np.float32), device=dev)
    return out

def scan(P, ids, H1, W1):
    """(len(ids), H1, W1) score maps on the level-1 grid for clusters ids."""
    total = torch.zeros(len(ids), H1, W1, device=dev)
    for s in LEVELS:
        bx = BOXL[s]; hb = bx // 2; RB = NODE[s]['RB']; fo, fl = FLOOR[s]; feat = NODE[s]['feat']; nF = len(feat)
        T = torch.tensor(np.stack([C[i]['template'][s][:, 0] for i in ids]), device=dev, dtype=torch.float32)   # (K, F, 3*bx*bx)
        tn = T.norm(dim=2).clamp_min(1e-9).reshape(-1)
        To = T[..., :2 * bx * bx].reshape(-1, 2, bx, bx); Tl = T[..., 2 * bx * bx:].reshape(-1, 1, bx, bx)
        X = P[s][None]; H, W = X.shape[2:]
        no = (F.avg_pool2d((X[:, :2] ** 2).sum(1, keepdim=True), bx, 1, hb, count_include_pad=True) * bx * bx).sqrt()[0, 0]
        nl = (F.avg_pool2d(X[:, 2:] ** 2, bx, 1, hb, count_include_pad=True) * bx * bx).sqrt()[0, 0]
        cos = (F.conv2d(X[:, :2], To, padding=hb)[0] / no.clamp_min(fo) + F.conv2d(X[:, 2:], Tl, padding=hb)[0] / nl.clamp_min(fl)) \
              / (np.sqrt(2) * tn[:, None, None])                                             # (K*F, H, W): cos at each box centre
        pad = F.pad(cos, (RB, RB, RB, RB), value=-9.0)                                     # best over shifts minus the spring,
        row = torch.full((cos.shape[0], H + 2 * RB, W), -9.0, device=dev)                  # in two passes (the spring is
        for vx in range(-RB, RB + 1): row = torch.maximum(row, pad[:, :, RB + vx:RB + vx + W] - SPRING * vx * vx / RB ** 2)   # a sum of
        best = torch.full_like(cos, -9.0)                                                   # vertical and horizontal parts:
        for vy in range(-RB, RB + 1): best = torch.maximum(best, row[:, RB + vy:RB + vy + H] - SPRING * vy * vy / RB ** 2)   # same result)
        best = best.reshape(len(ids), nF, H, W); lvl = torch.zeros(len(ids), H, W, device=dev)
        bpad = F.pad(best, (64, 64, 64, 64), value=-9.0)
        for f_, (fy, fx) in enumerate(feat):                                                # box f sits at centre + offset
            oy, ox = int(round(fy / s)), int(round(fx / s)); lvl += bpad[:, f_, 64 + oy:64 + oy + H, 64 + ox:64 + ox + W]
        lvl /= nF
        up = lvl if s == 1 else lvl.repeat_interleave(s, 1).repeat_interleave(s, 2)          # centre y -> round(y / s): nearest
        total += F.pad(up, (0, max(0, W1 - up.shape[2]), 0, max(0, H1 - up.shape[1])), value=-9.0)[:, :H1, :W1] / len(LEVELS)
    return total

def pooled_at(P, i, y, x):
    """The method's pooled match of cluster i at (y, x), computed directly (no convolutions)."""
    sc = 0.0
    for s in LEVELS:
        bx = BOXL[s]; hb = bx // 2; RB = NODE[s]['RB']; fo, fl = FLOOR[s]; T = C[i]['template'][s][:, 0]; vals = []
        for f_, (fy, fx) in enumerate(NODE[s]['feat']):
            t = torch.tensor(T[f_], device=dev, dtype=torch.float32); t = t / t.norm().clamp_min(1e-9); best = -9.0
            for vy in range(-RB, RB + 1):
                for vx in range(-RB, RB + 1):
                    cy, cx = int(round((y + fy) / s)) + vy, int(round((x + fx) / s)) + vx
                    w = P[s][:, cy - hb:cy + hb + 1, cx - hb:cx + hb + 1]
                    if w.shape[1:] != (bx, bx): continue
                    o, l = w[:2].reshape(-1), w[2].reshape(-1)
                    u = torch.cat([o / o.norm().clamp_min(fo), l / l.norm().clamp_min(fl)]) / np.sqrt(2)
                    best = max(best, float(t @ u) - SPRING * (vy * vy + vx * vx) / RB ** 2)
            vals.append(best)
        sc += np.mean(vals) / len(LEVELS)
    return sc

# ---------------- check: scan vs direct pooled match at member keypoints
big = sorted(range(len(C)), key=lambda i: -C[i]['n'])[:3]
print('check (cluster, face): scan at the keypoint vs direct pooled match')
for i in big:
    k = C[i]['members'][0][0]; fi, y, x = KPS[k]; P = planes(pc.lightness(pc.rgb(fi)))
    m = scan(P, [i], 1024, 1024)[0]; yi, xi = int(round(y)), int(round(x))
    print(f'  {i:4d} {fi:6d}: scan {float(m[yi, xi]):.4f}  direct {pooled_at(P, i, y, x):.4f}  (scan argmax {np.unravel_index(int(m.argmax()), m.shape)}, keypoint ({yi}, {xi}))')

# ---------------- time: K detectors on a 768 x 768 crop
ids = sorted(range(len(C)), key=lambda i: -C[i]['n'])[:K]; fi = KPS[C[ids[0]]['members'][0][0]][0]
L = pc.lightness(pc.rgb(fi))[128:896, 128:896]
t0 = time.time(); P = planes(L); t1 = time.time()
torch.cuda.synchronize(); t2 = time.time(); m = scan(P, ids, 768, 768); torch.cuda.synchronize(); t3 = time.time()
for _ in range(2): m = scan(P, ids, 768, 768)
torch.cuda.synchronize(); t4 = time.time(); per = (t4 - t3) / 2
print(f'\nK = {K} detectors on one 768 x 768 crop: filter planes {t1 - t0:.2f} s (CPU); scan {per:.2f} s (GPU; first call {t3 - t2:.2f} s)')
print(f'readout + test = 5000 crops: about {(5000 * (per + t1 - t0)) / 60:.0f} min per (U, K) setting for K = {K} '
      f'(planes are shared across K, so per U: {5000 * (t1 - t0) / 60:.0f} min planes + scans)')
