"""Shared, detector-independent parts of the paper evaluation (docs/protocol_v2.md / paper_protocol.json).

Pure numpy (plus scipy / PIL), importable from both venvs, so that every detector is scored by identical code:
data and draws, landmark labels, test crops, keypoints, clustering (average linkage with the one-per-face rule and the
permutation-null minimum size), naming, the primary metric (PCK of named detectors, part groups), the position-only
baseline, the secondary ridge-regression NME, and the paired bootstrap.
All positions are (y, x) in pixels of the image being processed: the aligned 1024 x 1024 image for training faces,
the 768 x 768 crop for test faces.
"""
import json, os
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

ROOT = Path(__file__).resolve().parent
FFHQ = Path(os.environ.get('FFHQ_DIR', ROOT / 'ffhq_nvlabs'))   # the official NVlabs FFHQ release (download_ffhq.py)
PROTO = json.loads((ROOT / 'paper_protocol.json').read_text())
_META = None
POINTS = ('eye_left', 'eye_right', 'brow_left', 'brow_right', 'nose_tip', 'mouth_centre')
PART_OF = {'eye_left': 'eye', 'eye_right': 'eye', 'brow_left': 'brow', 'brow_right': 'brow', 'nose_tip': 'nose', 'mouth_centre': 'mouth'}
CROP = PROTO['data']['test_crop']['size']
HB, GRID = 17, 4                                        # 518's grid: box half-width and 4 px step
NODE_M = 19                                             # 518's valid-area margin in grid cells (93 px from each edge at 1024)


def meta():
    global _META
    if _META is None: _META = json.loads((FFHQ / 'ffhq-dataset-v2.json').read_text())
    return _META


def image_path(i): return FFHQ / meta()[str(i)]['image']['file_path']


def crop_offset(i):
    """(x, y) offset of test image i's crop."""
    return tuple(PROTO['data']['test_crop']['offsets'][str(i)])


def rgb(i, crop=False):
    im = np.asarray(Image.open(image_path(i)).convert('RGB'))
    if crop:
        x, y = crop_offset(i); im = im[y:y + CROP, x:x + CROP]
    return np.ascontiguousarray(im)


def points(i, crop=False):
    """(6, 2) labelled points (y, x) in POINTS order; in crop coordinates when crop=True."""
    lm = np.array(meta()[str(i)]['image']['face_landmarks'], float)            # (68, 2) as (x, y)
    p = np.stack([lm[36:42].mean(0), lm[42:48].mean(0), lm[17:22].mean(0), lm[22:27].mean(0), lm[30], lm[48:60].mean(0)])[:, ::-1]
    if crop:
        x, y = crop_offset(i); p = p - np.array([y, x])
    return p


def iod(i):
    p = points(i); return float(np.linalg.norm(p[0] - p[1]))


# ---------------------------------------------------------------- draws
def u_draw(U, d):
    """Disjoint U draws from the training pool: one fixed permutation (seed 1), consecutive blocks."""
    perm = np.random.default_rng(PROTO['seeds']['U_draws']).permutation(PROTO['data']['train_pool'])
    assert (d + 1) * U <= len(perm)
    return [int(x) for x in perm[d * U:(d + 1) * U]]


def l_draws(Ulist, L_list, n_draws):
    """{(L, d): sorted list of L faces from Ulist}; one generator (seed 2) consumed in order."""
    rng = np.random.default_rng(PROTO['seeds']['L_draws']); out = {}
    for L, n in zip(L_list, n_draws):
        for d in range(n): out[L, d] = sorted(int(x) for x in rng.choice(Ulist, L, replace=False))
    return out


def test_faces(kind='primary'):
    return list(PROTO['data']['test_primary' if kind == 'primary' else 'test_secondary'])


# ---------------------------------------------------------------- lightness and keypoints (518)
def lightness(im):
    s = im.astype(np.float64) / 255; lin = np.where(s > .04045, ((s + .055) / 1.055)**2.4, s / 12.92)
    y = .21263901*lin[..., 0] + .71516868*lin[..., 1] + .07219232*lin[..., 2]
    f = np.where(y > .008856, np.cbrt(y), 7.787*y + 16/116); return ((116*f - 16) / 100).astype(np.float32)


def grid(size):
    """518's 4 px grid for an image of the given size, and its valid mask."""
    g = np.arange(HB, size - HB, GRID); n = len(g)
    ok = np.zeros((n, n), bool); ok[NODE_M:n - NODE_M, NODE_M:n - NODE_M] = True
    return g, ok


def valid_px(size):
    g, _ = grid(size); return g[NODE_M], g[len(g) - NODE_M - 1]


def keypoints_log32(L):
    """518's samples: extrema of the scale-normalised LoG at sigma 32, bright and dark, >= 16 cells apart, contrast > 20 % of the p99. Returns grid cells (row, col)."""
    g, ok = grid(L.shape[0])
    m = (-(32.0**2) * ndi.gaussian_laplace(L, 32.0))[g][:, g]; ref = np.percentile(np.abs(m), 99); out = []
    for sign in (1, -1):
        e = sign * m; loc = (e >= ndi.maximum_filter(e, size=33, mode='nearest')) & (e > 0.2 * ref) & ok
        out += [(int(y), int(x)) for y, x in np.argwhere(loc)]
    return out


def edge_ok(L, yx, r=10.0, sigma=32.0):
    """SIFT's edge-rejection test (Lowe 2004) at keypoints yx (image px, (row, col)): keep a point only if the Hessian of the
    image smoothed at sigma has both curvatures of the same sign and their ratio below r (tr^2 / det < (r + 1)^2 / r).
    Contours (strong curvature across, little along) fail; blobs and corners pass. Computed on the image shrunk 4x (sigma / 4)."""
    f = 4; Ls = ndi.zoom(ndi.gaussian_filter(L, .5 * np.sqrt(f * f - 1)), 1 / f, order=1); s = sigma / f
    Lyy, Lxx, Lxy = (ndi.gaussian_filter(Ls, s, order=o) for o in ((2, 0), (0, 2), (1, 1)))
    out = []
    for y, x in yx:
        i, j = min(int(round(y / f)), Ls.shape[0] - 1), min(int(round(x / f)), Ls.shape[1] - 1)
        tr, det = Lxx[i, j] + Lyy[i, j], Lxx[i, j] * Lyy[i, j] - Lxy[i, j] ** 2
        out.append(bool(det > 0 and tr * tr / det < (r + 1) ** 2 / r))
    return np.array(out, bool)


def keypoints_gradmax(L):
    """483's samples: local maxima (radius 4 cells) of gradient magnitude (sigma 2, Sobel) smoothed at sigma 16, in the top 40 % of that energy, >= 16 cells (64 px) apart, strongest first."""
    g, ok = grid(L.shape[0])
    sm = ndi.gaussian_filter(L, 2.0); E = ndi.gaussian_filter(np.hypot(ndi.sobel(sm, axis=1), ndi.sobel(sm, axis=0)), 16)[g][:, g].astype(np.float64)
    cand = (E >= ndi.maximum_filter(E, size=9, mode='nearest')) & (E >= np.quantile(E, .6)) & ok
    pts = np.argwhere(cand); pts = pts[np.argsort(-E[cand])]; keep = []
    for p in pts:
        if all(np.hypot(*(p - q)) >= 16 for q in keep): keep.append(p)
    return [(int(p[0]), int(p[1])) for p in keep]


def keypoints_random(size, n, rng):
    """n random valid grid cells, >= 16 cells (64 px) apart (the same spacing as the LoG keypoints), by rejection."""
    g, ok = grid(size); cells = np.argwhere(ok); keep = []
    for _ in range(100000):
        if len(keep) >= n: break
        p = cells[rng.integers(len(cells))]
        if all(np.hypot(*(p - q)) >= 16 for q in keep): keep.append(p)
    return [(int(p[0]), int(p[1])) for p in keep]


# ---------------------------------------------------------------- clustering (518) and the permutation null
def _to_xp(S, xp, rows=1024):
    """float32 copy of S in xp. For cupy from a host array: copied in row chunks, because one cp.asarray of an N x N host matrix makes CuPy
    pin (page-lock) a host staging buffer of the next power of two and keep it (8 GB at N = 35k, unreclaimable: systemd-oomd killed U = 1000)."""
    if xp is np or not isinstance(S, np.ndarray): D = xp.asarray(S, dtype=xp.float32); return D.copy() if D is S or xp is np else D
    D = xp.empty(S.shape, xp.float32)
    for r0 in range(0, len(S), rows): D[r0:r0 + rows] = xp.asarray(np.asarray(S[r0:r0 + rows], np.float32))
    return D

def average_linkage(S, face_of, tau, xp=np):
    """Average linkage on similarity S (N x N), never merging clusters that share a face; stops below tau. Returns member lists."""
    N = len(S); D = _to_xp(S, xp)                        # always a fresh copy: S stays intact
    fo = xp.asarray(face_of)
    for r0 in range(0, N, 4096): D[r0:r0 + 4096][fo[r0:r0 + 4096, None] == fo[None, :]] = -xp.inf      # same-face pairs, in row blocks (no N x N mask)
    size = np.ones(N); members = [[i] for i in range(N)]
    while True:
        k = int(D.argmax()); a, b = divmod(k, N); v = float(D[a, b])
        if not np.isfinite(v) or v < tau: break
        new = (size[a] * D[a] + size[b] * D[b]) / (size[a] + size[b])
        D[a] = new; D[:, a] = new; D[a, a] = -xp.inf; D[b] = -xp.inf; D[:, b] = -xp.inf
        size[a] += size[b]; members[a] += members[b]; members[b] = []
    return [m for m in members if m]


def average_linkage_unconstrained(S, tau, xp=np):
    """The same without the one-per-face rule (insight ablation): only self-pairs are excluded."""
    N = len(S); D = _to_xp(S, xp)
    xp.fill_diagonal(D, -xp.inf)
    size = np.ones(N); members = [[i] for i in range(N)]
    while True:
        k = int(D.argmax()); a, b = divmod(k, N); v = float(D[a, b])
        if not np.isfinite(v) or v < tau: break
        new = (size[a] * D[a] + size[b] * D[b]) / (size[a] + size[b])
        D[a] = new; D[:, a] = new; D[a, a] = -xp.inf; D[b] = -xp.inf; D[:, b] = -xp.inf
        size[a] += size[b]; members[a] += members[b]; members[b] = []
    return [m for m in members if m]


def tau_p99(S, face_of):
    same = face_of[:, None] == face_of[None, :]; return float(np.percentile(S[~same], 99))


def _upper_cross(face_of, r0, r1, N):
    """Mask of the cross-face, strictly-upper-triangle entries in rows r0:r1 (row-major order = np.triu_indices order)."""
    rows = np.arange(r0, r1)[:, None]; cols = np.arange(N)[None, :]
    return (cols > rows) & (face_of[r0:r1, None] != face_of[None, :])


def null_min_size(S, face_of, tau, reps=10, seed=0, xp=np, constrained=True, linkage=None):
    """1 + the largest cluster (in faces) over `reps` permutations of the cross-face similarities.
    Memory-light: the upper-triangle cross-face values are gathered and written back in row blocks, in np.triu_indices order,
    so the permutations are identical to the earlier all-at-once version (no N^2 index arrays). `linkage(P)` overrides the clustering."""
    rng = np.random.default_rng(seed); N = len(S); B = max(1, 2**26 // max(N, 1)); mx = []
    vals0 = np.empty(sum(int(_upper_cross(face_of, r0, min(r0 + B, N), N).sum()) for r0 in range(0, N, B)), S.dtype); k = 0
    for r0 in range(0, N, B):                            # gathered into one preallocated array (no second copy while concatenating)
        v = S[r0:min(r0 + B, N)][_upper_cross(face_of, r0, min(r0 + B, N), N)]; vals0[k:k + len(v)] = v; k += len(v)
    inplace = N > 20000                                  # U = 1000: shuffle vals0 itself (each rep is still a uniform permutation; saves N^2/2 floats); smaller N unchanged, bit-identical
    P = np.empty(S.shape, S.dtype)
    for _ in range(reps):
        vals = vals0 if inplace else vals0.copy(); rng.shuffle(vals); P[:] = S; k = 0
        for r0 in range(0, N, B):
            r1 = min(r0 + B, N); m = _upper_cross(face_of, r0, r1, N); n = int(m.sum())
            blk = P[r0:r1]; blk[m] = vals[k:k + n]; k += n
        iu_r, iu_c = None, None
        for r0 in range(0, N, B):                        # mirror: P[j, i] = P[i, j] for the permuted (i < j) entries
            r1 = min(r0 + B, N); m = _upper_cross(face_of, r0, r1, N); ii, jj = np.nonzero(m); P[jj, ii + r0] = P[ii + r0, jj]
        del vals
        if linkage is not None: cl = linkage(P)
        else: cl = average_linkage(P, face_of, tau, xp) if constrained else average_linkage_unconstrained(P, tau, xp)
        mx.append(max(len({face_of[i] for i in m}) for m in cl))
    del P, vals0
    return max(mx) + 1, mx


def nfaces(m, face_of): return len({face_of[i] for i in m})


# ---------------------------------------------------------------- naming (primary metric) and part groups
def naming(P, T, iods, t=0.25):
    """P: (nD, L, 2) detector argmax (y, x) on the L faces; T: (L, 6, 2) labelled points; iods: (L,).
    Returns named: (6,) detector index per point, and counts: (nD, 6) hits within t * IOD on the L faces."""
    d = np.linalg.norm(P[:, :, None, :] - T[None], axis=3) / iods[None, :, None]          # (nD, L, 6)
    counts = (d <= t).sum(1); med = np.median(d, 1)
    named = np.array([max(range(len(P)), key=lambda i: (counts[i, k], -med[i, k])) for k in range(len(POINTS))])
    return named, counts


def part_groups(counts, L):
    """518's rule: a detector joins the group of the point it hits most, if on >= 30 % of the L faces. {point index: [detectors]}."""
    best = counts.argmax(1); ok = (counts.max(1) >= max(.3 * L, 1)); return {k: [int(i) for i in np.flatnonzero(ok & (best == k))] for k in range(len(POINTS))}


def pck(pred, T, iods, t):
    """pred, T: (n, 6, 2); returns (n, 6) bool hits within t * IOD."""
    return np.linalg.norm(pred - T, axis=2) / iods[:, None] <= t


def position_only(T_L, mean_offset_yx):
    """Mean labelled position over the L faces (aligned coordinates), shifted into crop coordinates by the mean test offset. (6, 2)."""
    return T_L.mean(0) - np.asarray(mean_offset_yx)[None]


def mean_test_offset_yx():
    o = np.array(list(PROTO['data']['test_crop']['offsets'].values()), float)             # (x, y)
    return o[:, ::-1].mean(0)


# ---------------------------------------------------------------- secondary: ridge regression NME
def ridge_fit_predict(Xtr, Ytr, Xte, lambdas=None, L1_lambda=1e3):
    """Ridge with intercept, lambda by exact leave-one-out on the training rows (L = 1: fixed). X in px/1024."""
    lambdas = PROTO['metrics']['secondary']['ridge_lambdas'] if lambdas is None else lambdas
    mx, my = Xtr.mean(0), Ytr.mean(0); Xc, Yc = Xtr - mx, Ytr - my; n = len(Xtr)
    def fit(lam):
        K = Xc @ Xc.T; A = np.linalg.solve(K + lam * np.eye(n), np.eye(n)); return Xc.T @ A @ Yc, K @ A      # dual form: W, hat matrix
    if n == 1: lam = L1_lambda
    else:
        best = None
        for lam in lambdas:
            W, H = fit(lam); r = (Yc - H @ Yc) / (1 - np.diag(H) - 1 / n)[:, None]   # exact LOO: the unpenalised intercept adds 1/n to the hat diagonal
            e = (r**2).mean()
            if best is None or e < best[0]: best = (e, lam)
        lam = best[1]
    W, _ = fit(lam); return (Xte - mx) @ W + my, lam


def nme(pred, T, iods):
    """pred, T: (n, k, 2); returns per-face NME (mean over points of distance / IOD)."""
    return (np.linalg.norm(pred - T, axis=2).mean(1)) / iods


# ---------------------------------------------------------------- uncertainty
def bootstrap_mean(v, reps=1000, seed=0):
    """v: (n_test,) per-face values; returns mean and 95 % interval over test-face resamples."""
    idx = np.random.default_rng(seed).integers(0, len(v), (reps, len(v))); b = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
