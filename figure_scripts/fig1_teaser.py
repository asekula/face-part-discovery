"""Teaser (new method): one discovered eye detector -- the largest eye cluster of the U = 1000, 512 px run (the detector of
Figs. 3, 5, 6) -- on five eyes that are NOT members of that cluster (chosen for variety, including a dark-skinned eye). Boxes and
scores shown at level 2. Per eye: the full detector's (all levels) best match within 0.2 IOD of the image-left eye's labelled centre (applied at the eye; a sweep
shows the best match is stable out to 0.2 IOD on every teaser face) (the cluster's side: 122 of 173 members);
top: its 12 boxes in their fixed layout, with the fixed-layout score (mean box cosine, no shifts); bottom: each box moved to its
best match within +-4 level px, with the pooled score (mean of best cosine minus the spring). Images reduced to 512 px as in the run.
Laptop: run from the paper folder.  python3 figure_scripts/fig1_teaser.py [face ...]"""
import sys; sys.path.insert(0, 'code')
import pickle, numpy as np, scipy.ndimage as ndi, scipy.signal as sig, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import paper_common as pc

FACES = [int(f) for f in sys.argv[1:]] or [50618, 4024, 18543, 33001, 54372]
D = pickle.load(open('code/results/v3_U1000_r512/clusters.pkl', 'rb')); import os; CL = int(os.environ.get('TEASER_CL', 35)); C = D['clusters'][CL]   # eye cluster 35: its best members are varied (cluster 1's are all East Asian eyes)
MEMBERS = {D['KPS'][m[0]][0]: np.array(D['KPS'][m[0]][1:]) for m in C['members']}         # face -> the member keypoint (y, x)
print('members among the teaser faces:', [f for f in FACES if f in MEMBERS])
TPL = C['template'][2][:, 0]; FO, FL = D['FLOOR'][2]; RES = D.get('res', 1024)
# the full detector (all four levels), as the method locates parts: the run's own matching code (code/proto_leader2.py)
src = open('code/proto_leader2.py').read()
ns = dict(np=np, ndi=ndi, pc=pc, BOX=D['BOX'], HB=D['BOX'] // 2, SPRING=D['SPRING'], NODE=D['NODE'], LEVELS=tuple(D['NODE']), PAD=D['BOX'] // 2 + 8, FLOOR=D['FLOOR'], KPS=[])
exec(src[src.index('def planes'):src.index('FACEP = {}')], ns)
class Faces(dict):
    def __missing__(self, fi):
        L = pc.lightness(pc.rgb(fi)); r = 1024 / RES
        if r > 1: L = ndi.zoom(ndi.zoom(ndi.gaussian_filter(L, .5 * np.sqrt(r * r - 1)), 1 / r, order=1), r, order=1)
        self[fi] = {s: ns['planes'](L, s) for s in ns['LEVELS']}; return self[fi]
ns['FACEP'] = Faces(); exec(src[src.index('def unit'):src.index('def centre')], ns)
def full_match(f, y, x): ns['KPS'] = [(f, float(y), float(x))]; return float(ns['pooled'](ns['patches'](0), [C['template']])[0][0])
def locate(f):
    """Best full-detector match within 0.2 IOD of the image-left eye's labelled centre (a sweep over 0.10-0.35 IOD: the best
    match does not move beyond 0.15 for any teaser face, and for one face (23948) the brow wins beyond 0.25; this figure shows
    pooling, not detection accuracy): 8 px grid, then 2 px around the best (1024 px coords)."""
    ey, ex = MEMBERS[f] if f in MEMBERS else pc.points(f)[0]; r = (.1 if f in MEMBERS else .2) * pc.iod(f)   # members: around their own keypoint
    g = [(full_match(f, y, x), y, x) for y in np.arange(ey - r, ey + r + 1, 8) for x in np.arange(ex - r, ex + r + 1, 8) if np.hypot(y - ey, x - ex) <= r]
    _, by, bx = max(g); return max((full_match(f, y, x), y, x) for y in np.arange(by - 6, by + 7, 2) for x in np.arange(bx - 6, bx + 7, 2))
BOX, HB, RB, SPRING, S = 19, 9, 4, .05, 2
FEAT = np.round(D['NODE'][2]['feat'] / S).astype(int)
def planes(f):
    L = pc.lightness(pc.rgb(f)); r = 1024 / RES
    if r > 1: L = ndi.zoom(ndi.zoom(ndi.gaussian_filter(L, .5 * np.sqrt(r * r - 1)), 1 / r, order=1), r, order=1)
    Ls = ndi.zoom(ndi.gaussian_filter(L, .5 * np.sqrt(S * S - 1)), 1 / S, order=1)
    q = ndi.gaussian_filter(Ls, 1.0); gx, gy = ndi.sobel(q, 1), ndi.sobel(q, 0); m, th = np.hypot(gx, gy), np.arctan2(gy, gx)
    return np.stack([m * np.cos(2 * th), m * np.sin(2 * th)]), -9.0 * ndi.gaussian_laplace(Ls, 3.0)[None]
def corr(P, T): return sum(sig.fftconvolve(P[c], T[c, ::-1, ::-1], mode='same') for c in range(len(T)))
def wnorm(P): return np.sqrt(np.maximum(ndi.uniform_filter((P ** 2).sum(0), BOX) * BOX * BOX, 0))
def eye_scores(f):
    O, LG = planes(f); NO, NL = wnorm(O), wnorm(LG); H, W = O.shape[1:]; tot = np.zeros((H, W)); fixed = np.zeros((H, W)); cosm, args = [], []
    for k in range(len(FEAT)):
        To, Tl = TPL[k, :722].reshape(2, BOX, BOX), TPL[k, 722:].reshape(1, BOX, BOX)
        c = (corr(O, To) / np.maximum(NO, FO) + corr(LG, Tl) / np.maximum(NL, FL)) / (np.sqrt(2) * np.linalg.norm(TPL[k]))
        best = np.full(c.shape, -9.0); arg = np.zeros(c.shape + (2,), int)
        for vy in range(-RB, RB + 1):
            for vx in range(-RB, RB + 1):
                cand = np.roll(c, (-vy, -vx), (0, 1)) - SPRING * (vy * vy + vx * vx) / RB ** 2; up = cand > best; best[up] = cand[up]; arg[up] = (vy, vx)
        tot += np.roll(best, (-FEAT[k][0], -FEAT[k][1]), (0, 1)) / len(FEAT); fixed += np.roll(c, (-FEAT[k][0], -FEAT[k][1]), (0, 1)) / len(FEAT)
        args.append(arg)
    full, fy_, fx_ = locate(f); y0, x0 = int(round(fy_ / S)), int(round(fx_ / S))       # located by the full detector
    shifts = [args[k][y0 + FEAT[k][0], x0 + FEAT[k][1]] for k in range(len(FEAT))]
    return (y0, x0), fixed[y0, x0], tot[y0, x0], shifts, full

# drawn in the style of the first teaser: a colour per box, wide crops, scores in the corner
n = len(FACES); FW = 12 * n / 5; RX, RY = 124, 94; PW = (FW - .3) / n; PH = (PW - .06) * RY / RX; FH = 2 * PH + .25
fig = plt.figure(figsize=(FW, FH)); COL = plt.cm.tab20(np.arange(12) % 20)
for j, f in enumerate(FACES):
    (y0, x0), sf, sp, sh, full = eye_scores(f); cy, cx = 2 * y0, 2 * x0; im = pc.rgb(f)[cy - RY:cy + RY, cx - RX:cx + RX]
    print(f'face {f}: full-detector best match ({cy}, {cx}) px, score {full:.2f}; level 2 at that point: rigid {sf:.2f}, max pooling {sp:.2f}')
    for row, pooled in enumerate((False, True)):
        a = fig.add_axes([(.3 + j * PW) / FW, (.08 + (1 - row) * (PH + .08)) / FH, (PW - .06) / FW, PH / FH]); a.imshow(im, extent=(-RX, RX, RY, -RY))
        for k, (fy, fx) in enumerate(FEAT * S):
            dy, dx = (sh[k] * S) if pooled else (0, 0)
            a.add_patch(Rectangle((fx + dx - BOX, fy + dy - BOX), 2 * BOX, 2 * BOX, fill=False, lw=1.8, ec=COL[k]))
        a.set_xlim(-RX, RX); a.set_ylim(RY, -RY); a.set_xticks([]); a.set_yticks([])
        a.text(-RX + 8, RY - 8, f'{"max pooling" if pooled else "rigid"}: {sp if pooled else sf:.2f}', color='white', fontsize=9, ha='left', va='bottom',
               bbox=dict(boxstyle='square,pad=.25', fc='black', ec='none', alpha=.6))
        if n > 5 and row == 0: a.set_title(str(f), fontsize=9)
fig.text(.15 / FW, (.08 + PH + .08 + PH / 2) / FH, 'fixed layout', rotation=90, ha='center', va='center', fontsize=10)
fig.text(.15 / FW, (.08 + PH / 2) / FH, 'after max pooling', rotation=90, ha='center', va='center', fontsize=10)
out = 'figures/fig1_pooling_eyes' if n == 5 else 'figures/fig1_teaser_preview'
fig.savefig(out + '.pdf', dpi=200); fig.savefig(out + '.png', dpi=150); print('saved', out)
