"""Figure 3 (new method): how a discovered detector scores an eye in an image, at level 2 only.
The detector is a discovered eye cluster's level-2 part model (the aligned mean of its members' 12 boxes); the image eye is the
cluster's best-matching eye member (code/proto_gpu/fig3_pick.py packs both, with the energy floor, into a folder). The detector
is scanned over that face at level 2 exactly as in the method: per box, cosine of the box's descriptor (oriented gradient and
LoG, each group divided by max(its norm, energy floor)) with the template, best over shifts of up to +-4 level px minus the
spring 0.05 (d / 4)^2; mean over the 12 boxes. The best location is shown.
Left: the image eye's oriented-gradient map with the 12 boxes where they matched. Middle: for boxes 6, 7, 8 and 12 -- the image
eye's box at its matched place, cos, the detector's box -> the oriented-gradient cosine of the two (the channel drawn; LoG is
compared the same way). Right: the mean of the 12 scores -> one value.
Laptop: run from the paper folder.  python3 figure_scripts/fig3_scoring.py [data folder, default figure_scripts/fig3_data]"""
import sys; sys.path.insert(0, 'code')
import numpy as np, scipy.ndimage as ndi, scipy.signal as sig, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image
import paper_common as pc

DATA = sys.argv[1] if len(sys.argv) > 1 else 'figure_scripts/fig3_data'
Z = np.load(f'{DATA}/fig3_data.npz'); TPL, (FO, FL) = Z['template2'][:, 0], Z['floor2']
BOX, HB, RB, SPRING, S = 19, 9, 4, .05, 2
FEAT = np.round(Z['node2_feat'] / S).astype(int)                          # level-2 offsets of the 12 boxes
SHOW = [5, 6, 7, 11]                                                       # boxes drawn in the middle (0-based); "..." before the last

L = pc.lightness(np.asarray(Image.open(f'{DATA}/face.png').convert('RGB')))
RES = int(Z['res']) if 'res' in Z else 1024                                 # reduce as the run did (e.g. 512), enlarge back
if RES < 1024: r = 1024 / RES; L = ndi.zoom(ndi.zoom(ndi.gaussian_filter(L, .5 * np.sqrt(r * r - 1)), 1 / r, order=1), r, order=1)
Ls = ndi.zoom(ndi.gaussian_filter(L, .5 * np.sqrt(S * S - 1)), 1 / S, order=1)
q = ndi.gaussian_filter(Ls, 1.0); gx, gy = ndi.sobel(q, 1), ndi.sobel(q, 0); MAG, TH = np.hypot(gx, gy), np.arctan2(gy, gx)
O = np.stack([MAG * np.cos(2 * TH), MAG * np.sin(2 * TH)]); LG = -9.0 * ndi.gaussian_laplace(Ls, 3.0)[None]
def corr(P, T): return sum(sig.fftconvolve(P[c], T[c, ::-1, ::-1], mode='same') for c in range(len(T)))
def wnorm(P): return np.sqrt(np.maximum(ndi.uniform_filter((P ** 2).sum(0), BOX) * BOX * BOX, 0))
NO, NL = wnorm(O), wnorm(LG)
def pool(C):
    best = np.full(C.shape, -9.0); arg = np.zeros(C.shape + (2,), int)
    for vy in range(-RB, RB + 1):
        for vx in range(-RB, RB + 1):
            cand = np.roll(C, (-vy, -vx), (0, 1)) - SPRING * (vy * vy + vx * vx) / RB ** 2
            up = cand > best; best[up] = cand[up]; arg[up] = (vy, vx)
    return best, arg
H, W = Ls.shape; score = np.zeros((H, W)); arg, cos_o, T = [], [], []
for k in range(len(FEAT)):
    To, Tl = TPL[k, :722].reshape(2, BOX, BOX), TPL[k, 722:].reshape(1, BOX, BOX); T.append(To)
    c = (corr(O, To) / np.maximum(NO, FO) + corr(LG, Tl) / np.maximum(NL, FL)) / (np.sqrt(2) * np.linalg.norm(TPL[k]))
    pk, ak = pool(c); score += np.roll(pk, (-FEAT[k][0], -FEAT[k][1]), (0, 1)) / len(FEAT); arg.append(ak)
    cos_o.append(corr(O, To) / np.maximum(NO, 1e-9) / np.linalg.norm(To))
M = 24 + HB + RB; valid = np.zeros_like(score, bool); valid[M:H - M, M:W - M] = True
y0, x0 = np.unravel_index(np.argmax(np.where(valid, score, -9)), score.shape)
ky, kx = Z['kp'] / S; fi = int(Z['face'])
print(f'cluster {int(Z["cluster"])} ({int(Z["n"])} members), member face {fi}: best level-2 match ({y0}, {x0}), score {score[y0, x0]:.3f}; '
      f'the member keypoint is {np.hypot(y0 - ky, x0 - kx) * S / pc.iod(fi):.2f} IOD away')
match = []
for k, f in enumerate(FEAT):
    cy, cx = y0 + f[0], x0 + f[1]; vy, vx = arg[k][cy, cx]; match.append((cy + vy, cx + vx, cos_o[k][cy + vy, cx + vx]))
mean_o = np.mean([m[2] for m in match]); print('gradient cosines', np.round([m[2] for m in match], 2), 'mean', round(mean_o, 3))

def orient_rgb(m, th, vmax): return hsv_to_rgb(np.stack([(th % np.pi) / np.pi, np.ones_like(th), np.clip(m / vmax, 0, 1)], -1))
def win(a, y, x): return a[..., y - HB:y + HB + 1, x - HB:x + HB + 1]
FW, FH = 10.4, 4.8
fig = plt.figure(figsize=(FW, FH))
def ax_in(x, y, w, h): return fig.add_axes([x / FW, y / FH, w / FW, h / FH])
def fx(x): return x / FW
def fy(y): return y / FH
RX, RY = 40, 33                                                           # left: the image eye, cropped just around the boxes
BW = 4.1; big = ax_in(.05, (FH - BW * RY / RX) / 2 - .1, BW, BW * RY / RX)
reg = np.s_[y0 - RY:y0 + RY, x0 - RX:x0 + RX]
big.imshow(np.clip(ndi.zoom(orient_rgb(MAG[reg], TH[reg], np.percentile(MAG[reg], 99)), (4, 4, 1), order=1), 0, 1), extent=(-RX, RX, RY, -RY))
for k in range(len(FEAT)):
    my, mx, _ = match[k]; hl = k in SHOW
    big.add_patch(Rectangle((mx - x0 - HB - .5, my - y0 - HB - .5), BOX, BOX, fill=False, lw=2 if hl else 1,
                            ec='yellow' if hl else 'cyan', zorder=3 if hl else 2))
    big.text(mx - x0 - HB + 1, my - y0 - HB + 1, str(k + 1), color='yellow' if hl else 'cyan', fontsize=8, va='top', ha='left', zorder=4)
big.set_xlim(-RX, RX); big.set_ylim(RY, -RY); big.set_title('image: oriented gradient (level 2)\nthe 12 boxes where they matched', fontsize=10)

vn = np.percentile(np.concatenate([win(MAG, m[0], m[1]).ravel() for m in match]), 99)
vd = np.percentile(np.concatenate([np.hypot(*t).ravel() for t in T]), 99)
TS, X1, X2 = .78, 4.75, 6.05
rows = [3.55, 2.6, 1.65, .35]; sx = 8.15
fig.text(fx(X1 + TS / 2), fy(4.45), 'image', ha='center', fontsize=10); fig.text(fx(X2 + TS / 2), fy(4.45), 'detector', ha='center', fontsize=10)
for r, k in zip(rows, SHOW):
    my, mx, c = match[k]; nb = ax_in(X1, r, TS, TS); db = ax_in(X2, r, TS, TS)
    nb.imshow(orient_rgb(win(MAG, my, mx), win(TH, my, mx), vn), interpolation='bilinear')
    db.imshow(orient_rgb(np.hypot(*T[k]), np.arctan2(T[k][1], T[k][0]) / 2, vd), interpolation='bilinear')
    for a in (nb, db):
        for sp in a.spines.values(): sp.set_color('gold'); sp.set_linewidth(1.5)
    fig.text(fx(X1 - .12), fy(r + TS / 2), f'box {k + 1}', ha='right', va='center', fontsize=10)
    fig.text(fx((X1 + TS + X2) / 2), fy(r + TS / 2), 'cos', ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='circle,pad=.25', fc='white', ec='black', lw=.8))
    fig.text(fx(X2 + TS + .15), fy(r + TS / 2), f'= {c:.2f}', ha='left', va='center', fontsize=10)
    fig.add_artist(FancyArrowPatch((fx(X2 + TS + .75), fy(r + TS / 2)), (fx(sx), fy(2.3)), transform=fig.transFigure,
                                   arrowstyle='-|>', mutation_scale=10, lw=.8, color='0.3', shrinkB=22))
YE = (rows[2] + rows[3] + TS) / 2                                          # midway between the box-8 and box-12 rows
for xe in (X1 + TS / 2, (X1 + TS + X2) / 2, X2 + TS / 2, X2 + TS + .4):     # under the image, cos, detector and score columns
    fig.text(fx(xe), fy(YE), '⋮', ha='center', va='center', fontsize=16)
fig.text(fx(sx), fy(2.3), 'mean', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round,pad=.45', fc='white', ec='black', lw=.8))
fig.add_artist(FancyArrowPatch((fx(sx + .45), fy(2.3)), (fx(sx + 1.2), fy(2.3)), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=10, lw=.8, color='0.3'))
fig.text(fx(sx + 1.3), fy(2.3), f'{mean_o:.2f}', ha='left', va='center', fontsize=12, fontweight='bold')
fig.text(fx(sx + 1.5), fy(1.95), 'detector score', ha='center', va='top', fontsize=10)
for a in fig.axes: a.set_xticks([]); a.set_yticks([])
out = 'figures/fig3_scoring'; fig.savefig(out + '.pdf', dpi=200); fig.savefig(out + '.png', dpi=150); print('saved', out)
