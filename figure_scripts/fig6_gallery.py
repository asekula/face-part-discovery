"""Figure 6 (new method): the part gallery -- discovered part detectors drawn directly.
One row per part (the largest cluster sitting on eye / nose / mouth / brow if any, and the largest 'other' cluster): its 4
best-matching members with the 12 level-2 boxes where they matched; then the part model at level 2 -- oriented gradient and LoG --
aligned mean (each member's boxes where they matched) next to unaligned mean (each member's boxes at their fixed places).
Labels are used only to choose which clusters to show. Unaligned means: as saved by the GPU runs (template_unaligned); recomputed from the members with the run's own code
(code/proto_leader2.py's definitions) for older runs that lack them. Laptop: run from the paper folder.  python3 figure_scripts/fig6_gallery.py [run folder]"""
import sys, os, pickle; sys.path.insert(0, 'code')
import numpy as np, scipy.ndimage as ndi, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle
import paper_common as pc

RUN = sys.argv[1] if len(sys.argv) > 1 else 'code/results/v3_U1000_r512'
D = pickle.load(open(f'{RUN}/clusters.pkl', 'rb')); KPS, C, BOX = D['KPS'], D['clusters'], D['BOX']
src = open('code/proto_leader2.py').read()
ns = dict(np=np, ndi=ndi, pc=pc, BOX=BOX, HB=BOX // 2, SPRING=D['SPRING'], NODE=D['NODE'], LEVELS=tuple(D['NODE']),
          PAD=BOX // 2 + 8, FLOOR=D['FLOOR'], KPS=list(KPS))
exec(src[src.index('def planes'):src.index('FACEP = {}')], ns)
class Faces(dict):
    def __missing__(self, fi):
        L = pc.lightness(pc.rgb(fi)); self[fi] = {s: ns['planes'](L, s) for s in ns['LEVELS']}; return self[fi]
ns['FACEP'] = Faces()
exec(src[src.index('def unit'):src.index('def pooled')], ns)
R2 = D['NODE'][2]['RF'] + D['NODE'][2]['RB']; W2 = 2 * R2 + 1; CENTRE2 = R2 * W2 + R2

NAMES = ['eye', 'eye', 'brow', 'brow', 'nose', 'mouth']
def part_of(k):
    fi, y, x = KPS[k]; P = pc.points(fi); d = np.hypot(P[:6, 0] - y, P[:6, 1] - x) / pc.iod(fi)
    return NAMES[d.argmin()] if d.min() <= .25 else 'other'
lab = []
for c in C:
    ps = [part_of(m[0]) for m in c['members']]; top = max(set(ps), key=ps.count); lab.append(top if ps.count(top) >= .6 * len(ps) else 'mixed')
ROWS = [(p, max((i for i in range(len(C)) if lab[i] == p), key=lambda i: C[i]['n'], default=None)) for p in ('eye', 'nose', 'mouth', 'brow', 'other')]
ROWS = [(p, i) for p, i in ROWS if i is not None]; print('rows:', [(p, i, C[i]['n']) for p, i in ROWS])

def unaligned(i):
    acc = []
    for m in C[i]['members']:
        ns['KPS'] = [KPS[m[0]]]; acc.append(ns['patches'](0)[2][:, :, CENTRE2])
    return np.mean(acc, 0)
def orient_rgb(m, th, vmax): return hsv_to_rgb(np.stack([(th % np.pi) / np.pi, np.ones_like(th), np.clip(m / vmax, 0, 1)], -1))
def mosaic(t, part, vm=None, gap=2):
    """Level-2 template (12, 1, D) as a 3 x 4 grid of BOX x BOX tiles: oriented gradient (RGB) or LoG (values)."""
    B = BOX; H, W = 3 * B + 2 * gap, 4 * B + 3 * gap; out = np.ones((H, W, 3)) if part == 'o' else np.full((H, W), np.nan)
    o = t[:, 0, :2 * B * B].reshape(12, 2, B, B); l = t[:, 0, 2 * B * B:].reshape(12, B, B)
    for k in range(12):
        r, c = divmod(k, 4); sl = np.s_[r * (B + gap):r * (B + gap) + B, c * (B + gap):c * (B + gap) + B]
        out[sl] = orient_rgb(np.hypot(o[k, 0], o[k, 1]), np.arctan2(o[k, 1], o[k, 0]) / 2, vm) if part == 'o' else l[k]
    return out
def crop(fi, y, x, r=80):
    im = pc.rgb(fi); y, x = int(round(y)), int(round(x)); return im[y - r:y + r, x - r:x + r]

FW, RH = 11.15, 1.25; FH = .45 + RH * len(ROWS)
fig = plt.figure(figsize=(FW, FH))
def ax_in(x, y, w, h): a = fig.add_axes([x / FW, y / FH, w / FW, h / FH]); a.set_xticks([]); a.set_yticks([]); return a
TS = 1.0; MW = 1.45; MH = MW * (3 * BOX + 4) / (4 * BOX + 6); X0 = 4 * (TS + .06) + .55
heads = ['members (best 4), level-2 boxes where they matched', 'edges, aligned', 'edges, unaligned', 'blobs, aligned', 'blobs, unaligned']
fig.text((.1 + 2 * (TS + .06)) / FW, (FH - .3) / FH, heads[0], ha='center', fontsize=10)
for j, h in enumerate(heads[1:]): fig.text((X0 + j * (MW + .12) + MW / 2) / FW, (FH - .3) / FH, h, ha='center', fontsize=10)
HOME = D['NODE'][2]['feat']
for r, (part, i) in enumerate(ROWS):
    y0 = FH - .45 - (r + 1) * RH + .1
    fig.text(.02 / FW, (y0 + TS / 2) / FH, f'{part}\n{C[i]["n"]} faces', ha='left', va='center', fontsize=10, rotation=90)
    for n, m in enumerate(sorted(C[i]['members'], key=lambda m: -m[2])[:4]):
        a = ax_in(.35 + n * (TS + .06), y0, TS, TS); fi, ky, kx = KPS[m[0]]; a.imshow(crop(fi, ky, kx), extent=(-80, 80, 80, -80))
        for k, f in enumerate(HOME):                                         # level-2 boxes where they matched
            dy, dx = (np.array(divmod(m[1][2][k][0], W2)) - R2) * 2; h = BOX * 2 / 2
            a.add_patch(Rectangle((f[1] + dx - h, f[0] + dy - h), 2 * h, 2 * h, fill=False, lw=.6, ec='cyan', alpha=.8))
        a.set_xlim(-80, 80); a.set_ylim(80, -80)
    al = C[i]['template'][2]; un = C[i]['template_unaligned'][2] if 'template_unaligned' in C[i] else unaligned(i)   # saved by GPU runs
    vm = np.percentile(np.hypot(*al[:, 0, :2 * BOX * BOX].reshape(12, 2, -1).transpose(1, 0, 2)), 99)
    vl = np.nanpercentile(np.abs(mosaic(al, 'l')), 99)
    for j, (t, part_) in enumerate([(al, 'o'), (un, 'o'), (al, 'l'), (un, 'l')]):
        a = ax_in(X0 + j * (MW + .12), y0 + (TS - MH) / 2, MW, MH); mz = mosaic(t, part_, vm)
        a.imshow(mz, interpolation='bilinear') if part_ == 'o' else a.imshow(np.ma.masked_invalid(mz), cmap='RdBu_r', vmin=-vl, vmax=vl, interpolation='bilinear')
out = 'figures/fig6_gallery'; fig.savefig(out + '.pdf', dpi=200); fig.savefig(out + '.png', dpi=150); print('saved', out)
