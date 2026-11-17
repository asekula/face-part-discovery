"""Figure 5 (new method): the discovery pipeline, from a discovery run's saved clusters (default: the local U = 100 prototype).
(1) An unlabelled face with its keypoints, on the face and on the sigma-32 LoG map; one keypoint (an eye) highlighted.
(2) Leader clustering: that keypoint's detector matched against a few cluster leaders (full pooled match, all levels); it joins
    the best one at or above tau that has no member from its face. Each leader is drawn as its part model at level 2 (oriented-gradient tiles).
(3) One cluster: its 6 best-matching members' crops -> its part model (aligned mean), drawn at level 2 (oriented gradient and LoG tiles).
(4) Parts chosen with labels (the main pipeline, select_parts.py): the cross-validated NME as parts are added, and the first K chosen.
(5) Those K parts' detections on a test crop, from the protocol run (v3/parts/ourssel_U<U>_K32.npz).
Labels are used only to choose which clusters to show. Matching code: code/proto_leader2.py (its definitions, run here).
Laptop: run from the paper folder.  python3 figure_scripts/fig5_pipeline.py [run folder] [K] [detection face]"""
import sys, os, re, time, pickle, json; sys.path.insert(0, 'code')
import numpy as np, scipy.ndimage as ndi, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import FancyArrowPatch, Rectangle
import paper_common as pc

RUN = sys.argv[1] if len(sys.argv) > 1 else 'code/results/v3_U1000_r512'
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
D = pickle.load(open(f'{RUN}/clusters.pkl', 'rb')); KPS, C, TAU = D['KPS'], D['clusters'], D['TAU']
DETFACE = int(sys.argv[3]) if len(sys.argv) > 3 else 22061

# the prototype's matching code: its definitions only (planes .. centre), with faces filtered on demand
src = open('code/proto_leader2.py').read()
ns = dict(np=np, ndi=ndi, pc=pc, BOX=D['BOX'], HB=D['BOX'] // 2, SPRING=D['SPRING'], NODE=D['NODE'], LEVELS=tuple(D['NODE']),
          PAD=D['BOX'] // 2 + 8, FLOOR=D['FLOOR'], KPS=list(KPS))
exec(src[src.index('def planes'):src.index('FACEP = {}')], ns)
class Faces(dict):
    def __missing__(self, fi):
        L = pc.lightness(pc.rgb(fi)); r = 1024 / D.get('res', 1024)
        if r > 1: L = ndi.zoom(ndi.zoom(ndi.gaussian_filter(L, .5 * np.sqrt(r * r - 1)), 1 / r, order=1), r, order=1)   # as the run
        self[fi] = {s: ns['planes'](L, s) for s in ns['LEVELS']}; return self[fi]
ns['FACEP'] = Faces()
exec(src[src.index('def unit'):src.index('def centre')], ns)
def patches_at(fi, y, x): ns['KPS'] = [(fi, y, x)]; return ns['patches'](0)
def match(fi, y, x, ids): return ns['pooled'](patches_at(fi, y, x), [C[i]['template'] for i in ids])[0]

# which clusters sit on which part (for choosing what to show only)
def part_of(k):
    fi, y, x = KPS[k]; P = pc.points(fi); d = np.hypot(P[:6, 0] - y, P[:6, 1] - x) / pc.iod(fi)
    return ['eye', 'eye', 'brow', 'brow', 'nose', 'mouth'][d.argmin()] if d.min() <= .25 else 'other'
lab = []
for c in C:
    ps = [part_of(m[0]) for m in c['members']]; top = max(set(ps), key=ps.count); lab.append(top if ps.count(top) >= .6 * len(ps) else 'mixed')
by = lambda p: sorted([i for i, l in enumerate(lab) if l == p], key=lambda i: -C[i]['n'])
EYE = by('eye')[0]; QFACE = int(os.environ.get('FIG_QUERY', 34318))           # the keypoint followed: an adult member of the eye cluster
mem = max((m for m in C[EYE]['members'] if KPS[m[0]][0] == QFACE), key=lambda m: m[2]); KQ = mem[0]; QF, QY, QX = KPS[KQ]
LEAD = [EYE] + by('eye')[1:2] + by('nose')[:1] + by('mouth')[:1]
sc = match(QF, QY, QX, LEAD); t1 = time.time(); match(QF, QY, QX, [EYE]); print(f'one match {time.time() - t1:.2f} s', flush=True)
print('query', KPS[KQ], 'scores', dict(zip(LEAD, np.round(sc, 3))), 'tau', round(TAU, 3))

# parts chosen with labels (the main pipeline): select_parts.py's greedy order and CV NME trace; their detections on a test
# crop from the protocol run (v3/parts/ourssel_U<U>_K32.npz, 384 px crop coordinates)
SEL = json.load(open(f'{RUN}/parts_labelsel.json')); kept = SEL['parts'][:K]; TRACE = SEL['cv_nme_trace']
U_ = len(D['U']); Z = np.load(f'code/v3/parts/ourssel_U{U_}_K32.npz'); TFACE = int(pc.test_faces('secondary')[0])
tx, ty = json.load(open('code/paper_protocol.json'))['data']['test_crop']['offsets'][str(TFACE)]
IM = pc.rgb(TFACE)[ty:ty + 768, tx:tx + 768]; DET = Z['test'][0, :K] * 2                       # -> 768 px display crop
print('chosen', [(i, C[i]['n'], lab[i]) for i in kept], 'test face', TFACE)

# ---------------- drawing
FW, FH = 12.65, 4.2
fig = plt.figure(figsize=(FW, FH))
def ax_in(x, y, w, h): a = fig.add_axes([x / FW, y / FH, w / FW, h / FH]); a.set_xticks([]); a.set_yticks([]); return a
def arrow(x0, y0, x1, y1, **kw):
    fig.add_artist(FancyArrowPatch((x0 / FW, y0 / FH), (x1 / FW, y1 / FH), transform=fig.transFigure, arrowstyle='-|>',
                                   mutation_scale=10, lw=.8, color='0.3', **kw))
def head(x, t): fig.text(x / FW, 4.02 / FH, t, ha='center', va='bottom', fontsize=10)
def crop(fi, y, x, r=80):
    im = pc.rgb(fi); y, x = int(round(y)), int(round(x)); return im[max(y - r, 0):y + r, max(x - r, 0):x + r]
def mean_crop(i, n=40): return np.mean([crop(*KPS[m[0]]).astype(float) for m in C[i]['members'][:n]], 0) / 255
def orient_rgb(m, th, vmax): return hsv_to_rgb(np.stack([(th % np.pi) / np.pi, np.ones_like(th), np.clip(m / vmax, 0, 1)], -1))
def mosaic(t, part, gap=2):
    """Level-2 template (12, 1, D) as a 3 x 4 grid of 19 x 19 tiles: oriented gradient (RGB) or LoG (values)."""
    B = D['BOX']; out = np.ones((3 * B + 2 * gap, 4 * B + 3 * gap, 3)) if part == 'o' else np.full((3 * B + 2 * gap, 4 * B + 3 * gap), np.nan)
    o = t[:, 0, :2 * B * B].reshape(12, 2, B, B); l = t[:, 0, 2 * B * B:].reshape(12, B, B); vm = np.percentile(np.hypot(o[:, 0], o[:, 1]), 99)
    for k in range(12):
        r, c = divmod(k, 4); sl = np.s_[r * (B + gap):r * (B + gap) + B, c * (B + gap):c * (B + gap) + B]
        out[sl] = orient_rgb(np.hypot(o[k, 0], o[k, 1]), np.arctan2(o[k, 1], o[k, 0]) / 2, vm) if part == 'o' else l[k]
    return out

# (1) keypoints on the face and the LoG map
head(1.05, '(1) keypoints'); Lq = pc.lightness(pc.rgb(QF)); m32 = -(32.0 ** 2) * ndi.gaussian_laplace(Lq, 32.0)
KF = np.array([KPS[k][1:] for k in range(len(KPS)) if KPS[k][0] == QF])
a1 = ax_in(.05, 2.05, 1.9, 1.9); a1.imshow(pc.rgb(QF)); a2 = ax_in(.05, .05, 1.9, 1.9)
vl = np.percentile(np.abs(m32), 99); a2.imshow(m32, cmap='RdBu_r', vmin=-vl, vmax=vl)
for a in (a1, a2):
    a.scatter(KF[:, 1], KF[:, 0], s=7, fc='none', ec='yellow', lw=.6); a.scatter([QX], [QY], s=40, fc='none', ec='lime', lw=1.4)
# (2) leader clustering
head(3.55, '(2) leader clustering'); q = ax_in(2.25, 1.55, 1.0, 1.0); q.imshow(crop(QF, QY, QX))
for sp in q.spines.values(): sp.set_color('lime'); sp.set_linewidth(1.5)
fig.text(2.75 / FW, 1.45 / FH, 'keypoint', ha='center', va='top', fontsize=10)
joined = next((j for j in np.argsort(-sc) if sc[j] >= TAU), None)
for r, (i, s_) in enumerate(zip(LEAD, sc)):
    y = 3.1 - r * .98; a = ax_in(3.75, y + .1, .95, .95 * 61 / 82); a.imshow(mosaic(C[i]['template'][2], 'o'), interpolation='bilinear')
    arrow(3.28, 2.05, 3.72, y + .41, shrinkB=2)
    ok = r == joined; fig.text(4.75 / FW, (y + .41) / FH, f'{s_:.2f}' + (' joins' if ok else ''), ha='left', va='center', fontsize=10,
                               color='green' if ok else ('0.2' if s_ >= TAU else '0.5'), fontweight='bold' if ok else 'normal')
fig.text(4.3 / FW, .05 / FH, f'leaders\' part models; $\\tau$ = {TAU:.2f}', ha='center', va='bottom', fontsize=10)
# (3) one cluster -> its part model
head(7.05, '(3) a cluster and its part model'); M = sorted(C[EYE]['members'], key=lambda m: -m[2])[:6]
for n, m in enumerate(M):
    r, c = divmod(n, 2); a = ax_in(5.75 + c * .78, 2.75 - r * .78, .72, .72); a.imshow(crop(*KPS[m[0]]))
fig.text(6.5 / FW, .5 / FH, f'{len(M)} best-matching\nof {C[EYE]["n"]} members', ha='center', va='center', fontsize=10)
arrow(7.3, 2.0, 7.65, 2.0)
a = ax_in(7.7, 2.15, 1.25, 1.25 * 61 / 82); a.imshow(mosaic(C[EYE]['template'][2], 'o'), interpolation='bilinear')
lg = mosaic(C[EYE]['template'][2], 'l'); vl = np.nanpercentile(np.abs(lg), 99)
a = ax_in(7.7, 1.05, 1.25, 1.25 * 61 / 82); a.imshow(np.ma.masked_invalid(lg), cmap='RdBu_r', vmin=-vl, vmax=vl, interpolation='bilinear')
fig.text(8.32 / FW, .85 / FH, 'part model\n(level 2)', ha='center', va='top', fontsize=10)
# (4) parts chosen with labels
head(9.72, '(4) parts chosen with labels'); a = fig.add_axes([9.42 / FW, 2.6 / FH, 1.0 / FW, 1.2 / FH])
a.plot(range(1, len(TRACE) + 1), TRACE, color='tab:orange', lw=1.5); a.tick_params(labelsize=8)
a.set_xlabel('parts chosen', fontsize=9); a.set_ylabel('CV NME (%)', fontsize=9); [s_.set_visible(False) for k_, s_ in a.spines.items() if k_ in ('top', 'right')]
fig.text(9.72 / FW, 1.98 / FH, f'first {K} chosen:', ha='center', va='center', fontsize=9)
cols = plt.cm.tab10(np.arange(K) % 10)
for r, i in enumerate(kept):
    t = ax_in(9.02 + (r % 4) * .37, 1.42 - (r // 4) * .38, .33, .33); t.imshow(crop(*KPS[max(C[i]['members'], key=lambda m: m[2])[0]]))
    for sp in t.spines.values(): sp.set_color(cols[r]); sp.set_linewidth(2)
# (5) detection
head(11.55, '(5) detection'); a = ax_in(10.83, .95, 1.45, 1.45); a.imshow(IM)
for k, (y, x) in enumerate(DET): a.scatter([x], [y], s=22, color=cols[k], ec='white', lw=.6)
fig.text(11.55 / FW, .8 / FH, f'the {K} parts on a test crop,\nbest match each', ha='center', va='top', fontsize=10)
out = 'figures/fig5_pipeline'; fig.savefig(out + '.pdf', dpi=200); fig.savefig(out + '.png', dpi=150); print('saved', out)
