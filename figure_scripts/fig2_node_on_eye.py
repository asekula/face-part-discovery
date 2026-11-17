"""Figure 2 (new method): one eye region, its two filter channels, and the detector instantiated there at levels 1 / 2 / 4 / 8.
Row 1: the region; its oriented-gradient map (hue = gradient direction mod 180, brightness = strength) and signed LoG map (red
bright, blue dark), both at level 1 (full resolution); the image at each level (blurred and shrunk by the level,
enlarged smoothly for display) with the node's boxes: 12 / 12 / 6 / 1 boxes of 19 / 38 / 76 / 152 px, every box 19 x 19 pixels at its
level, one box in yellow. Row 2: what each of the 12 level-2 boxes holds, at its place in the 3 x 4 grid but drawn apart:
oriented gradient and LoG; the level-8 box's samples (oriented gradient, LoG); the colour wheel. Geometry and filters as in code/proto_gpu/proto_leader_gpu.py
(the frozen node, --level1 single). Laptop: run from the paper folder.
  python3 figure_scripts/fig2_node_on_eye.py [face_id] [eye: 0 = image-left, 1 = image-right]"""
import sys; sys.path.insert(0, 'code')
import numpy as np, scipy.ndimage as ndi, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle
import paper_common as pc

FACE = int(sys.argv[1]) if len(sys.argv) > 1 else 25449; EYE = int(sys.argv[2]) if len(sys.argv) > 2 else 0
BOX, HB = 19, 9
HOME = (np.array([(dy, dx) for dy in (-8, 0, 8) for dx in (-12, -4, 4, 12)]) * 4).astype(float)
NODE = {1: HOME, 2: HOME, 4: np.array([(dy, dx) for dy in (-22, 22) for dx in (-44, 0, 44)], float), 8: np.zeros((1, 2))}
HI = {1: 5, 2: 5, 4: 1, 8: 0}                                                   # the box drawn in yellow at each level
R = 100                                                                   # half-width of the region shown, image px

im = pc.rgb(FACE); L = pc.lightness(im); cy, cx = np.round(pc.points(FACE)[EYE]).astype(int)
def level(s): return L if s == 1 else ndi.zoom(ndi.gaussian_filter(L, .5 * np.sqrt(s * s - 1)), 1 / s, order=1)   # the method's pyramid
def channels(Ls):
    q = ndi.gaussian_filter(Ls, 1.0); gx, gy = ndi.sobel(q, 1), ndi.sobel(q, 0)
    return np.hypot(gx, gy), np.arctan2(gy, gx), -9.0 * ndi.gaussian_laplace(Ls, 3.0)
def crop(a, s):
    """The shown region from an image at level s, enlarged smoothly to full size."""
    r, y, x = R // s, cy // s, cx // s; a = a[y - r:y + r, x - r:x + r]; return a if s == 1 else ndi.zoom(a, s, order=1)
def orient_rgb(m, th, vmax):
    return hsv_to_rgb(np.stack([(th % np.pi) / np.pi, np.ones_like(th), np.clip(m / vmax, 0, 1)], -1))

L2 = level(2); m2, th2, lg2 = channels(L2)
W, H = 12, 4.35                                                                    # inches; same width and fonts as before
fig = plt.figure(figsize=(W, H))
def ax_in(x, y, w, h): return fig.add_axes([x / W, y / H, w / W, h / H])         # axes placed in inches
# row 1: seven panels -- the region, its two channels at level 1, the boxes at levels 1 / 2 / 4 / 8
PS, PG, Y1 = 1.63, .08, 2.15
P = [ax_in(.05 + i * (PS + PG), Y1, PS, PS) for i in range(7)]
P[0].imshow(im[cy - R:cy + R, cx - R:cx + R]); P[0].set_title('eye region', fontsize=10)
m1, th1, lg1 = channels(L); sl1 = np.s_[cy - R:cy + R, cx - R:cx + R]
P[1].imshow(orient_rgb(m1[sl1], th1[sl1], np.percentile(m1[sl1], 99))); P[1].set_title('oriented gradient\n(level 1)', fontsize=10)
vl = np.percentile(np.abs(lg1[sl1]), 99); P[2].imshow(lg1[sl1], cmap='RdBu_r', vmin=-vl, vmax=vl)
P[2].set_title('LoG (level 1)\nred bright, blue dark', fontsize=10)
for ax, (s, feat) in zip(P[3:], NODE.items()):
    ax.imshow(crop(level(s), s), cmap='gray', vmin=0, vmax=1)
    for k, f in enumerate(feat):
        y, x = R + f[0], R + f[1]; h = BOX * s / 2; hi = k == HI[s]
        ax.add_patch(Rectangle((x - h, y - h), 2 * h, 2 * h, fill=False, lw=2 if hi else .8, ec='yellow' if hi else 'cyan',
                               alpha=1 if hi else .55, zorder=3 if hi else 2))
    ax.set_title(f'level {s}\n{len(feat)} box{"es" * (len(feat) > 1)} of {BOX * s} px', fontsize=10)

# row 2: the 19 x 19 samples each level-2 box holds, at its place in the 3 x 4 grid but apart; then the colour wheel
tiles = []
for f in NODE[2]:
    y, x = int(round((cy + f[0]) / 2)), int(round((cx + f[1]) / 2)); sl = np.s_[y - HB:y + HB + 1, x - HB:x + HB + 1]
    tiles.append((m2[sl], th2[sl], lg2[sl]))
vm = np.percentile(np.concatenate([t[0].ravel() for t in tiles]), 99)            # one scale for all 12 boxes
vg = np.percentile(np.abs(np.concatenate([t[2].ravel() for t in tiles])), 99)
T, G, Y2, WHL, SEP = .42, .06, .2, 1.2, .55                                 # smaller tiles: the level-8 box fits in the row
GW, GH = 4 * T + 3 * G, 3 * T + 2 * G; LT = GH                                  # level-8 tiles as tall as a level-2 grid
X0 = (W - (2 * GW + 2 * LT + .12 + WHL + 3 * SEP)) / 2
for g, (x0, name) in enumerate([(X0, 'oriented gradient'), (X0 + GW + SEP, 'LoG')]):
    fig.text((x0 + GW / 2) / W, (Y2 + GH + .1) / H, f'level 2 boxes: {name}', ha='center', fontsize=10)
    for k, (tm, tt, tl) in enumerate(tiles):
        r, c = divmod(k, 4); ax = ax_in(x0 + c * (T + G), Y2 + (2 - r) * (T + G), T, T)
        if g == 0: ax.imshow(orient_rgb(tm, tt, vm), interpolation='bilinear')
        else: ax.imshow(tl, cmap='RdBu_r', vmin=-vg, vmax=vg, interpolation='bilinear')
        for sp in ax.spines.values(): sp.set_color('gold' if k == HI[2] else 'cyan'); sp.set_linewidth(2.5 if k == HI[2] else 1)
# the level-8 box: its 19 x 19 samples at level 8 (oriented gradient, LoG)
m8, th8, lg8 = channels(level(8)); y8, x8 = int(round(cy / 8)), int(round(cx / 8)); s8 = np.s_[y8 - HB:y8 + HB + 1, x8 - HB:x8 + HB + 1]
x8a = X0 + 2 * GW + 2 * SEP; fig.text((x8a + LT + .06) / W, (Y2 + GH + .1) / H, 'level 8 box', ha='center', fontsize=10)
a8 = ax_in(x8a, Y2, LT, LT); a8.imshow(orient_rgb(m8[s8], th8[s8], np.percentile(m8[s8], 99)), interpolation='bilinear')
b8 = ax_in(x8a + LT + .12, Y2, LT, LT); v8 = np.percentile(np.abs(lg8[s8]), 99); b8.imshow(lg8[s8], cmap='RdBu_r', vmin=-v8, vmax=v8, interpolation='bilinear')
for a_ in (a8, b8):
    for sp in a_.spines.values(): sp.set_color('gold'); sp.set_linewidth(2.5)
wh = ax_in(x8a + 2 * LT + .12 + SEP, Y2 + (GH - WHL) / 2, WHL, WHL); yy, xx = np.mgrid[-1:1:201j, -1:1:201j]; rr = np.hypot(yy, xx)
wheel = hsv_to_rgb(np.stack([(np.arctan2(yy, xx) % np.pi) / np.pi, np.ones_like(rr), np.clip(rr, 0, 1)], -1)); wheel[rr > 1] = 1
wh.imshow(wheel); wh.set_title('gradient direction\n(mod 180°)', fontsize=10)
for sp in wh.spines.values(): sp.set_visible(False)
for ax in fig.axes: ax.set_xticks([]); ax.set_yticks([])
out = 'figures/fig2_node_on_eye'; fig.savefig(out + '.pdf', dpi=200); fig.savefig(out + '.png', dpi=150); print('saved', out, FACE, EYE)
