"""Is 512 px enough for our method? Compares discovery runs side by side (e.g. results/U1000 at 1024 px vs results/U1000_r512).
Uses the official landmarks of the U training faces for diagnosis only (no test faces).
Per run: TAU, clusters of >= 3; per part (eye, brow, nose, mouth): clusters that sit on it (>= 60 % of members within 0.25 IOD),
the largest such cluster, and the share of the U faces that at least one of them covers; for the 32 largest clusters: the share
of members that sit on any of the 6 labelled points.
Rules, fixed before the runs (2026-10-05): at 1024 px, among no level 1 / single boxes / 2 x 2 sub-boxes, take the simplest whose
part coverages and top-32 share are all within 5 points of the best version's; then 512 px is enough if, for that version, none of
them drops by > 5 points from 1024 px (no level 1 needs no 512 run: levels 2-8 already work on images shrunk 2x or more).
Run from this folder:  FFHQ_DIR=... python compare_res.py results/U1000 results/U1000_r512"""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_common as pc
IDX = dict(eye=(0, 1), brow=(2, 3), nose=(4,), mouth=(5,)); SHARE, T, TOP = .6, .25, 32
PTS = {}
def nearest(fi, y, x):
    """(index of the nearest labelled point, distance / IOD)."""
    if fi not in PTS: PTS[fi] = (pc.points(fi), pc.iod(fi))
    P, iod = PTS[fi]; d = np.hypot(P[:6, 0] - y, P[:6, 1] - x) / iod; return int(np.argmin(d)), float(d.min())
def summary(res):
    D = pickle.load(open(f'{res}/clusters.pkl', 'rb')); KPS, C = D['KPS'], D['clusters']; nU = len(D['U'])
    hit = [[nearest(*KPS[m[0]]) for m in c['members']] for c in C]
    row = dict(res=D.get('res', 1024), level1=D.get('level1', 'sub') + ('' if D.get('BOXL', {}).get(2, 19) == 19 else f"/{D['BOXL'].get(1, '-')},{D['BOXL'][2]}"), TAU=D['TAU'], n=len(C))
    for part, idx in IDX.items():
        on = [ci for ci, h in enumerate(hit) if np.mean([j in idx and d <= T for j, d in h]) >= SHARE]
        faces = {KPS[m[0]][0] for ci in on for m in C[ci]['members']}
        row[part] = (len(on), max((C[ci]['n'] for ci in on), default=0), len(faces) / nU)
    top = sorted(range(len(C)), key=lambda ci: -C[ci]['n'])[:TOP]
    row['top'] = np.mean([d <= T for ci in top for _, d in hit[ci]])
    return row
rows = [(r, summary(r)) for r in sys.argv[1:]]
print(f'{"run":28s} {"px":>5s} {"lv1":>12s} {"TAU":>6s} {"clust":>6s}  ' + '  '.join(f'{p + " (n/max/cover)":>24s}' for p in IDX) + f'  {"top-32 on part":>14s}')
for r, w in rows:
    print(f'{r:28s} {w["res"]:5d} {w["level1"]:>12s} {w["TAU"]:6.3f} {w["n"]:6d}  ' +
          '  '.join(f'{w[p][0]:>8d} {w[p][1]:>6d} {w[p][2]:>7.0%}' for p in IDX) + f'  {w["top"]:14.0%}')
if len(rows) == 2:
    a, b = rows[0][1], rows[1][1]
    gaps = {p: 100 * (b[p][2] - a[p][2]) for p in IDX}; gaps['top-32'] = 100 * (b['top'] - a['top'])
    print('\ndifference (second - first), percentage points:', {k: round(v, 1) for k, v in gaps.items()})
    print(f'largest drop {-min(gaps.values()):.1f} points, largest gain {max(gaps.values()):.1f} points')
