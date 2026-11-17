"""For Figure 3: pick the largest eye cluster of a discovery run and its best-matching eye member, and pack what the figure needs
(the cluster's level-2 aligned-mean template, the energy floor, the member's face image and keypoint) into one folder, so the
figure can be drawn on the laptop. An eye cluster: >= 60 % of members within 0.25 IOD of an eye centre. Best member: the
highest match score among the members on an eye. Labels are used only to pick what to show.
Optional third argument: a cluster id instead of the largest eye cluster; fourth: a member face id to show.
Run on the desktop from this folder:  FFHQ_DIR=... python fig3_pick.py results/U300_single_r1024 fig3_data [cluster]"""
import sys, os, shutil, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_common as pc
RES, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
D = pickle.load(open(f'{RES}/clusters.pkl', 'rb')); KPS = D['KPS']
def on_eye(k):
    fi, y, x = KPS[k]; P = pc.points(fi); return min(np.hypot(P[j, 0] - y, P[j, 1] - x) for j in (0, 1)) / pc.iod(fi) <= .25
eyes = [(c['n'], ci) for ci, c in enumerate(D['clusters']) if np.mean([on_eye(m[0]) for m in c['members']]) >= .6]
n, ci = max(eyes)
if len(sys.argv) > 3: ci = int(sys.argv[3]); n = D['clusters'][ci]['n']           # a chosen eye cluster (the teaser's)
C = D['clusters'][ci]
cand = [m for m in C['members'] if on_eye(m[0])]
if len(sys.argv) > 4: cand = [m for m in cand if KPS[m[0]][0] == int(sys.argv[4])]      # a chosen member face (e.g. an adult)
mem = max(cand, key=lambda m: m[2]); k, best, sc = mem; fi, y, x = KPS[k]
print(f'{len(eyes)} eye clusters; largest: cluster {ci} with {n} members; best eye member: face {fi} at ({y:.0f}, {x:.0f}), score {sc:.3f}')
np.savez(f'{OUT}/fig3_data.npz', template2=C['template'][2], floor2=np.array(D['FLOOR'][2]), face=fi, kp=np.array([y, x]),
         best2=np.array(best[2]), score=sc, cluster=ci, n=n, node2_feat=D['NODE'][2]['feat'], res=D.get('res', 1024))
dst = f'{OUT}/face.png'; os.path.exists(dst) and os.chmod(dst, 0o644); shutil.copyfile(pc.image_path(fi), dst); print('saved', OUT)
