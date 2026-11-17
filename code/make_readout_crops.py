"""Readout crops for protocol v3 (generated once; every method uses this file): each of the 2000 readout faces (the training
pool minus U = 1000 draw 0, in training-pool order) cropped twice with the test rule -- a 768 x 768 crop of the aligned 1024 px
image (384 at 512), the 6 labelled points (eye centres, brow centres, nose tip, mouth centre) at least 100 px inside the crop,
offsets uniform over the feasible range per image (integers), seed 1. Offsets are (x, y), as in paper_protocol.json.
Run from this folder:  python make_readout_crops.py   -> v3/readout_crops.json"""
import os, json, numpy as np, paper_common as pc
P = json.load(open('paper_protocol.json'))['data']; S, M = P['test_crop']['size'], P['test_crop']['min_margin_px']
u1000 = set(pc.u_draw(1000, 0)); faces = [int(f) for f in P['train_pool'] if int(f) not in u1000]; assert len(faces) == 2000
rng = np.random.default_rng(1); out = []
for f in faces:
    p = pc.points(f)                                                     # (6, 2) as (y, x)
    lo = np.maximum(0, np.ceil(p.max(0) + M - S)).astype(int); hi = np.minimum(1024 - S, np.floor(p.min(0) - M)).astype(int)
    assert (lo <= hi).all(), f
    out.append(dict(face=f, crops=[[int(rng.integers(lo[1], hi[1] + 1)), int(rng.integers(lo[0], hi[0] + 1))] for _ in range(2)]))
os.makedirs('v3', exist_ok=True)
json.dump(dict(note='2000 readout faces x 2 crops; offsets (x, y) into the aligned 1024 px image; crop = aligned[y:y+768, x:x+768]; '
               'at 512 px: offsets / 2, size 384. Test rule (6 points >= 100 px inside), uniform per image, seed 1.', size=S,
               faces=out), open('v3/readout_crops.json', 'w'))
print('saved v3/readout_crops.json:', len(out), 'faces; first', out[0])
