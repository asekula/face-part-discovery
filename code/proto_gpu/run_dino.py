"""Protocol v3, DINO co-parts (Amir et al. 2021, official code github.com/ShirAmir/dino-vit-features, part_cosegmentation.py),
split into fit (on the U faces) and assign (readout and test crops) -> v3/parts/dino_U<U>_K<K>.npz.
Kept from the official code: ViT-S/8, key facet, layer 11, no binning; saliency votes (thresh 0.065, 75 %); first clustering
with the elbow rule (0.975, 1..14 clusters, faiss k-means niter 300 nredo 10, every 100th descriptor); second clustering of the
salient descriptors into num_parts = K (same k-means, every 100th); per image: squared distances to the part centroids and the
nearest background centroid, minus their max, nearest upsampling, dense CRF (Gaussian sxy 3 compat 50; bilateral sxy 5 srgb 13
compat 15; 10 iterations), argmax.
Changed for the protocol (handoff_competitors.md): fitted on the U faces only, then applied to new crops; stride 8 (stride 4
does not fit at 512 px); images at 512 px (U faces) and 384 px crops (the readout / test crops at 512 px); part centre = the
CRF-probability-weighted centroid (SCOPS formula; 'dino'), with the hard-label centroid of the official output kept as 'dinohard';
sensitivity row 'dinoallfg': the whole image as foreground (parts fitted on all descriptors, no background label), since faces fill
the frame; no crop augmentation (the default). (2026-10-06: the first run saved hard-label centroids only; up to 70 % of parts were
missing at K = 32 because the saliency step kept 1 of 6-7 first-stage clusters, about 15 % of each image, as foreground.)
Run on the desktop from proto_gpu (torch venv with faiss-cpu and pydensecrf):
  FFHQ_DIR=... python -u run_dino.py [--repo ../dino-vit-features] [--U 100 1000] [--K 4 8 16 32] [--no_crf]"""
import argparse, os, sys, json, time, numpy as np, torch
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_common as pc
ap = argparse.ArgumentParser(); ap.add_argument('--repo', default='../dino-vit-features'); ap.add_argument('--U', type=int, nargs='+', default=[100, 1000])
ap.add_argument('--K', type=int, nargs='+', default=[4, 8, 16, 32]); ap.add_argument('--out', default='../v3/parts')
ap.add_argument('--crops', default='../v3/readout_crops.json'); ap.add_argument('--no_crf', action='store_true'); ap.add_argument('--workers', type=int, default=8)
args = ap.parse_args(); sys.path.insert(0, os.path.abspath(args.repo))
from extractor import ViTExtractor
import faiss
if not args.no_crf: import pydensecrf.densecrf as dcrf
os.makedirs(args.out, exist_ok=True); dev = 'cuda'; t0 = time.time()
def log(*a): print(f'[{time.time() - t0:7.0f}s]', *a, flush=True)
LAYER, FACET, BIN, THRESH, VOTES, ELBOW, SAMPLE = 11, 'key', False, 0.065, 75, 0.975, 100
ex = ViTExtractor('dino_vits8', stride=8, device=dev)
from torchvision import transforms
NORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=ex.mean, std=ex.std)])

def image(face, crop=None):
    """PIL RGB at 512 px scale: the whole face (512 x 512) or a 768 crop of the 1024 image reduced to 384."""
    im = Image.open(pc.image_path(face)).convert('RGB')
    if crop is not None: x, y = crop; im = im.crop((x, y, x + 768, y + 768))
    return im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
@torch.no_grad()
def descriptors(im, saliency=False):
    b = NORM(im)[None].to(dev); d = ex.extract_descriptors(b, LAYER, FACET, BIN)[0, 0].cpu().numpy().astype(np.float32)
    s = ex.extract_saliency_maps(b)[0].cpu().numpy() if saliency else None
    return d, s, tuple(ex.num_patches)
def kmeans(x, k):
    km = faiss.Kmeans(d=x.shape[1], k=k, niter=300, nredo=10); km.train(np.ascontiguousarray(x, np.float32)); return km

# ---------------- fit on the U faces (U = 1000 draw 0 prefix, as for ours)
FIT = {}
for U in args.U:
    faces = pc.u_draw(1000, 0)[:U]; descs, sals = [], []
    for i, f in enumerate(faces):
        d, s, _ = descriptors(image(f), saliency=True); faiss.normalize_L2(d); descs.append(d); sals.append(s)
        if i % 200 == 199: log(f'U={U}: descriptors {i + 1}/{U}')
    allx = np.concatenate(descs); sampled = np.concatenate([d[::SAMPLE] for d in descs]); ssd = []
    for n in range(1, 15):                                                  # first clustering, elbow rule
        km = kmeans(sampled, n); D_, lab = km.index.search(allx, 1); ssd.append(D_.sum() / len(allx))
        if len(ssd) > 1 and ssd[-1] > ELBOW * ssd[-2]: break
    labs = np.split(lab[:, 0], np.cumsum([len(d) for d in descs])[:-1]); votes = np.zeros(km.k)
    for l, s in zip(labs, sals):
        for c in range(km.k):
            if (l == c).any() and s[l == c].mean() > THRESH: votes[c] += 1
    salient = np.where(votes >= np.ceil(U * VOTES / 100))[0]; bg = [c for c in range(km.k) if c not in salient]
    fg = np.concatenate([d[np.isin(l, salient)][::SAMPLE] for d, l in zip(descs, labs)])
    FIT[U] = dict(bg=km.centroids[bg].copy(), parts={K: kmeans(fg, K).centroids.copy() for K in args.K},
                  parts_allfg={K: kmeans(sampled, K).centroids.copy() for K in args.K})        # sensitivity: whole image = foreground
    log(f'U={U}: first clustering {km.k} clusters, salient {list(salient)}, {len(fg)} foreground descriptors sampled; parts fitted for K = {args.K}')
    del descs, allx

# ---------------- assign: readout (2000 x 2) and test (1000) crops; one ViT pass per crop serves every (U, K)
def centres(args_):
    """Part centres from one crop's distances (official assignment + CRF): (soft, hard), each (K, 2) as (y, x) in 384 px crop
    coordinates. soft = centroid weighted by the part's CRF probability map (SCOPS formula, the protocol's rule; never missing);
    hard = centroid of the pixels whose final label is the part (the official output; missing if none). bg=None: no background."""
    d, hw, rgb, parts, bg, no_crf = args_; K = len(parts)
    dp = ((d[:, None] - parts[None]) ** 2).sum(2)
    u = dp if bg is None or len(bg) == 0 else np.concatenate([dp, ((d[:, None] - bg[None]) ** 2).sum(2).min(1, keepdims=True)], 1)
    u = u.reshape(*hw, u.shape[1]); u = u - u.max(-1, keepdims=True)
    fy, fx = rgb.shape[0] // u.shape[0], rgb.shape[1] // u.shape[1]                     # nearest upsampling (as torch.nn.Upsample),
    u = np.ascontiguousarray(np.repeat(np.repeat(u, fy, 0), fx, 1)[:rgb.shape[0], :rgb.shape[1]], np.float32)   # numpy only in workers
    if no_crf:
        e = np.exp(-(u - u.min(-1, keepdims=True))); Q = (e / e.sum(-1, keepdims=True)).reshape(-1, u.shape[-1]).T
    else:
        c = dcrf.DenseCRF2D(u.shape[1], u.shape[0], u.shape[2]); c.setUnaryEnergy(np.ascontiguousarray(u.reshape(-1, u.shape[-1]).T))
        c.addPairwiseGaussian(sxy=(3, 3), compat=50, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        c.addPairwiseBilateral(sxy=5, srgb=13, rgbim=np.ascontiguousarray(rgb), compat=15, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = np.array(c.inference(10))                                           # (labels, H*W) probabilities
    final = np.argmax(Q, axis=0).reshape(rgb.shape[:2]); H, W = rgb.shape[:2]
    yy, xx = np.mgrid[:H, :W]; yy, xx = yy.ravel().astype(np.float64), xx.ravel().astype(np.float64)
    soft = np.full((K, 2), np.nan, np.float32); hard = np.full((K, 2), np.nan, np.float32)
    for k in range(K):
        w = Q[k].astype(np.float64)
        if w.sum() > 0: soft[k] = (w @ yy / w.sum(), w @ xx / w.sum())
        yx = np.argwhere(final == k)
        if len(yx): hard[k] = yx.mean(0)
    return soft, hard
RC = json.load(open(args.crops))['faces']; TF = pc.test_faces('secondary'); OFF = json.load(open('paper_protocol.json'))['data']['test_crop']['offsets']
VAR = {'dino': ('parts', True, 0), 'dinohard': ('parts', True, 1), 'dinoallfg': ('parts_allfg', False, 0)}   # name: (fit, background?, soft/hard)
SET = [(v, U, K) for v in VAR for U in args.U for K in args.K]; part_file = f'{args.out}/dino.partial.npz'
R = {s: np.full((len(RC), 2, s[2], 2), np.nan, np.float32) for s in SET}; T = {s: np.full((len(TF), s[2], 2), np.nan, np.float32) for s in SET}
done = np.zeros(len(RC) + len(TF), bool)
if os.path.exists(part_file):
    z = np.load(part_file, allow_pickle=True); R, T, done = z['R'].item(), z['T'].item(), z['done']; log('resuming from', part_file)
jobs = [('r', n) for n in range(len(RC))] + [('t', n) for n in range(len(TF))]
with Pool(args.workers) as pool:                                                   # workers use numpy and pydensecrf only
    for q, (kind, n) in enumerate(jobs):
        if done[q]: continue
        crops = [(RC[n]['face'], c) for c in RC[n]['crops']] if kind == 'r' else [(TF[n], OFF[str(TF[n])])]
        for ci, (f, c) in enumerate(crops):
            im = image(f, c); d, _, hw = descriptors(im); faiss.normalize_L2(d); rgb = np.array(im)
            runs = [(f, bgk, U, K) for f, bgk in (('parts', True), ('parts_allfg', False)) for U in args.U for K in args.K]
            res = {(f, U, K): r for (f, _, U, K), r in zip(runs, pool.map(centres, [(d, hw, rgb, FIT[U][f][K], FIT[U]['bg'] if bgk else None, args.no_crf)
                                                                                for f, bgk, U, K in runs]))}
            for (v, U, K) in SET:
                fk, _, which = VAR[v]; r = res[(fk, U, K)][which]
                if kind == 'r': R[(v, U, K)][n, ci] = r
                else: T[(v, U, K)][n] = r
        done[q] = True
        if q % 20 == 19: log(f'{q + 1}/{len(jobs)} faces')
        if q % 200 == 199: np.savez(part_file, R=np.array(R, dtype=object), T=np.array(T, dtype=object), done=done); log('saved progress')
wall = time.time() - t0
NOTE = dict(dino='official defaults; part centre = CRF-probability-weighted centroid (protocol rule)',
            dinohard='official defaults; part centre = centroid of the final hard labels (official output)',
            dinoallfg='sensitivity: whole image treated as foreground (no saliency filtering); soft centroid')
for v, U, K in SET:
    s = (v, U, K); meta = dict(method=v, U=U, K=K, variant=NOTE[v], crf=not args.no_crf, stride=8, load_size='512 (fit) / 384 crops', wall_time_s=round(wall), code='proto_gpu/run_dino.py')
    np.savez(f'{args.out}/{v}_U{U}_K{K}.npz', readout=R[s], test=T[s], readout_missing=np.isnan(R[s][..., 0]), test_missing=np.isnan(T[s][..., 0]), meta=json.dumps(meta))
if os.path.exists(part_file): os.remove(part_file)
log(f'saved {args.out}/{{dino,dinohard,dinoallfg}}_U*_K*.npz')
