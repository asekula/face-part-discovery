"""Protocol v3, DFF (Deep Feature Factorization, Collins et al. ECCV 2018), reimplemented in PyTorch -> v3/parts/dff_U<U>_K<K>.npz.
As in the paper: VGG-19 (ImageNet weights) conv5_4 features after ReLU; the features of all positions of the U faces stacked
into one non-negative matrix A (positions x 512 channels) and factorised A ~ W H by NMF with K factors (multiplicative updates,
Lee & Seung, Frobenius loss, 200 iterations, random non-negative initialisation, seed 0 -- the update rule of the official code);
rows of H are the K concepts, columns of W reshaped per image are the concept heat maps.
For new crops (readout, test): H fixed, W solved by the same multiplicative updates (200 iterations); part centre = the
heat-map-weighted centroid (SCOPS), mapped to pixels at the conv5_4 stride (16 px, cell centre); missing if the heat map is ~0.
Images: U faces at 512 px (32 x 32 feature grid), crops at 384 px (24 x 24), ImageNet normalisation.
Run on the desktop from proto_gpu (torch venv with torchvision):
  FFHQ_DIR=... python -u run_dff.py [--U 100 1000] [--K 4 8 16 32]"""
import argparse, os, sys, json, time, numpy as np, torch, torchvision
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_common as pc
ap = argparse.ArgumentParser(); ap.add_argument('--U', type=int, nargs='+', default=[100, 1000]); ap.add_argument('--K', type=int, nargs='+', default=[4, 8, 16, 32])
ap.add_argument('--out', default='../v3/parts'); ap.add_argument('--crops', default='../v3/readout_crops.json'); ap.add_argument('--iters', type=int, default=200)
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True); dev = 'cuda'; t0 = time.time(); EPS = 1e-9
def log(*a): print(f'[{time.time() - t0:7.0f}s]', *a, flush=True)
vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features[:36].eval().to(dev)   # up to relu5_4
NORM = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
def image(face, crop=None):
    im = Image.open(pc.image_path(face)).convert('RGB')
    if crop is not None: x, y = crop; im = im.crop((x, y, x + 768, y + 768))
    return im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
@torch.no_grad()
def feats(im):
    f = vgg(NORM(im)[None].to(dev))[0]; return f.permute(1, 2, 0).reshape(-1, f.shape[0]), f.shape[1:]   # (h*w, 512), (h, w)
@torch.no_grad()
def nmf(A, k, iters, H=None, seed=0):
    """A ~ W H, non-negative, multiplicative updates; if H is given it stays fixed and only W is solved."""
    g = torch.Generator(device=dev).manual_seed(seed); scale = (A.mean() / k).sqrt()
    W = torch.rand(A.shape[0], k, device=dev, generator=g) * scale
    fixed = H is not None
    if not fixed: H = torch.rand(k, A.shape[1], device=dev, generator=g) * scale
    for _ in range(iters):
        if not fixed: H *= (W.T @ A) / (W.T @ W @ H + EPS)
        W *= (A @ H.T) / (W @ (H @ H.T) + EPS)
    return W, H

# ---------------- fit: concepts H per (U, K) on the U faces (U = 1000 draw 0 prefix, as for every method)
CON = {}
for U in args.U:
    A = torch.cat([feats(image(f))[0] for f in pc.u_draw(1000, 0)[:U]]); log(f'U={U}: feature matrix {tuple(A.shape)}')
    for K in args.K:
        W, H = nmf(A, K, args.iters); CON[(U, K)] = H
        log(f'U={U} K={K}: relative error {float((A - W @ H).norm() / A.norm()):.3f}')
    del A, W; torch.cuda.empty_cache()

# ---------------- assign: readout (2000 x 2) and test (1000) crops
def centres(F, hw, H):
    W, _ = nmf(F, H.shape[0], args.iters, H=H); h, w = hw; Wm = W.T.reshape(-1, h, w)          # (K, h, w) heat maps
    ys = torch.arange(h, device=dev, dtype=torch.float32)[:, None] * 16 + 7.5; xs = torch.arange(w, device=dev, dtype=torch.float32)[None] * 16 + 7.5
    tot = Wm.sum((1, 2)); cy = (Wm * ys).sum((1, 2)) / tot.clamp_min(EPS); cx = (Wm * xs).sum((1, 2)) / tot.clamp_min(EPS)
    out = torch.stack([cy, cx], 1).cpu().numpy().astype(np.float32); out[(tot <= 1e-6 * F.sum()).cpu().numpy()] = np.nan; return out
RC = json.load(open(args.crops))['faces']; TF = pc.test_faces('secondary'); OFF = json.load(open('paper_protocol.json'))['data']['test_crop']['offsets']
SET = list(CON); R = {s: np.full((len(RC), 2, s[1], 2), np.nan, np.float32) for s in SET}; T = {s: np.full((len(TF), s[1], 2), np.nan, np.float32) for s in SET}
for n, r in enumerate(RC):
    for c, crop in enumerate(r['crops']):
        F, hw = feats(image(r['face'], crop))
        for s in SET: R[s][n, c] = centres(F, hw, CON[s])
    if n % 200 == 199: log(f'readout {n + 1}/{len(RC)}')
for n, f in enumerate(TF):
    F, hw = feats(image(f, OFF[str(f)]))
    for s in SET: T[s][n] = centres(F, hw, CON[s])
    if n % 200 == 199: log(f'test {n + 1}/{len(TF)}')
wall = time.time() - t0
for U, K in SET:
    meta = dict(method='dff', U=U, K=K, features='VGG-19 conv5_4 (relu5_4), ImageNet weights', nmf_iters=args.iters, wall_time_s=round(wall),
                code='proto_gpu/run_dff.py (PyTorch reimplementation)')
    np.savez(f'{args.out}/dff_U{U}_K{K}.npz', readout=R[(U, K)], test=T[(U, K)], readout_missing=np.isnan(R[(U, K)][..., 0]),
             test_missing=np.isnan(T[(U, K)][..., 0]), meta=json.dumps(meta))
log(f'saved {args.out}/dff_U*_K*.npz')
