"""Download the protocol's FFHQ images (paper_protocol.json: train pool + test) from the official NVlabs release.

Each file is fetched by its Google Drive id from ffhq-dataset-v2.json, checked against the official MD5 and size,
written to $FFHQ_DIR (default ./ffhq_nvlabs)/<file_path>, and made read-only. Files already present with the right MD5 are skipped.
"""
import hashlib, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('FFHQ_DIR', HERE / 'ffhq_nvlabs'))   # needs ROOT/ffhq-dataset-v2.json from the NVlabs release first
proto = json.loads((HERE / 'paper_protocol.json').read_text())
meta = json.loads((ROOT / 'ffhq-dataset-v2.json').read_text())
ids = sorted(set(proto['data']['train_pool'] + proto['data']['test_primary'] + proto['data']['test_secondary']))
specs = [meta[str(i)]['image'] for i in ids]


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def fetch(spec, attempts=6):
    dst = ROOT / spec['file_path']
    if dst.exists() and dst.stat().st_size == spec['file_size'] and md5(dst) == spec['file_md5']: return 'skip'
    dst.parent.mkdir(parents=True, exist_ok=True); tmp = dst.with_suffix('.tmp')
    fid = spec['file_url'].split('id=')[1]
    for a in range(attempts):
        r = subprocess.run(['curl', '-sSfL', '--max-time', '300', '-o', str(tmp),
                            f'https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t'], capture_output=True)
        if r.returncode == 0 and tmp.stat().st_size == spec['file_size'] and md5(tmp) == spec['file_md5']:
            tmp.rename(dst); os.chmod(dst, 0o444); return 'ok'
        time.sleep(min(60, 5 * 2 ** a))
    tmp.unlink(missing_ok=True)
    return 'FAIL: ' + (r.stderr.decode()[-200:] if r.returncode else f'size/md5 mismatch ({tmp.stat().st_size if tmp.exists() else 0} bytes)')


if __name__ == '__main__':
    t0 = time.time(); done = {'ok': 0, 'skip': 0}; fails = []
    with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
        for k, (spec, res) in enumerate(zip(specs, ex.map(fetch, specs))):
            if res in done: done[res] += 1
            else: fails.append((spec['file_path'], res))
            if (k + 1) % 200 == 0: print(f'{k + 1}/{len(specs)}: {done} failed {len(fails)} ({time.time() - t0:.0f}s)', flush=True)
    print(f'finished: {done}, failed {len(fails)} ({time.time() - t0:.0f}s)')
    for f in fails[:20]: print('  ', f)
    Path(ROOT / 'download_failures.json').write_text(json.dumps(fails, indent=1))
