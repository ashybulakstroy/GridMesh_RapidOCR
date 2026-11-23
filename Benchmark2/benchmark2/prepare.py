"""Reproducible import of 17 genuine Torgsoft ROI cells from existing JPEGs."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, urllib.request
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get('BENCHMARK2_BASE_PROJECT', r'C:\Work\Projects\Prj_18_OCR_2'))
REPO = 'https://raw.githubusercontent.com/ashybulakstroy/GridMesh_RapidOCR/main/benchmark/data/images/'
TESSDATA = 'https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda: f.read(1024 * 1024), b''):
            h.update(c)
    return h.hexdigest()


def download(url, target):
    """No download occurs at import time; all downloads run only by explicit prepare command."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + '.download')
    try:
        req = urllib.request.Request(url, headers={'User-Agent':'Benchmark2/1.0'})
        with urllib.request.urlopen(req, timeout=90) as response, temp.open('wb') as out:
            shutil.copyfileobj(response, out)
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def prepare(*, download_missing=True, fetch_tessdata=True):
    samples = json.loads((ROOT/'data/samples.json').read_text(encoding='utf-8'))
    if len(samples) != 17 or len({x['id'] for x in samples}) != 17:
        raise ValueError('Expected exactly 17 unique input cells')
    sources = ROOT/'data/sources'; cells = ROOT/'data/cells'
    sources.mkdir(parents=True, exist_ok=True); cells.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in sorted({s['source'] for s in samples}):
        first = next(s for s in samples if s['source'] == name)
        dest = sources/name
        possible = [BASE/'benchmark/data/images'/name, BASE/'App/tests'/name, dest]
        known = next((p for p in possible if p.is_file() and sha(p) == first['image_sha256']), None)
        if known is None:
            if not download_missing:
                raise FileNotFoundError(f'No verified source image {name}; inspected {possible}')
            print(f'[DOWNLOAD] original JPEG {name}', flush=True)
            download(REPO+name,dest)
        elif known != dest:
            shutil.copy2(known,dest)
        if sha(dest) != first['image_sha256']:
            raise ValueError(f'IMAGE_SHA256_MISMATCH: {name}')
    for s in samples:
        with Image.open(sources/s['source']) as img:
            im = ImageOps.exif_transpose(img).convert('RGB')
            w,h = im.size
            x0,y0,x1,y1 = s['bbox']
            a=max(0,min(w-1,int(round(x0*w)))); b=max(1,min(w,int(round(x1*w))))
            c=max(0,min(h-1,int(round(y0*h)))); d=max(1,min(h,int(round(y1*h))))
            if a>=b or c>=d: raise ValueError(f'Bad ROI: {s["id"]}')
            crop=im.crop((a,c,b,d))
            path=cells/(s['id']+'.png')
            crop.save(path,format='PNG')
            manifest.append({**s,'file':str(path.relative_to(ROOT)).replace('\\','/'),
                             'source_width':w,'source_height':h,'width':crop.width,'height':crop.height,
                             'crop_sha256':sha(path),'pixel_bbox':[a,c,b,d]})
    out=ROOT/'data/cells_manifest.json'
    out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if fetch_tessdata:
        dest=ROOT/'models/tessdata';dest.mkdir(parents=True,exist_ok=True)
        for lang in ('rus','eng'):
            path=dest/(lang+'.traineddata')
            if not path.is_file() or path.stat().st_size < 100000:
                if not download_missing: continue
                print(f'[DOWNLOAD] Tesseract language {lang}', flush=True)
                try:
                    download(TESSDATA+lang+'.traineddata',path)
                except Exception as exc:
                    print(f'[WARNING] tessdata {lang} unavailable: {exc}; Tesseract will be marked UNAVAILABLE',flush=True)
    print(f'[READY] {len(manifest)} true-size ROIs from 3 SHA256-verified JPEGs -> {out}', flush=True)
    return manifest


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--offline',action='store_true');ap.add_argument('--no-tessdata',action='store_true')
    a=ap.parse_args();prepare(download_missing=not a.offline,fetch_tessdata=not a.no_tessdata)
if __name__=='__main__':main()
