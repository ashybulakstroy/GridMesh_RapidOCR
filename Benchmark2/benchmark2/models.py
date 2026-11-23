"""Fetch only the required recognition ONNX weights, never a GPU package."""
from __future__ import annotations
import hashlib, json, shutil, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=Path(os.environ.get('BENCHMARK2_BASE_PROJECT',r'C:\Work\Projects\Prj_18_OCR_2'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def prepare_models(*,download_missing=True):
    specs=json.loads((ROOT/'config/models.json').read_text(encoding='utf-8'))
    reports=[]
    for s in specs:
        if 'model_path' not in s:continue
        rel=Path(s['model_path']);target=ROOT/rel
        source=BASE/rel
        try:
            if source.is_file() and not target.is_file():
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
            if not target.is_file() and download_missing:
                from huggingface_hub import hf_hub_download
                name=s['hf_repo']; print(f'[MODEL DOWNLOAD] {s["id"]} from {name}',flush=True)
                downloaded=hf_hub_download(repo_id=name,filename=s.get('hf_file','inference.onnx'))
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(downloaded,target)
            if not target.is_file():
                reports.append({'id':s['id'],'status':'MISSING','path':str(target)})
                continue
            digest=sha(target)
            expected=s.get('model_sha256')
            if expected and expected.lower()!=digest.lower():
                reports.append({'id':s['id'],'status':'HASH_MISMATCH','path':str(target),'actual_sha256':digest,'expected_sha256':expected})
                continue
            reports.append({'id':s['id'],'status':'READY','path':str(target),'sha256':digest,'size_bytes':target.stat().st_size})
        except Exception as exc:
            reports.append({'id':s['id'],'status':'ERROR','path':str(target),'error':f'{type(exc).__name__}: {exc}'})
            print(f'[MODEL ERROR] {s["id"]}: {exc}; benchmark will record UNAVAILABLE',flush=True)
    out=ROOT/'data/models_manifest.json'
    out.write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
    return reports
if __name__=='__main__':
    for x in prepare_models():print(x)
