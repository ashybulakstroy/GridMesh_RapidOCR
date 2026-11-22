"""Opt-in RapidOCR recognizer ensemble using benchmark model IDs.

Read options from App/config.setting and catalog from benchmark/config/models.json
when available. Existing recognition primary stays untouched. No Tesseract.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from time import perf_counter

from ..config import ROOT, setting, positive_int
from .experiment_rec_trace import record_rec
from .debug_performance import report_issue

# Reuse secondary ONNX sessions across receipts, never one per cell.
_SESSIONS = {}
_VERIFIED_FILES = {}


def load_catalog(root=ROOT):
    root=Path(root)
    preferred=root/'benchmark/config/models.json'
    fallback=Path(__file__).with_name('experiment_model_catalog.json')
    path=preferred if preferred.is_file() else fallback
    data=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,list):
        raise ValueError('Model catalog must contain a JSON list')
    catalog={}
    for item in data:
        identifier=item.get('id')
        if not isinstance(identifier,str) or not identifier or identifier in catalog:
            raise ValueError('Duplicate/invalid model ID')
        catalog[identifier]=item
    return catalog


def requested_ids(task, *, read=setting):
    if read('OCR_E2_ENABLE','0') != '1':
        return []
    defaults={'numeric':'v6_small_official', 'footer':'v6_small_official', 'vat':'v6_small_official'}
    if task not in defaults:
        raise ValueError('Unknown model task')
    raw=read('OCR_E2_'+task.upper()+'_MODELS',defaults[task])
    # Whitespace around comma is harmless. Empty list explicitly disables task.
    return list(dict.fromkeys(v.strip() for v in raw.split(',') if v.strip()))


def _file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def inspect_model(model, root=ROOT):
    """Validate explicit models. Official routed models are not downloaded here."""
    if not model.get('enabled'):
        return {'status':'disabled'}
    if 'model_path' not in model:
        return {'status':'rapidocr_managed_cache_unverified',
                'note':'RapidOCR may download this official model on first use if cache is empty'}
    p=Path(model['model_path'])
    if not p.is_absolute():p=Path(root)/p
    if not p.is_file():
        return {'status':'missing_file','path':str(p)}
    expected=model.get('model_sha256')
    if expected:
        key=(str(p),p.stat().st_size,p.stat().st_mtime_ns,expected)
        verified=_VERIFIED_FILES.get(key)
        if verified is None:
            verified=(_file_hash(p).lower()==expected.lower())
            _VERIFIED_FILES[key]=verified
        if not verified:
            return {'status':'hash_mismatch','path':str(p)}
    return {'status':'ready','path':str(p)}


class ExperimentModelPool:
    """Lazy model resolution; per-receipt calls are capped; errors stay visible."""
    def __init__(self, *, root=ROOT, read=setting, catalog=None):
        self.root=Path(root)
        self.read=read
        self.catalog=load_catalog(self.root) if catalog is None else catalog
        self.call_limit=positive_int('OCR_E2_MAX_MODEL_CALLS',12) if read is setting else int(read('OCR_E2_MAX_MODEL_CALLS','12'))
        if self.call_limit<1:raise ValueError('OCR_E2_MAX_MODEL_CALLS must be positive')
        self.calls=0
        self.errors=[]
        self.observations=[]
        self.selected={}
        for task in ('numeric','footer','vat'):
            ids=requested_ids(task,read=read)
            unknown=[i for i in ids if i not in self.catalog]
            if unknown:
                raise ValueError(f'Unknown OCR model ID(s): {unknown}')
            self.selected[task]=ids

    def models_for(self,task):
        return list(self.selected.get(task,()))

    def _get_session(self,model_id,model):
        from .engine import engine, _lock, _LANE, onnx_threads
        if model_id=='v5_cyrillic_mobile':
            return engine()
        status=inspect_model(model,self.root)
        if status['status'] not in ('ready','rapidocr_managed_cache_unverified'):
            raise FileNotFoundError(f"{model_id}: {status['status']} {status.get('path','')}")
        path=status.get('path')
        # RapidOCR wraps mutable pre/post-processing state as well as ONNX.
        # Each inference lane must own its worker, including secondary models.
        token=(_LANE.get(),model_id,path,str(self.root))
        with _lock:
            if token not in _SESSIONS:
                from rapidocr import RapidOCR
                from rapidocr.utils.typings import OCRVersion,ModelType
                params={
                    'Global.use_det':False,'Global.use_cls':False,
                    'Rec.rec_batch_num':1,
                    'EngineConfig.onnxruntime.intra_op_num_threads':onnx_threads(),
                    'EngineConfig.onnxruntime.inter_op_num_threads':1,
                }
                if path:
                    params['Rec.model_path']=path
                    params['Rec.lang_type']=model.get('lang','cyrillic')
                    params['Rec.rec_img_shape']=list(model.get('rec_img_shape',[3,48,640]))
                    if model.get('rec_keys_path'):
                        keys=Path(model['rec_keys_path'])
                        params['Rec.rec_keys_path']=str(keys if keys.is_absolute() else self.root/keys)
                else:
                    params['Rec.ocr_version']=getattr(OCRVersion,model['ocr_version'])
                    params['Rec.model_type']=getattr(ModelType,model['model_type'])
                    params['Rec.lang_type']=model.get('lang','en')
                _SESSIONS[token]=RapidOCR(params=params)
            return _SESSIONS[token]

    def recognize(self,model_id,image,stage):
        model=self.catalog.get(model_id)
        if model is None:
            raise ValueError(f'Unregistered model: {model_id}')
        stamp=perf_counter()
        if self.calls>=self.call_limit:
            return {'model':model['name'],'model_id':model_id,
                    'error':'model_call_budget_exhausted','wall_ms':0.0}
        self.calls+=1
        reached=False
        invoked=False
        model_start=stamp
        try:
            from .engine import _lock
            with _lock:
                session=self._get_session(model_id,model)
                model_start=perf_counter()
                invoked=True
                result=session.recognize_txt([image])
            record_rec(model=model_id,ms=(perf_counter()-model_start)*1000,success=True)
            reached=True
            if result.txts is None or not result.txts:
                raise RuntimeError('REC-only returned no text')
            text=str(result.txts[0]);confidence=float(result.scores[0])
            obs={'text':text,'confidence':confidence,'model':model.get('name',model_id),
                 'model_id':model_id,'stage':stage,'wall_ms':round((perf_counter()-stamp)*1000,3)}
            self.observations.append(obs)
            return obs
        except Exception as exc:
            if invoked and not reached:
                record_rec(model=model_id,ms=(perf_counter()-model_start)*1000,success=False,
                           note=f'{type(exc).__name__}: {exc}')
            item={'model':model.get('name',model_id),'model_id':model_id,
                  'stage':stage,'error':f'{type(exc).__name__}: {exc}',
                  'wall_ms':round((perf_counter()-stamp)*1000,3)}
            self.errors.append(item)
            report_issue(source='ExperimentModelPool',operation=stage,
                         kind='model_exception',error=exc,
                         context={'model_id':model_id},fallback='preserve_original')
            return item

    def stats(self):
        return {'enabled':self.read('OCR_E2_ENABLE','0')=='1',
                'models':self.selected,'model_calls':self.calls,
                'model_call_limit':self.call_limit,'successful':len(self.observations),
                'errors':self.errors}
