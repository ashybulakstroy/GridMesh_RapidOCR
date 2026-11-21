"""Standalone RapidOCR OCR-only benchmark; no dependency on the working App."""
from __future__ import annotations
import datetime as dt
import json, os, platform, sys, time, traceback, subprocess, threading
from pathlib import Path
from .measure import gt_fields, verify_math, sha256_file, extract_rois, recognize_cells, evaluate, is_numeric_field
from .report import write_report

ROOT=Path(__file__).resolve().parents[1]

def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def read_ground_truth():
    return {d['receipt_id']:d for p in sorted((ROOT/'data'/'ground_truth').glob('*.json')) for d in [load_json(p)]}


def selected_receipts(scope, config):
    gt=read_ground_truth()
    if scope=='quick':return [gt[r] for r in config['quick_receipt_ids']]
    return list(gt.values())


def check_inputs(gts=None, show=True):
    gts=gts if gts is not None else list(read_ground_truth().values())
    all_errors=[];statuses=[]
    for gt in gts:
        rid=gt['receipt_id']; errs=verify_math(gt)
        source=ROOT/'data'/'images'/gt['source_image'];annotation_file=ROOT/'data'/'annotations'/(rid+'.json')
        found=source.is_file();ann_found=annotation_file.is_file()
        if not found:errs.append(f"IMAGE_MISSING: {gt['source_image']}")
        if not ann_found:errs.append(f"ANNOTATION_MISSING: {rid}.json")
        if found and ann_found:
            try:
                a=load_json(annotation_file)
                if a.get('receipt_id')!=rid:errs.append('RECEIPT_ID_MISMATCH')
                if a.get('image_file')!=gt['source_image']:errs.append('IMAGE_FILENAME_MISMATCH')
                if not a.get('image_sha256'):errs.append('SHA256_MISSING')
                elif a['image_sha256'].lower()!=sha256_file(source):errs.append('SHA256_MISMATCH')
                seen=[]
                for item in a.get('rois',[]):
                    field=item.get('field_id');box=item.get('bbox')
                    if field not in gt_fields(gt):errs.append(f'UNKNOWN_FIELD: {field}')
                    if not isinstance(box,list) or len(box)!=4 or not all(isinstance(n,(int,float)) for n in box) or not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):
                        errs.append(f'BAD_BBOX: {field}')
                    seen.append(field)
                for field in gt_fields(gt):
                    if field not in seen:errs.append(f'ROI_MISSING: {field}')
            except (OSError,ValueError,KeyError,TypeError) as exc:errs.append(f'BAD_ANNOTATION: {exc}')
        verification=gt.get('verification',{})
        agent_verified=bool(verification.get('numeric_verified_by_agent') and verification.get('names_verified_by_agent'))
        status='READY' if not errs and gt.get('approved_by_user',False) else ('READY_AGENT_VERIFIED' if not errs and agent_verified else ('READY_GT_DRAFT' if not errs else 'BLOCKED'))
        if show:
            print(f'[{status}] {rid}: {len(gt["products"])} products; total {gt["printed_total"]} KZT')
            if gt.get('approved_by_user') is not True:print('    GT_DRAFT: утверждение владельцем ещё не выставлено' + ('; проверено агентом' if agent_verified else ''))
            for e in errs:print('    - '+e)
        statuses.append({'receipt_id':rid,'status':status,'errors':errs})
        all_errors.extend((rid, x) for x in errs)
    return statuses,all_errors


def plan(scope='quick',mode='both',sweep='none',repeats=None):
    config=load_json(ROOT/'config'/'experiment.json');gts=selected_receipts(scope,config)
    repeats=repeats or config['repeat_count']
    if repeats<1:raise ValueError('repeats must be >= 1')
    models=load_json(ROOT/'config'/'models.json');enabled=[m for m in models if m.get('enabled') and m['id'] in config['model_ids']]
    modes=['single_cell','batch'] if mode=='both' else [mode]
    configs=3 if sweep=='threads' else (5 if sweep=='batch' else 1)
    print('ПОДГОТОВКА К ЛОКАЛЬНОМУ БЕНЧМАРКУ — БЕЗ ЗАПУСКА OCR')
    print(f'Чеки: {len(gts)}: '+', '.join(gt['receipt_id'] for gt in gts))
    print('Модели: '+', '.join(m['name'] for m in enabled))
    print('Режимы: '+', '.join(modes))
    print(f'Измеряемых повторов на конфигурацию/чек/режим: {repeats}; прогрев: {config["warmups"]} раз на модель')
    print(f'Планируемое число измеряемых попыток: {len(gts)*len(enabled)*len(modes)*configs*repeats}')
    print('single_cell = строго ОДИН вызов recognize_txt([crop]) на каждую ячейку; сумма измерений отдельно')
    print('batch = recognize_txt([crop,...]) на пачку; итоговое время отдельно')
    print('Загрузка моделей и запуск запрещены без --confirm-load; sweep дополнительно требует --confirm-sweep.')


def _resolve_model_path(value):
    path=Path(value)
    return path if path.is_absolute() else ROOT.parent/path


def _create_recognizer(model, intra, batch_size):
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import LangRec, OCRVersion, ModelType
    params={
       'Global.use_det':False,'Global.use_cls':False,
       'Rec.rec_batch_num':batch_size,
       'EngineConfig.onnxruntime.intra_op_num_threads':intra,
       'EngineConfig.onnxruntime.inter_op_num_threads':1,
    }
    if model.get('model_path'):
        path=_resolve_model_path(model['model_path'])
        if not path.is_file():raise FileNotFoundError(f"MODEL_MISSING: {path}")
        expected=model.get('model_sha256')
        if expected and sha256_file(path).lower()!=expected.lower():raise ValueError(f"MODEL_SHA256_MISMATCH: {path}")
        params['Rec.model_path']=str(path)
        params['Rec.rec_img_shape']=list(model.get('rec_img_shape',[3,48,640]))
        if model.get('lang'):params['Rec.lang_type']=getattr(LangRec,str(model['lang']).upper())
    else:
        params.update({'Rec.ocr_version':getattr(OCRVersion,model['ocr_version']),
                       'Rec.model_type':getattr(ModelType,model['model_type']),
                       'Rec.lang_type':model['lang']})
    return RapidOCR(params=params)


def _load_cells(gt, crop_source="roi"):
    from PIL import Image, ImageOps
    import numpy as np
    start=time.perf_counter_ns()
    path=ROOT/'data'/'images'/gt['source_image']
    with Image.open(path) as im:
        rgb=ImageOps.exif_transpose(im).convert('RGB');img=np.ascontiguousarray(np.asarray(rgb)[:,:,::-1])
    ann=load_json(ROOT/'data'/'annotations'/(gt['receipt_id']+'.json'))
    if crop_source == 'gridmesh_total':
        from .gridmesh import extract_gridmesh
        cells, _ = extract_gridmesh(img, ann, include_total=True, cache_dir=ROOT/"data"/"gridmesh_coordinates")
    else:
        cells=extract_rois(img,ann)
    return cells,(time.perf_counter_ns()-start)/1e6


def _sampled_memory():
    try:
        import psutil
        process=psutil.Process()
        return process.memory_info().rss/(1024**2)
    except ImportError:return None


class ResourceSampler:
    """Sampled peak RSS, not a guaranteed instantaneous memory high-water mark."""
    def __init__(self):
        self.stop_flag=threading.Event()
        self.samples=0
        self.peak=None
        self.thread=None
        try:
            import psutil
            self.process=psutil.Process()
        except ImportError:
            self.process=None
    def _sample(self):
        while not self.stop_flag.wait(.05):
            try:
                value=self.process.memory_info().rss/(1024**2)
                self.peak=max(self.peak or value,value)
                self.samples+=1
            except Exception:break
    def start(self):
        if self.process is None:return
        try:self.peak=self.process.memory_info().rss/(1024**2)
        except Exception:return
        self.thread=threading.Thread(target=self._sample,daemon=True)
        self.thread.start()
    def finish(self):
        self.stop_flag.set()
        if self.thread:self.thread.join(timeout=1)
        if self.process:
            try:
                val=self.process.memory_info().rss/(1024**2)
                self.peak=max(self.peak or val,val)
            except Exception:pass
        return self.peak


def _run_receipt(gt, recognizer, model, mode, batch_size, repeat, extra_meta):
    w0=time.perf_counter_ns();c0=time.process_time_ns()
    out={'receipt_id':gt['receipt_id'],'model_id':model['id'],'mode':mode,'repeat':repeat,'status':'ERROR',**extra_meta}
    before=_sampled_memory()
    monitor=ResourceSampler();monitor.start()
    try:
        cells,crop_ms=_load_cells(gt, extra_meta.get("crop_source", "roi"))
        field_scope=extra_meta.get('field_scope','all' if model['cyrillic'] else 'numeric_only')
        if field_scope=='numeric_only':cells=[(f,c) for f,c in cells if is_numeric_field(f)]
        out['field_scope']=field_scope
        result=recognize_cells(cells,recognizer,mode,batch_size)
        quality=evaluate(gt,result['texts'],script_supported=model['cyrillic'],field_scope=field_scope)
        wall=(time.perf_counter_ns()-w0)/1e6;cpu=(time.process_time_ns()-c0)/1e6
        rss=_sampled_memory()
        out.update(status='OK',receipt_total_wall_ms=round(wall,4),cpu_ms=round(cpu,4),
                   cpu_core_equivalents=round(cpu/wall,4) if wall else None,
                   process_cpu_percent_one_core_basis=round(100*cpu/wall,2) if wall else None,
                   start_rss_mb=before,end_rss_mb=rss,
                   crop_ms=round(crop_ms,4),quality=quality,**result)
    except Exception as exc:
        out.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),
                   receipt_total_wall_ms=round((time.perf_counter_ns()-w0)/1e6,4),
                   cpu_ms=round((time.process_time_ns()-c0)/1e6,4))
    finally:
        out['peak_rss_sampled_mb']=monitor.finish()
        out['rss_samples']=monitor.samples
        out['rss_sampling_interval_ms']=50
    return out


def run(*,scope='quick',mode='both',sweep='none',repeats=None,confirm_load=False,confirm_sweep=False,crop_source="roi"):
    if not confirm_load:raise RuntimeError('NO_RUN: missing --confirm-load. Компьютер занят; без разрешения OCR не запускается.')
    if sweep!='none' and not confirm_sweep:raise RuntimeError('NO_SWEEP: нужен флаг --confirm-sweep и разрешение пользователя')
    config=load_json(ROOT/'config'/'experiment.json')
    gts=selected_receipts(scope,config)
    _,errors=check_inputs(gts,show=True)
    if errors:raise RuntimeError('RUN_BLOCKED: исходные фотографии/аннотации/эталоны не готовы; не запускать модели')
    repeats=repeats or config['repeat_count']
    if repeats<1:raise ValueError('repeats must be >= 1')
    models=[m for m in load_json(ROOT/'config'/'models.json') if m.get('enabled') and m['id'] in config['model_ids']]
    modes=['single_cell','batch'] if mode=='both' else [mode]
    if sweep=='threads':sweep_config=[(i,config['batch_size']) for i in config['parameters_for_later_sweep']['onnx_intra_threads']]
    elif sweep=='batch':sweep_config=[(config['onnx_intra_threads'],k) for k in config['parameters_for_later_sweep']['batch_size']]
    else:sweep_config=[(config['onnx_intra_threads'],config['batch_size'])]
    # Prepare missing geometry before any timed trials or model loading.
    if crop_source == 'gridmesh_total':
        for gt in gts:
            _load_cells(gt, crop_source)
    now=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    directory=ROOT/'data'/'output'/now;directory.mkdir(parents=True,exist_ok=False)
    info={'run_id':now,'status':'RUNNING','started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
          'platform':platform.platform(),'python':sys.version,'logical_cpus':os.cpu_count(),
          'scope':scope,'sweep':sweep,'modes':modes,'repeats':repeats,'warmups':config['warmups'],'models':[x['id'] for x in models],
          'crop_source':crop_source,
          'geometry_timing_policy':'coordinate cache prepared before model loading and timed trials; OCR call timing excludes geometry and crop preparation',
          'geometry_policy':'annotation-assisted matching; synthetic annotated total; no automatic footer detection' if crop_source=='gridmesh_total' else None,
          'roi_policy':{'numeric':'tight_ink_bbox','names':'multi_line_split_dehyphenated'},
          'record_source':'DIRECT_LOCAL_MEASUREMENTS_ONLY',
          'environment':{'PYTHONHASHSEED':os.environ.get('PYTHONHASHSEED')},
          'image_sha256':{g['receipt_id']:sha256_file(ROOT/'data'/'images'/g['source_image']) for g in gts}}
    try:
        import psutil
        info.update(psutil_version=psutil.__version__,host_cpu_before_percent=psutil.cpu_percent(interval=0.1),
                    total_ram_gb=round(psutil.virtual_memory().total/(1024**3),2))
    except ImportError:info['psutil_status']='NOT_INSTALLED'
    try:
        from importlib.metadata import version
        info['rapidocr_version']=version('rapidocr');info['onnxruntime_version']=version('onnxruntime')
    except Exception as exc:info['version_info_error']=str(exc)
    (directory/'run_manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    raw=directory/'raw.jsonl'
    try:
        with raw.open('a',encoding='utf-8') as fh:
            for intra,bsize in sweep_config:
                try:
                    import cv2
                    cv2.setNumThreads(config['opencv_threads'])
                except ImportError:pass
                for model in models:
                    tag={'crop_source':crop_source,'onnx_intra_threads':intra,'batch_size_configured':bsize,'opencv_threads':config['opencv_threads']}
                    print(f"MODEL {model['id']} intra={intra} batch_size={bsize}",flush=True)
                    init_started=time.perf_counter_ns()
                    try:
                        engine=_create_recognizer(model,intra,bsize)
                        init_ms=(time.perf_counter_ns()-init_started)/1e6
                        sample_cells,_=_load_cells(gts[0], crop_source)
                        if not model['cyrillic']:sample_cells=[(f,c) for f,c in sample_cells if is_numeric_field(f)]
                        test_img=sample_cells[0][1]
                        warm_started=time.perf_counter_ns()
                        for _ in range(config['warmups']):
                            warm=engine.recognize_txt([test_img])
                            if not getattr(warm,'txts',None):raise RuntimeError('warmup returned no recognition')
                        warm_ms=(time.perf_counter_ns()-warm_started)/1e6
                        tag.update(engine_init_ms=round(init_ms,3),warmup_ms=round(warm_ms,3))
                    except Exception as exc:
                        failed={'receipt_id':'ALL','model_id':model['id'],'mode':'MODEL_INIT','repeat':None,
                                'status':'ERROR','error':str(exc),'traceback':traceback.format_exc(),**tag}
                        fh.write(json.dumps(failed,ensure_ascii=False)+'\n');fh.flush()
                        print('ERROR loading model: '+str(exc),flush=True)
                        continue
                    for gt in gts:
                        for repeat in range(1,repeats+1):
                            order=modes if repeat%2==1 else list(reversed(modes))
                            for m in order:
                                row=_run_receipt(gt,engine,model,m,bsize,repeat,tag)
                                fh.write(json.dumps(row,ensure_ascii=False)+'\n');fh.flush()
                                print(f" {gt['receipt_id']} {m} #{repeat}: {row.get('status')} {row.get('receipt_total_wall_ms')} ms; "
                                      f"sum cell={row.get('sum_cell_ocr_ms')} ms",flush=True)
        info['status']='FINISHED'
    except BaseException:
        info['status']='INTERRUPTED';info['exception']=traceback.format_exc()
        raise
    finally:
        info['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        (directory/'run_manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        write_report(directory)
        print('RESULT_DIR:',directory,flush=True)
    return directory
