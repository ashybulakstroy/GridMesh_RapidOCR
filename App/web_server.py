from pathlib import Path
from contextlib import asynccontextmanager
import json
from threading import Lock, Thread
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from .ocr import warmup
from .ocr.experiment_manual import apply_user_receipt
from .config import setting, output_directory, display_path

ROOT = Path(__file__).resolve().parents[1]
@asynccontextmanager
async def lifespan(app):
    # Startup does not wait for model loading/ONNX inference.
    warmup.start(is_ocr_busy=lambda: processing.locked())
    yield

app = FastAPI(title='OCR чеков — лаборатория', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=ROOT/'App'/'web'), name='static')
jobs = {}
jobs_lock = Lock()
processing = Lock()


def run_process(*args, **kwargs):
    # Do not import heavy OCR pipeline before HTTP server is ready.
    from .pipeline import process
    return process(*args, **kwargs)

@app.get('/')
def index():
    return FileResponse(ROOT/'App'/'web'/'index.html')

@app.get('/api/status')
def status():
    with jobs_lock:
        return {'processing':processing.locked(), 'running_jobs':[k for k,v in jobs.items() if v['status']=='running'], 'ocr_warmup':warmup.snapshot()}

@app.post('/api/process')
async def recognize(file: UploadFile = File(...), rectify: bool = Form(False)):
    data = await file.read(25_000_001)
    if len(data) > 25_000_000: raise HTTPException(413, 'Файл слишком большой')
    await run_in_threadpool(warmup.wait_primary)
    if not processing.acquire(blocking=False):
        raise HTTPException(409, 'Дождитесь завершения текущего чека')
    try:
        return await run_in_threadpool(run_process, data, rectify=rectify)
    except Exception as error:
        raise HTTPException(422, str(error)) from error
    finally:
        processing.release()

@app.post('/api/jobs', status_code=202)
async def create_job(file: UploadFile = File(...), rectify: bool = Form(False)):
    data = await file.read(25_000_001)
    if len(data) > 25_000_000: raise HTTPException(413, 'Файл слишком большой')
    await run_in_threadpool(warmup.wait_primary)
    if not processing.acquire(blocking=False): raise HTTPException(409, 'Дождитесь завершения текущего чека')
    job_id = uuid4().hex
    with jobs_lock:
        jobs[job_id] = {'id':job_id, 'status':'running', 'progress':{}, 'result':None, 'error':None}
    def publish(update):
        with jobs_lock:
            jobs[job_id]['progress'].update(update)
    def worker():
        try:
            result = run_process(data, publish, job_id, rectify=rectify)
            with jobs_lock:
                jobs[job_id].update(status='done', result=result)
        except Exception as exc:
            with jobs_lock:
                jobs[job_id].update(status='failed', error=str(exc))
        finally:
            processing.release()
    Thread(target=worker, daemon=True).start()
    return {'id':job_id}

@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    with jobs_lock:
        if job_id not in jobs: raise HTTPException(404, 'Задание не найдено')
        return dict(jobs[job_id])

@app.get('/api/jobs/{job_id}/preview')
def get_preview(job_id: str):
    with jobs_lock:
        if job_id not in jobs: raise HTTPException(404, 'Задание не найдено')
    output_dir = output_directory()
    directory = (output_dir / job_id).resolve()
    if not directory.is_relative_to(output_dir) or not (directory/'rectified.png').is_file():
        raise HTTPException(404, 'Сетка ещё не готова')
    return FileResponse(directory/'rectified.png', media_type='image/png')

@app.get('/api/jobs/{job_id}/early')
def download_early_result(job_id: str):
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(404, 'Задание не найдено')
    output_dir = output_directory()
    directory = (output_dir / job_id).resolve()
    path = directory/'e4_early_result.json'
    if not directory.is_relative_to(output_dir) or not path.is_file():
        raise HTTPException(404, 'Предварительный результат ещё не готов')
    return FileResponse(path, media_type='application/json',filename='e4_early_result.json')

@app.put('/api/jobs/{job_id}/receipt')
def save_edited_receipt(job_id: str, receipt: dict):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or job['status'] != 'done': raise HTTPException(404, 'Готовое задание не найдено')
        directory = (ROOT / job['result']['job_directory']).resolve()
    if not directory.is_relative_to(output_directory()) or receipt.get('job_id') != job_id or not isinstance(receipt.get('rows'), list):
        raise HTTPException(400, 'Некорректные данные чека')
    # Server independently recomputes Decimal arithmetic and locks valid
    # browser-confirmed figures; never trust client-provided math flags.
    try:
        clean = apply_user_receipt(job['result'], receipt)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    target = directory/'receipt_edited.json'
    tmp = directory/'receipt_edited.json.tmp'
    tmp.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(target)
    # Persist user ground truth and model-vs-user feedback separately.
    truth=directory/'numeric_user_truth.json'
    truth_tmp=directory/'numeric_user_truth.json.tmp'
    truth_tmp.write_text(json.dumps({'manual_truth':clean.get('numeric_user_truth',[]),
        'model_feedback':clean.get('numeric_model_feedback',[])},
        ensure_ascii=False, indent=2),encoding='utf-8')
    truth_tmp.replace(truth)
    with jobs_lock:
        jobs[job_id]['result']={**job['result'],**clean}
    return {'saved': True, 'file': display_path(target), 'receipt': clean}
