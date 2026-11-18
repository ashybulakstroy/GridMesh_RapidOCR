from pathlib import Path
import json
from threading import Lock, Thread
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from .pipeline import process
from .config import setting

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title='OCR чеков — лаборатория')
app.mount('/static', StaticFiles(directory=ROOT/'App'/'web'), name='static')
jobs = {}
jobs_lock = Lock()
processing = Lock()

@app.get('/')
def index():
    return FileResponse(ROOT/'App'/'web'/'index.html')

@app.get('/api/status')
def status():
    with jobs_lock:
        return {'processing':processing.locked(), 'running_jobs':[k for k,v in jobs.items() if v['status']=='running']}

@app.post('/api/process')
async def recognize(file: UploadFile = File(...), rectify: bool = Form(False)):
    data = await file.read(25_000_001)
    if len(data) > 25_000_000: raise HTTPException(413, 'Файл слишком большой')
    if not processing.acquire(blocking=False):
        raise HTTPException(409, 'Дождитесь завершения текущего чека')
    try:
        return await run_in_threadpool(process, data, rectify=rectify)
    except Exception as error:
        raise HTTPException(422, str(error)) from error
    finally:
        processing.release()

@app.post('/api/jobs', status_code=202)
async def create_job(file: UploadFile = File(...), rectify: bool = Form(False)):
    data = await file.read(25_000_001)
    if len(data) > 25_000_000: raise HTTPException(413, 'Файл слишком большой')
    if not processing.acquire(blocking=False): raise HTTPException(409, 'Дождитесь завершения текущего чека')
    job_id = uuid4().hex
    with jobs_lock:
        jobs[job_id] = {'id':job_id, 'status':'running', 'progress':{}, 'result':None, 'error':None}
    def publish(update):
        with jobs_lock:
            jobs[job_id]['progress'].update(update)
    def worker():
        try:
            result = process(data, publish, job_id, rectify=rectify)
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
    directory = (ROOT / setting('OUTPUT_DIR', 'data/output') / job_id).resolve()
    if not directory.is_relative_to(ROOT) or not (directory/'rectified.png').is_file():
        raise HTTPException(404, 'Сетка ещё не готова')
    return FileResponse(directory/'rectified.png', media_type='image/png')

@app.put('/api/jobs/{job_id}/receipt')
def save_edited_receipt(job_id: str, receipt: dict):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or job['status'] != 'done': raise HTTPException(404, 'Готовое задание не найдено')
        directory = (ROOT / job['result']['job_directory']).resolve()
    if not directory.is_relative_to(ROOT) or receipt.get('job_id') != job_id or not isinstance(receipt.get('rows'), list):
        raise HTTPException(400, 'Некорректные данные чека')
    clean = {k:v for k,v in receipt.items() if k not in ('original','rectified')}
    target = directory/'receipt_edited.json'
    tmp = directory/'receipt_edited.json.tmp'
    tmp.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(target)
    return {'saved': True, 'file': str(target.relative_to(ROOT)).replace('\\','/')}
