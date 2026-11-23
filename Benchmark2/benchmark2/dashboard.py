"""Local-only HTTP viewer. Browser displays live progress and final HTML report."""
from __future__ import annotations
import json, mimetypes, os, threading, webbrowser, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import unquote
from .prepare import ROOT

DASHBOARD = ROOT/'web'/'dashboard.html'
PORT = int(os.environ.get('BENCHMARK2_PORT', '18018'))

def launch():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=unquote(self.path.split('?',1)[0]);latest=ROOT/'results'/'latest.txt'
            run_id=latest.read_text(encoding='utf-8').strip() if latest.exists() else None
            if path in ('/','/index.html'):
                target=DASHBOARD;typ='text/html'
            elif path == '/api/status':
                target=ROOT/'results'/'status.json';typ='application/json'
            elif path=='/api/summary':
                target=ROOT/'results'/run_id/'summary.json' if run_id else ROOT/'missing.json';typ='application/json'
            elif path=='/report':
                target=ROOT/'results'/run_id/'report.html' if run_id else ROOT/'missing.html';typ='text/html'
            elif path.startswith('/data/cells/'):
                target=ROOT/'data/cells'/Path(path).name;typ='image/png'
            elif path.startswith('/results/') and run_id:
                allowed={'report.html','raw.csv','raw.jsonl','summary.json','environment.json','inputs.json','status.json'}
                part=Path(path).name
                if part not in allowed:self.send_error(403);return
                target=ROOT/'results'/run_id/part;typ=mimetypes.guess_type(part)[0] or 'application/octet-stream'
            else:self.send_error(404);return
            if not target.is_file():self.send_error(404);return
            content=target.read_bytes()
            self.send_response(200);self.send_header('Content-Type',typ+'; charset=utf-8');self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(content)));self.end_headers()
            try:self.wfile.write(content)
            except BrokenPipeError:pass
        def log_message(self,*args):pass
    try:
        srv=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
    except OSError as exc:
        raise RuntimeError(f'Dashboard port 127.0.0.1:{PORT} is busy: {exc}. '
                           f'Set BENCHMARK2_PORT to another port.') from exc
    thread=threading.Thread(target=srv.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{srv.server_port}/'
    webbrowser.open(url)
    print(f'[DASHBOARD] {url}',flush=True)
    return srv,url
