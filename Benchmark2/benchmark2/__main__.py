"""CLI: python -m benchmark2 prepare|models|run|validate."""
import argparse, json, sys, traceback
from .prepare import prepare, ROOT
from .models import prepare_models

def validate():
    samples=json.loads((ROOT/'data/samples.json').read_text(encoding='utf-8'))
    models=json.loads((ROOT/'config/models.json').read_text(encoding='utf-8'))
    assert len(samples)==17 and len({x['id'] for x in samples})==17
    assert len(models)==6 and all(s['cyrillic'] for s in models)
    print('CHECK: 17 original ROIs, 6 Cyrillic engines, 3 passes incl. warmup')
    try:
        import platform
        print('Python:',platform.python_version(),', platform:',platform.platform())
    except ImportError:pass

def main():
    # Windows consoles may use a code page that cannot encode progress glyphs.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='backslashreplace')
    p=argparse.ArgumentParser(description='Benchmark2 — 6 Cyrillic OCR models, 17 real ROIs')
    p.add_argument('command',choices=['prepare','models','validate','run'])
    p.add_argument('--offline',action='store_true',help='Never download from network')
    p.add_argument('--no-browser',action='store_true')
    p.add_argument('--no-prepare',action='store_true',help='Run only on already prepared local files')
    a=p.parse_args()
    if a.command=='validate':validate();return
    if a.command=='prepare':prepare(download_missing=not a.offline);return
    if a.command=='models':
        for m in prepare_models(download_missing=not a.offline):print(m)
        return
    from .dashboard import launch
    from .runner import run, pctstring
    srv=None
    if not a.no_browser:srv,url=launch()
    last=[-1]
    def progress(s):
        if s['completed']!=last[0]:
            last[0]=s['completed']
            elapsed=s['elapsed_sec'];eta=s['eta_sec'];rest=('—' if eta is None else f'{eta:.1f}s')
            print(f"{pctstring(s['completed'],s['total'])} {s['model']} {s.get('pass_number',0)}/3 "
                  f"cell={s.get('cell','—')} elapsed={elapsed:.1f}s ETA={rest} "
                  f"CPU={s.get('cpu_percent','—')}%",flush=True)
    error=None
    try:
        dest=run(offline=a.offline,auto_prepare=not a.no_prepare,progress_cb=progress)
        print('[FINISHED] report:',dest/'report.html',flush=True)
        print('[FINISHED] csv:',dest/'raw.csv',flush=True)
    except Exception:
        traceback.print_exc();error=True
    if srv:
        print('Отчёт открыт в браузере. Для остановки локального сервера нажмите Enter.',flush=True)
        try:input()
        except (KeyboardInterrupt,EOFError):pass
        srv.shutdown()
    if error:sys.exit(1)
if __name__=='__main__':main()
