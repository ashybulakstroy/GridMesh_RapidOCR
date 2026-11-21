"""Build a self-contained HTML report from existing native pipeline measurements."""
import html
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/output/cpu_threads_experiment'
NAMES = {'small': 'Малый чек · 3 строки', 'medium': 'Средний чек · 13 строк', 'long': 'Длинный чек · 30 строк'}


def esc(value):
    return html.escape(str(value))


def number(value):
    from decimal import Decimal
    return format(Decimal(value), 'f') if value is not None else 'не распознано'


def label(profile):
    name = f'{profile["onnx"]}/{profile["opencv"]}'
    return name + (' (повтор базы)' if profile['tag'].startswith('08_') else '')


def records(profile, source):
    return [r for r in profile['records'] if not r['warmup'] and r['source'] == source]


def main():
    profiles = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(DATA.glob('[0-9]*.json'))]
    assert len(profiles) == 9, 'Report requires the nine completed profiles; it does not run benchmarks'
    baseline = {name: records(profiles[0], name)[0]['quality'] for name in NAMES}
    parts = []
    for source, title in NAMES.items():
        base = median(r['performance']['wall_ms'] for r in records(profiles[0], source))
        maximum = max(median(r['performance']['wall_ms'] for r in records(p, source)) for p in profiles)
        chart, rows, phase_rows = [], [], []
        for p in profiles:
            rr = records(p, source)
            wall = median(r['performance']['wall_ms'] for r in rr)
            cpu = median(r['performance']['process_cpu_ms'] for r in rr)
            stable = sum(r['quality'] == baseline[source] for r in rr)
            matches = sum(r['quality']['math_match'] for r in rr)
            vat = sum(r['quality']['vat']['status'] == 'match' for r in rr)
            bad = 'bad' if stable != len(rr) else 'good' if matches == vat == len(rr) else 'caution'
            chart.append(f'<div class="bar-row"><b>{esc(label(p))}</b><div class="track"><div class="bar" style="width:{wall/maximum*100:.2f}%"></div></div><span>{wall/1000:.3f} с</span></div>')
            rows.append(f'<tr><th>{esc(label(p))}</th><td>{wall/1000:.3f}</td><td>{min(r["performance"]["wall_ms"] for r in rr)/1000:.3f}–{max(r["performance"]["wall_ms"] for r in rr)/1000:.3f}</td><td>{base/wall:.2f}× ({(base/wall-1)*100:+.1f}%)</td><td>{cpu/1000:.3f}</td><td>{median(r["performance"]["cpu_core_equivalents"] for r in rr):.2f}</td><td class="{bad}">{stable}/{len(rr)}</td><td>{matches}/{len(rr)}</td><td>{vat}/{len(rr)}</td></tr>')
        phases = [s['name'] for s in records(profiles[0], source)[0]['performance']['samples']]
        for phase in phases:
            values = []
            for p in profiles:
                values.append(median(next((s['wall_ms'] for s in r['performance']['samples'] if s['name'] == phase), 0) for r in records(p, source))/1000)
            phase_rows.append('<tr><th>'+esc(phase)+'</th>'+''.join(f'<td>{v:.3f}</td>' for v in values)+'</tr>')
        ref = baseline[source]
        reference = f'База: напечатано {number(ref["total"])}, рассчитано {number(ref["computed_total"])} ₸; НДС: {esc(ref["vat"]["status"])}.'
        parts.append(f'''<section class="source" data-source="{source}"><h2>{title}</h2><p>{reference}</p>
        <div class="chart">{''.join(chart)}</div><p class="muted">Длина столбца — время полного процесса. Меньше — быстрее. Подписи ONNX/OpenCV.</p>
        <div class="scroll"><table><thead><tr><th>Потоки</th><th>Медиана, с</th><th>Мин.–макс., с</th><th>Скорость к базе</th><th>CPU, с</th><th>Среднее ядер</th><th>Числа как в базе</th><th>Итог совпал</th><th>НДС совпал</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
        <details><summary>Штатные этапы обработки: медиана времени, с</summary><div class="scroll"><table><thead><tr><th>Этап</th>{''.join('<th>'+esc(label(p))+'</th>' for p in profiles)}</tr></thead><tbody>{''.join(phase_rows)}</tbody></table></div></details></section>''')

    diffs = []
    runs = []
    for p in profiles:
        for r in p['records']:
            if r['warmup']:
                continue
            q, ref = r['quality'], baseline[r['source']]
            differences = []
            for i, (old, new) in enumerate(zip(ref['rows'], q['rows'])):
                for field in ('quantity', 'price', 'printed_total'):
                    if old.get(field) != new.get(field):
                        differences.append(f'Строка {i+1}, {field}: {number(old.get(field))} → {number(new.get(field))}')
            if len(q['rows']) != len(ref['rows']):
                differences.append(f'Число строк {len(ref["rows"])} → {len(q["rows"])}')
            if q != ref:
                diffs.append(f'<tr><th>{esc(label(p))}</th><td>{NAMES[r["source"]]}</td><td>{r["repeat"]+1}</td><td>{"<br>".join(esc(d) for d in differences)}</td><td>{number(q["computed_total"])}</td><td>{r["batch"]["numeric_roi_skipped"]}</td><td><code>{r["job_id"]}</code></td></tr>')
            runs.append(f'<tr><td>{esc(label(p))}</td><td>{NAMES[r["source"]]}</td><td>{r["repeat"]+1}</td><td>{r["performance"]["wall_ms"]/1000:.3f}</td><td>{number(q["total"])}</td><td>{number(q["computed_total"])}</td><td>{esc(q["vat"]["status"])}</td><td>{r["batch"]["numeric_roi_ms"]:.3f}</td><td><code>{r["job_id"]}</code></td></tr>')

    machine = profiles[0]['machine']
    runtime_rows = []
    for p in profiles:
        intra = sorted({s['intra'] for s in p['sessions'].values()})
        runtime_rows.append(f'<tr><th>{esc(label(p))}</th><td>{esc(intra)}</td><td>{p["opencv_actual"]}</td><td>{esc(", ".join(p["sessions"]))}</td></tr>')
    css = '''body{font:16px/1.5 system-ui,sans-serif;background:#f0f4f8;color:#24334a;margin:0}main{max-width:1400px;margin:auto;padding:24px}h1{font-size:32px;line-height:1.2}h2{font-size:23px}.card,section,details.top{background:white;border:1px solid #dbe3ed;border-radius:12px;padding:20px;margin:20px 0}.conclusion{background:#e4f1ff;border-left:5px solid #2863aa}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px;border-bottom:1px solid #dce4ed;text-align:left}thead{background:#eef4fb}th{font-weight:600}.scroll{overflow:auto}code{font-size:12px}.muted{color:#607089;font-size:14px}.good{color:#13713b;font-weight:700}.bad{color:#b4232c;font-weight:700}.caution{color:#946400;font-weight:700}.bar-row{display:grid;grid-template-columns:170px minmax(100px,1fr) 90px;gap:10px;align-items:center;font-size:14px;margin:8px 0}.track{height:20px;background:#edf1f6;border-radius:3px}.bar{height:100%;background:#3979bf;border-radius:3px}summary{cursor:pointer;font-weight:600}select{padding:9px;font:inherit}.pill{display:inline-block;padding:4px 9px;background:#fff2ce;border-radius:6px}.hidden{display:none}@media(max-width:650px){main{padding:12px}h1{font-size:25px}.bar-row{grid-template-columns:110px minmax(60px,1fr) 70px;font-size:12px}.card,section{padding:12px}}'''
    report = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Потоки CPU и скорость OCR — 20 ноября 2026</title><style>{css}</style></head><body><main>
    <p class="muted">Проект 18 · 20 ноября 2026 · эксперимент на этой машине</p><h1>Увеличение потоков не дает пропорционального ускорения OCR</h1>
    <div class="card conclusion"><b>Текущая рекомендация: оставить ONNX=2 и OpenCV=2.</b><p>4/2 ускоряет средний чек примерно на 12%, но замедляет малый примерно на 47%. 10/10 медленнее базы примерно в 5 раз на малом, в 4,3 раза на среднем и в 1,8 раза на длинном чеке. Универсального выигрыша от увеличения лимитов нет.</p><p><span class="pill">Дальнейшие тесты производительности приостановлены по просьбе пользователя.</span> Этот отчет построен только из уже завершенных прогонов.</p></div>
    <div class="card"><h2>Как измеряли</h2><p>8 разных конфигураций плюс повтор базы 2/2 в конце: 9 отдельных процессов. В каждом — прогрев всех трех чеков и по 3 измеряемых повтора каждого чека. Всего 81 измеряемый прогон и 27 прогревочных. Порядок чеков в повторах менялся. Все задачи выполнялись последовательно.</p>
    <p>Источник времени — штатный <code>result.performance</code>: wall_ms, process_cpu_ms, samples и cpu_history. Мониторинг ЦП с шагом около 0,2 с не отключался. Все исходные задания и performance_profile.json сохранены приложением. Холодные прогоны не участвуют в медианах.</p>
    <p>Параметры из .env переопределялись переменными окружения только дочерних процессов. Рабочий .env не изменялся. Реальные intra-op потоки ONNX и число потоков OpenCV проверены через API библиотек. Inter-op ONNX остается равным 1. Это число потоков библиотек, а не общий процент ограничения CPU для приложения.</p>
    <p>ОС сообщает {machine['cpu_logical']} логических и {machine['cpu_physical_reported']} физических CPU; память {machine['memory_gb']:.1f} ГБ. {esc(machine['platform'])}. CPU-секунды относятся ко всему процессу и включают его потоки; среднее число занятых ядер взято из штатной телеметрии.</p>
    <p>Повтор базы в конце позволяет оценить дрейф нагрузки. Разницы порядка нескольких процентов недостаточно для уверенного выбора новой настройки. Это замер трех фотографий, а не оценка всех типов чеков или общего процента ошибок.</p></div>
    <label>Показать чек: <select id="source-filter"><option value="all">Все три</option>{''.join(f'<option value="{k}">{v}</option>' for k,v in NAMES.items())}</select></label>
    {''.join(parts)}
    <section><h2>Качество: обнаруженные нестабильные результаты</h2><p>Сравнивались все распознанные количества, цены, напечатанные суммы строк, типы строк, итог, расчетная сумма и статус НДС с базой 2/2. Два измеряемых прогона отличаются. В обоих сработал временной лимит подготовки numeric ROI; для части ячеек подготовка была пропущена. Данные указывают на зависимость результата от бюджета подготовки, а не доказывают прямое влияние числа потоков на математический результат нейросети.</p>
    <div class="scroll"><table><thead><tr><th>Потоки</th><th>Чек</th><th>Повтор</th><th>Изменения против базы</th><th>Сумма строк</th><th>ROI пропущено</th><th>Задание</th></tr></thead><tbody>{''.join(diffs)}</tbody></table></div>
    <p class="caution">У длинного чека уже в базе есть ошибки: 539288 вместо 538936, НДС 74338 и несовпадение ставки. Ни одна конфигурация не сделала этот чек корректным. Стабильность 3/3 означает сохранение базового результата, а не отсутствие ошибок.</p></section>
    <section><h2>Что следует из эксперимента</h2><p>Распараллеливание оплачивается дополнительной работой потоков. Малые задания особенно чувствительны к этим расходам. Геометрия, подготовка, OCR и сохранение масштабируются по-разному; один лимит не ускоряет весь процесс одинаково. В отчет включены отдельные этапы, чтобы отделить выигрыш нейросетевого участка от общего времени.</p><p>Настройки 6/6, 8/8 и 10/10 не дают общего выигрыша на выбранных чеках. Перед принятием 4/2 потребовалась бы рабочая смесь чеков и проверка стабильности; сейчас менять .env оснований нет. Отдельная задача качества — исключить зависимость необходимых проверок цифр от исчерпания необязательного бюджета подготовки.</p></section>
    <details class="top"><summary>Фактические настройки библиотек</summary><div class="scroll"><table><thead><tr><th>Профиль</th><th>ONNX intra-op</th><th>OpenCV</th><th>Загруженные сессии</th></tr></thead><tbody>{''.join(runtime_rows)}</tbody></table></div></details>
    <details class="top"><summary>Все 81 измеряемый прогон</summary><div class="scroll"><table><thead><tr><th>Потоки</th><th>Чек</th><th>Повтор</th><th>Время, с</th><th>Печатный итог</th><th>Сумма строк</th><th>НДС</th><th>ROI, мс</th><th>Задание</th></tr></thead><tbody>{''.join(runs)}</tbody></table></div></details>
    <p class="muted">Воспроизведение: scripts/benchmark_threads.py (новые прогоны сейчас приостановлены). Сборка отчета без тестирования: scripts/report_threads.py. Исходные JSON, журналы, SHA256 фотографий и manifest.json: data/output/cpu_threads_experiment/. Полные данные также встроены в этот HTML.</p>
    <script id="experiment-data" type="application/json">{json.dumps(profiles,ensure_ascii=False).replace('<',chr(92)+'u003c')}</script>
    <script>document.getElementById('source-filter').onchange=e=>document.querySelectorAll('.source').forEach(s=>s.classList.toggle('hidden',e.target.value!=='all'&&s.dataset.source!==e.target.value));</script>
    </main></body></html>'''
    for output in (DATA/'report.html', ROOT/'docs/CPU_THREADS_EXPERIMENT.html'):
        output.write_text(report, encoding='utf-8')
        print(output)


if __name__ == '__main__':
    main()
