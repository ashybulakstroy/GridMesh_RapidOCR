# GridMesh_RapidOCR — DEBUG Performance V2

**Целевой репозиторий:** `ashybulakstroy/GridMesh_RapidOCR`  
**Исходный коммит `main`:** `1acd37d3cab4df1775f4e43db7d2fd81a33e07dc`, 22.11.2026, 12:25 UTC.  
**Тип пакета:** обратимый экспериментальный *патч* для текущего Python-проекта; не полная замена проекта.

## Что нового по сравнению с DEBUG V1

С 22.11.2026 основной OCR изменён: `OCR_TABLE_MODE=cell`, `read_parallel()` запускает таблицу и итог двумя независимыми линиями (footer/table), `read_cells()` обходит столбцы, `read_batches()` стал резервным путём. Добавлены OriginalROI E4, дополнительные v6-модели, штатный `rec_trace`, а в геометрии — `recover_fold` и `recover_left_edges`. **DEBUG V1 рассчитан на старый пакетный конвейер и здесь не устанавливается.**

V2 интегрируется с этими механизмами, не меняя алгоритмов OCR, пакетные размеры, потоки, модели, обработку НДС, вычисления и порядок проверки данных.

## Куда сохраняется статистика

После успешного завершения каждого чека:

- `data/output/<job_id>/performance_profile.json` — штатные `wall_ms`, `process_cpu_ms`, `samples`, `cpu_history`, `budget_events` + `issues` при **любом** `LOG_LEVEL`; при `DEBUG` дополнительно `debug`.
- `data/output/<job_id>/receipt.json` — штатный результат с существующим `performance` и `rec_trace` (поведение проекта не изменяется).
- `data/output/<job_id>/numeric_experiment.json` — существующая E4/E2-аналитика; эксперимент не дублирует сюда новый TRACE.
- `processing_log.json` и `error.json` — как в текущем проекте.

При перехваченном исключении `performance_profile.json` и `error.json` записываются по прежней логике, но с диагностическим контекстом; без промежуточных записей каждые 500 мс.

### INFO

Сохраняет стандартные измерения и краткие `issues[]`: непустые Python-исключения и некорректные результаты в RapidOCR/ONNX, нештатные пустые OCR-строки, ошибки model-ensemble, table/footer lane и необработанные ошибки конвейера. Штатное многострочное чтение и обычный fallback не объявляются ошибкой.

### DEBUG

В `performance_profile.json → debug`:

| Поле | Содержимое |
|---|---|
| `configuration` | Применённые значения ONNX/OpenCV/batch, `OCR_TABLE_MODE`, включение E2/E4, названия запрошенных v6-моделей, версии Python/RapidOCR/OpenCV |
| `ocr_cells[]` | Каждая первичная ячейка в режиме `cell`: ID, поле, размеры, длительность `read_cell`, стратегия, состояние и RAM после вызова |
| `ocr_batches[]` | Каждый реально запущенный пакет в `batch` или fallback: группа, ID ячеек, размеры, длительность вызова, оценка padding |
| `ocr_lanes[]` | Время table/footer из нового параллельного механизма; на ошибке длительность может быть `null` |
| `model_calls_from_rec_trace` | Агрегация **существующего** `rec_trace` по модели v5/v6 и режиму (`REC_only`, `DET+REC`): вызовы, затраты, ошибки, экспозиции ячеек |
| `geometry_substeps[]` | AutoGrid, Physical Graph, horizontal recovery, fold recovery, left-edge recovery, Grid Mesh, safety/overlay |
| `cell_crop_summary` | Количество ячеек, длительность/размеры по колонкам |
| `ocr_fallbacks[]` | Продолжительность многострочных и аварийных возвратов на поячейковое чтение |
| `memory_mb` | RSS в начале, конце и максимум **только по контрольным точкам** |
| `issues[]` | Подробные диагностические события: источник, тип, длительность при наличии, traceback Python-исключения |

**Границы интерпретации:** `read_cell` и `recognize_txt()` включают подготовку, ONNX inference и декодирование. Их время **не равно чистому времени ONNX kernel**. Записи table/footer могут перекрываться: суммы `ocr_lanes` и `model_calls_from_rec_trace` **нельзя** использовать как wall-time всего чека. Подсчёт внутренних DET+REC-вызовов детектора остаётся непрозрачным. Встроенные предупреждения C/C++ ONNX/OpenCV, не поднятые до Python, не перехватываются.

## Установка на Windows

1. **Закрыть OCR-сервис** (`start.bat`). Распаковать ZIP. Папка в архиве одна.
2. Изменить `PROJECT=` в `install_debug_v2.bat`, если рабочая папка отличается от `C:\Work\Projects\Prj_18_OCR_2`.
3. Запустить `install_debug_v2.bat`. Он сначала проверяет **все исходные Git-blobs и якоря**, а уже затем устанавливает патч.
4. В файле проекта `.env` выставить `LOG_LEVEL=DEBUG` и запустить сервер.
5. Обработать чек и открыть `data/output/<job_id>/performance_profile.json`.

Те же команды вручную из распакованного каталога:

```bat
"C:\Work\Projects\Prj_18_OCR_2\.venv\Scripts\python.exe" apply_experiment.py --project "C:\Work\Projects\Prj_18_OCR_2" --dry-run
"C:\Work\Projects\Prj_18_OCR_2\.venv\Scripts\python.exe" apply_experiment.py --project "C:\Work\Projects\Prj_18_OCR_2"
```

Скрипт **не изменяет** `.env`, `.env.example` и не включает DEBUG автоматически. Выбор `LOG_LEVEL` полностью остаётся за вами.

Текущий `.env.example` в GitHub устанавливает `OCR_ONNX_THREADS=6` и `OCR_REC_BATCH_NUM=8` — их эксперимент не меняет; фактическая конфигурация вашего компьютера берётся из работающего процесса/`.env`.

## Сохранение и откат

Изменяются **шесть** файлов: `App/pipeline.py`, `App/ocr/p1_batch.py`, `App/geometry/service.py`, `App/ocr/ocr_performance.py`, `App/ocr/cell_mode.py`, `App/ocr/experiment_models.py`. Добавляется `App/ocr/debug_performance.py`.

Перед записью создаются резервные копии в `data/experiments/gridmesh_debug_performance_v2/backup/` и манифест SHA256. При ошибке установки восстанавливаются исходные файлы; при несовпадении текущего GitHub-кода установка **откажет до любых изменений**. Скрипт не поддерживает насильную установку на неизвестный код — это намеренно.

Для отката: закрыть сервер и запустить `restore_debug_v2.bat`. Либо вручную:

```bat
"C:\Work\Projects\Prj_18_OCR_2\.venv\Scripts\python.exe" restore_experiment.py --project "C:\Work\Projects\Prj_18_OCR_2" --dry-run
"C:\Work\Projects\Prj_18_OCR_2\.venv\Scripts\python.exe" restore_experiment.py --project "C:\Work\Projects\Prj_18_OCR_2"
```

Откат не перезапишет изменения, сделанные после установки. Параметр `--force` в restore использовать только при понимании последствий.

## Проверки и ограничения

```bat
"C:\Work\Projects\Prj_18_OCR_2\.venv\Scripts\python.exe" -m pytest tests\test_debug_v2.py -q
```

Проверена синтаксическая корректность всех шести патчеров на структурных тестовых фрагментах, интерфейс DEBUG-коллектора, сквозная передача `ContextVar` в дочерний поток, запись кратких ошибок в INFO, проверка Git-хеша, установщик/откат и защита от перезаписи последующих изменений. Отдельно контрольные якоря сопоставлены с реальными файлами GitHub.

**Непроверенное:** полный OCR на вашем Windows с моделями RapidOCR/ONNX (в среде сборки модели не установлены), собственная стоимость DEBUG в миллисекундах и фактическая точность OCR после установки. Перед принятием измерить `INFO` vs `DEBUG` на одинаковых чеках после прогрева. Не считать `DEBUG`-тайминги чистым production benchmark — в DEBUG есть дополнительные вызовы `psutil.memory_info()`.


## Приёмка в основном проекте — 22.11.2026

Установлено на точный исходный коммит 1acd37d. Локальный `.env` уже содержал LOG_LEVEL=DEBUG, настройка сохранена. Диагностические таймеры p1_batch переведены на независимый rec_counter, чтобы не влиять на тестовые часы ROI-бюджетов. Автономные тесты коллектора используют отдельное имя пакета вместо подмены App в sys.modules. Резервный манифест отражает установленные интеграционные изменения.

На реальном чеке 252161 проведены пять прогретых запусков до установки и по пять INFO/DEBUG после установки с чередованием режимов. Все названия и выбранные числовые поля совпали. Медиана полного цикла: до 2,149 с; INFO 2,149 с; DEBUG 2,138 с. Такое различие не доказывает ускорения DEBUG. В DEBUG сохранены 51 ячейка, две линии OCR и пять геометрических подэтапов, ошибки отсутствуют. Локальные данные и HTML: data/experiment_reports/gridmesh_debug_performance_v2/.
