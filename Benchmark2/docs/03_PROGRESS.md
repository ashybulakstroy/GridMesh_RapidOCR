# 03. Отчётность и прогресс

Дата журнала: **2026-11-23**. Все времена — локальные, по логам ПК `ERZHAN-OFFICE-2`.

## 3.1 Сводка состояния

| Проверка | Результат | Свидетельство |
|---|---|---|
| Отдельный venv | Готов | `Benchmark2\.venv`, Python 3.14.2 |
| Пакеты | 135 установлено | `pip list --format=freeze` |
| Конфликты зависимостей | Нет | `pip check` → `No broken requirements found.` |
| Валидация протокола | Успешно | `CHECK: 17 original ROIs, 6 Cyrillic engines, 3 passes incl. warmup` |
| Unit-тесты | 6 passed | `python -m pytest tests -q` — `6 passed in 0.87s` |
| Размер venv | 554,8 MB | замер каталога `.venv` |
| ROI | 17 PNG | `data\cells\`, `data\cells_manifest.json` |
| Исходные JPEG | 3, SHA256 совпали | `data\samples.json`, `cells_manifest.json` |
| Внешние модели | READY | `data\models_manifest.json` |
| Smoke-тесты 6 движков | 5 с текстом, Windows OCR частично пуст | см. раздел 3.4 |
| Setup-лог | Полный, exit 0 | `logs\setup_20261123_153009.log` |
| Полный прогон 306 операций | **Не запускался** | `results\` содержит только `.gitkeep` |

## 3.2 Хронология работ

### 2026-11-23, днём — инвентаризация

- Проверен родительский проект `C:\Work\Projects\Prj_18_OCR_2`.
- Родительский `.venv`: Python 3.14.2, 43 пакета, среди них `rapidocr 3.10.0`,
  `onnxruntime 1.32.0`, `numpy 2.5.5`, `Pillow 13.0.0`, `opencv-python-headless 5.1.0.95`.
- Найдены и классифицированы существующие AI-модели (полный список — в [04_MODELS.md](04_MODELS.md)).
- Установлено, что для бенчмарка пригодны:
  - `models\ppocrv6_tiny_cyrillic\inference.onnx`;
  - `cyrillic_PP-OCRv5_rec_mobile.onnx` из RapidOCR-кэша родительского venv.
- Английские модели (`en_PP-OCRv5`, `PP-OCRv6_rec_tiny/small`) и числовая CNOCR-модель
  в кириллический бенчмарк не включены.

### ~15:16 — первая полная установка

- Создан отдельный `Benchmark2\.venv` (не копия бинарного venv, а пересоздание
  по `pip freeze` родителя + `requirements-extra.txt`).
- Полный консольный вывод установки сохранён как
  `logs\setup_20261123_151659_initial.log` — **180 728 байт**.
- Установка завершилась `[READY]`, `pip check` без ошибок, `benchmark2 validate` успешно.

### ~15:17 — проверки окружения

- `python -m pytest tests -q` → `6 passed in 0.87s`.
- `python -m benchmark2 validate` → `CHECK: 17 original ROIs, 6 Cyrillic engines, 3 passes incl. warmup`.
- Версии: RapidOCR 3.10.0, onnxruntime 1.32.0, tesserocr 2.13.0 (Tesseract 5.5.4),
  WinRT Media.Ocr 4!10.0.28000.2957, winrt-runtime 4.2.0, huggingface-hub 2.4.0,
  psutil 7.2.2, OpenCV 5.1.0.95, NumPy 2.5.5, Pillow 13.0.0.

### ~15:18 — подготовка 17 ROI и данные

- Выполнен `python -m benchmark2 prepare`.
- Проверены SHA256 трёх исходных JPEG.
- Созданы 17 PNG в `data\cells\` без масштабирования.
- Скачаны `tessdata_fast`:
  - `rus.traineddata` — 3 861 738 байт, SHA256 `e16e5e036cce1d9ec2b00063cf8b54472625b9e14d893a169e2b0dedeb4df225`;
  - `eng.traineddata` — 4 113 088 байт, SHA256 `7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2`.
- Создан `data\cells_manifest.json` (17 записей, размеры, bbox, SHA256 вырезок).

### ~15:18–15:19 — smoke-тесты моделей

- v5 Cyrillic, v6 Tiny, Tesseract загрузились и вернули текст на первой строке `R01_name_01`.
- v3 Cyrillic отсутствовал локально; RapidOCR скачал его с ModelScope:
  `cyrillic_PP-OCRv3_rec_mobile.onnx`, 8 972 413 байт (8,56 MiB).
- v3 загрузился и вернул текст.
- v6 Medium отсутствовал; запущена загрузка с Hugging Face
  `metrika-labs/PP-OCRv6_medium_rec_cyrillic_onnx`.

### ~15:21 — v6 Medium и manifest

- v6 Medium скачан: `inference.onnx`, 62 764 959 байт,
  SHA256 `997e543cc73c2f99935fce99a79a74a5165602c816e15cf2b3e923e5a3420ac0`.
- `python -m benchmark2 models` записал `data\models_manifest.json`:
  v6 Tiny и v6 Medium — `READY`.
- v6 Medium успешно распознал smoke-ячейку.

### ~15:23 — неудачная попытка логов через `Start-Transcript`

- В `setup.ps1` был добавлен `Start-Transcript`.
- Проверка показала: Windows PowerShell 5.1 пишет в транскрипт только сообщения
  PowerShell (`Write-Host`), но **не пишет нативный вывод `pip`**.
- Дополнительно `Tee-Object` в тот же файл конфликтовал с блокировкой транскрипта.
- Вывод: транскрипт непригоден; код `Start-Transcript` удалён.
- Неполный файл лога удалён, чтобы не путать с рабочим логом.

### ~15:28 — неудачная попытка wrapper «в одном процессе»

- Создан `scripts\setup_logged.ps1`, который вызывал `setup.ps1` в текущем процессе
  с `*>&1 | Tee-Object`.
- Причина сбоя: `setup.ps1` работает с `$ErrorActionPreference = 'Stop'`.
  Перенаправление `*>&1` превращает stderr от `pip` в `NativeCommandError`,
  и установка обрывается на строке `pip install -r snapshot`.
- Получен частичный лог `setup_20261123_152826.log`; после успешного исправления удалён.

### ~15:30 — рабочая схема логов

- Wrapper переработан: `setup.ps1` запускается **дочерним процессом** `powershell.exe`,
  а объединение stdout/stderr и запись в лог выполняет только wrapper
  (`*>&1 | Tee-Object`). Так stderr `pip` не попадает в EAP `Stop` установщика.
- Повторный запуск завершён успешно:
  - лог: `logs\setup_20261123_153009.log`;
  - размер: **144 772 байта**;
  - последняя строка: `[BENCHMARK2] Setup exit code: 0`;
  - в логе присутствуют `pip`-строки, `No broken requirements found.`,
    `CHECK: 17 original ROIs...`, `[READY]`.
- Логи добавлены в `.gitignore` (`logs/*`, кроме `.gitkeep`).
- `setup_benchmark2.bat` и `INSTALL_AND_RUN.bat` переведены на `setup_logged.ps1`.

### 2026-11-23, после 15:30 — документация

- Создан каталог `docs\` с данным комплектом:
  задача, план, прогресс, модели, установка/логи, решения/риски.
- Обновлены `README.md` и `START_HERE.txt` — указано расположение setup-логов.

## 3.3 Технические smoke-тесты: фактические наблюдения

Это **не результаты бенчмарка**, а проверка работоспособности адаптеров на одной
тестовой вырезке (первая строка `R01_name_01`, 317×28 px), если не указано иное.

| Движок | Загрузка | Текст на smoke-ячейке | Confidence |
|---|---|---|---|
| `v3_cyrillic_mobile` | OK | есть, 18 символов | 0.82486 |
| `v5_cyrillic_mobile` | OK | есть | 0.98481 |
| `v6_tiny_cyrillic_optional` | OK | есть | 0.98973 |
| `v6_medium_cyrillic` | OK | есть, 19 символов | 0.99983 |
| `tesseract_fast` | OK | есть | 0.87 |
| `windows_ocr` | OK | **пусто** на строке 317×28 | — |

Windows OCR на других вырезках:

- `R01_name_01` целиком (317×172) — 97 символов, текст есть;
- `R01_vat_label` (317×33) — пусто;
- `R02_vat_label` (336×32) — пусто;
- `R01_total` (270×47) — 11 символов, текст есть.

Проверка гипотезы о минимальной высоте:

| Вырезка | Оригинал | +10 px белой рамки | ×2 | ×2 + 10 px |
|---|---|---|---|---|
| `R01_vat_label` 317×33 | пусто | 11 символов, «НДС 16 %» с шумом | 4 символа, `16 %` | 11 символов, шум |
| `R02_vat_label` 336×32 | пусто | 8 символов, `НДС 16 %` | 10 символов, `ндс 16 0/0` | 10 символов, `ндс 16 0/0` |
| `R01_total` 270×47 | 11 символов, `55 564,00 т` | 10 символов, шум | пусто | 12 символов, `55 564,00 т!` |

Вывод: проблема существует, но padding/upscale не даёт стабильного улучшения на всех
ячейках. Политика пока не выбрана; молчаливое изменение входов запрещено.
См. [02_PLAN.md](02_PLAN.md#23-этап-8--политика-windows-ocr) и
[06_DECISIONS_AND_RISKS.md](06_DECISIONS_AND_RISKS.md).

## 3.4 Где лежат свидетельства

| Артефакт | Путь |
|---|---|
| Полный первичный setup-лог | `logs\setup_20261123_151659_initial.log` (180 728 байт) |
| Полный setup-лог через wrapper | `logs\setup_20261123_153009.log` (144 772 байта, exit 0) |
| Конфигурация 6 моделей | `config\models.json` |
| Snapshot пакетов родителя | `config\base_requirements_snapshot.txt` |
| Манифест ROI | `data\cells_manifest.json` |
| Манифест моделей | `data\models_manifest.json` |
| Исходные эталоны/координаты | `data\samples.json` |
| Тесты | `tests\test_protocol.py` |
| Маркер пути установки | `.venv\.benchmark2-root.txt` |

## 3.5 Что ещё не сделано

- Не выбран вариант политики Windows OCR для узких строк.
- Не запущен полный прогон 306 операций.
- Нет `results\<run_id>\raw.jsonl`, `summary.json`, `report.html` с реальными замерами.
- Эталонные тексты не утверждены владельцем.
- Каталог `docs\` ещё не добавлен в git; остальные файлы Benchmark2 уже отслеживаются
  (коммит `3d9fa3e` на момент проверки).

## 3.6 Текущий статус

**Подготовка завершена, бенчмарк не запущен.**
Следующее действие — решение по Windows OCR и команда полного прогона
`run_benchmark2.bat` на `http://127.0.0.1:18018/`.
