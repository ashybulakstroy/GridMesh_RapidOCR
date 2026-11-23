# 04. Модели и инвентаризация

Дата инвентаризации: **2026-11-23**.
Все размеры в байтах, SHA256 — в нижнем регистре.

## 4.1 Участники бенчмарка

| ID | Движок | Источник весов | Размер | SHA256 | Статус |
|---|---|---|---:|---|---|
| `v3_cyrillic_mobile` | PP-OCRv3 Mobile Cyrillic (маршрут RapidOCR PP-OCRv4) | RapidOCR 3.10.0, ModelScope `RapidAI/RapidOCR` `v3.10.0/onnx/PP-OCRv4/rec/cyrillic_PP-OCRv3_rec_mobile.onnx` | 8 972 413 | `1efb65bdc460af1c0e8733d005b20952b17ca5aac10ddb56c968333791c5eaa3` | загружен, smoke OK |
| `v5_cyrillic_mobile` | PP-OCRv5 Mobile Cyrillic | скопирован из RapidOCR-кэша родительского `.venv` | 8 074 092 | `90f761b4bfcce0c8c561c0cb5c887b0971d3ec01c32164bdf7374a35b0982711` | скопирован, smoke OK |
| `v6_tiny_cyrillic_optional` | PP-OCRv6 Tiny Cyrillic (Metrika Labs ONNX) | скопирован из `Prj_18_OCR_2\models\ppocrv6_tiny_cyrillic\inference.onnx`; резерв — HF `metrika-labs/PP-OCRv6_tiny_rec_cyrillic_onnx` | 2 499 854 | `d804d23767efd82e188f591d77589d213d102a759785436c89ae6afb88721d05` | READY, smoke OK |
| `v6_medium_cyrillic` | PP-OCRv6 Medium Cyrillic (Metrika Labs ONNX) | HF `metrika-labs/PP-OCRv6_medium_rec_cyrillic_onnx` | 62 764 959 | `997e543cc73c2f99935fce99a79a74a5165602c816e15cf2b3e923e5a3420ac0` | READY, smoke OK |
| `tesseract_fast` | Tesseract LSTM + `tessdata_fast` `rus+eng` | `tesserocr 2.13.0` (встроенный Tesseract 5.5.4), языки скачаны с `tesseract-ocr/tessdata_fast` | `rus` 3 861 738; `eng` 4 113 088 | `rus` `e16e5e036cce1d9ec2b00063cf8b54472625b9e14d893a169e2b0dedeb4df225`; `eng` `7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2` | smoke OK |
| `windows_ocr` | Windows.Media.Ocr (ru-RU) через PyWinRT | системный OCR Windows 10, языковой пакет `ru-ru` | `MsOcrRes.orp` 242 366 | системный файл, SHA256 не фиксировался | smoke с ограничением |

### Где лежат веса

| Модель | Путь |
|---|---|
| v3 | `.venv\Lib\site-packages\rapidocr\models\cyrillic_PP-OCRv3_rec_mobile.onnx` |
| v5 | `.venv\Lib\site-packages\rapidocr\models\cyrillic_PP-OCRv5_rec_mobile.onnx` |
| v6 Tiny | `models\ppocrv6_tiny_cyrillic\inference.onnx` (+ `inference.yml`) |
| v6 Medium | `models\ppocrv6_medium_cyrillic\inference.onnx` |
| Tesseract `rus` | `models\tessdata\rus.traineddata` |
| Tesseract `eng` | `models\tessdata\eng.traineddata` |
| Windows OCR ru | `C:\Windows\OCR\ru-ru\MsOcrRes.orp` |

Манифест внешних моделей: `data\models_manifest.json`.
Конфигурация участников: `config\models.json`.

## 4.2 Что уже было в родительском проекте

Инвентаризация выполнена **до** скачивания, чтобы не качать лишнее.

### Пригодно и использовано

| Объект | Размер | SHA256 | Как использовано |
|---|---:|---|---|
| `models\ppocrv6_tiny_cyrillic\inference.onnx` | 2 499 854 | `d804d23767efd82e188f591d77589d213d102a759785436c89ae6afb88721d05` | скопирован в Benchmark2 |
| `.venv\Lib\site-packages\rapidocr\models\cyrillic_PP-OCRv5_rec_mobile.onnx` | 8 074 092 | `90f761b4bfcce0c8c561c0cb5c887b0971d3ec01c32164bdf7374a35b0982711` | скопирован в Benchmark2 |

Также в `models\ppocrv6_tiny_cyrillic\` были `inference.yml` (7 873 байта), `LICENSE`, `NOTICE`.

### Не подходит для кириллического бенчмарка

| Объект | Причина |
|---|---|
| `en_PP-OCRv5_rec_mobile.onnx` (7 872 351) | английская rec-модель |
| `PP-OCRv6_rec_tiny.onnx` (4 489 813) | официальная multilingual/en-модель, кириллического маршрута нет |
| `PP-OCRv6_rec_small.onnx` (21 234 383) | то же |
| `ch_PP-OCRv5_det_mobile.onnx` (4 819 576) | детектор, в этом бенчмарке не нужен |
| `ch_ppocr_mobile_v2.0_cls_mobile.onnx` (585 532) | классификатор ориентации, не участник |
| `PP-OCRv6_det_small.onnx` (9 929 594) | детектор |
| `data\models\cnocr\2.3\number-densenet_lite_136-fc\...onnx` (2 763 784) | числовая CNOCR-модель, не кириллический OCR |
| HF-кэш `PaddlePaddle/*` | в snapshots только `refs/main` (40 байт), пригодных весов нет |
| HF-кэш `intfloat/multilingual-e5-base` | эмбеддинги, не OCR |

### Системные OCR-компоненты

- **Tesseract 5.4.0** установлен в `C:\Program Files\Tesseract-OCR\`, но доступны
  только языки `eng` и `osd`; `rus` отсутствовал. Поэтому бенчмарк использует
  собственный `tesserocr 2.13.0` с Tesseract **5.5.4** и отдельный `tessdata_fast`.
- **Windows OCR**: пакет `ru-ru` установлен, `OcrEngine.available_recognizer_languages`
  возвращает `['en-US', 'ru']`. Файл `C:\Windows\OCR\ru-ru\MsOcrRes.orp` присутствует.

## 4.3 Что скопировано, а что скачано

| Действие | Модель | Причина |
|---|---|---|
| Копирование | v6 Tiny Cyrillic | уже был в родительском проекте, SHA256 совпал |
| Копирование | v5 Cyrillic Mobile | уже был в RapidOCR-кэше родительского venv, SHA256 совпал |
| Скачивание RapidOCR | v3 Cyrillic Mobile | локально отсутствовал; RapidOCR скачал с ModelScope при первом обращении |
| Скачивание Hugging Face | v6 Medium Cyrillic | локально отсутствовал; HF-репозиторий публичный, не gated |
| Скачивание | `tessdata_fast` `rus`, `eng` | в системе не было `rus`; `eng` взят из того же набора для единого пайплайна |

Проверка HF перед загрузкой: репозиторий `metrika-labs/PP-OCRv6_medium_rec_cyrillic_onnx`
публичный (`private: false`, `gated: false`), файл `inference.onnx` доступен по HTTP 200,
размер 62 764 959 байт.

## 4.4 Параметры распознавания

- RapidOCR-модели: детектор **отключён**, на вход подаётся уже вырезанная ячейка.
- v6 Tiny/Medium: архитектурный вход `[3, 48, 640]`; используется `inference.onnx`
  Metrika Labs. Для v6 Tiny SHA256 зафиксирован в `config/models.json`.
- Tesseract: `PSM.SINGLE_LINE` для однострочных сегментов, языки `rus+eng`;
  многострочные названия разбиваются одинаковым алгоритмом для всех моделей,
  затем строки объединяются; время всех частей входит в один замер ячейки.
- Windows OCR: язык `ru`; изображение подаётся как есть, без padding.
- Никакого тюнинга потоков, батчей, словарей или входного масштаба под конкретную
  модель нет. Одинаковые правила применяются ко всем участникам.

## 4.5 Windows OCR: важное ограничение

Windows OCR чувствителен к высоте строки. Наблюдения на реальных ROI:

| Вход | Результат |
|---|---|
| `R01_vat_label`, 317×33, оригинал | пустая строка |
| `R02_vat_label`, 336×32, оригинал | пустая строка |
| `R01_name_01`, часть строки 317×28 | пустая строка |
| `R01_name_01`, весь блок 317×172 | 97 символов, текст распознан |
| `R01_total`, 270×47 | 11 символов, текст распознан |
| `R01_vat_label` + 10 px белой рамки | 11 символов, «НДС 16 %» с шумом |
| `R02_vat_label` + 10 px белой рамки | 8 символов, `НДС 16 %` |
| `R01_total` + 10 px белой рамки | 10 символов, шум (хуже оригинала) |
| `R01_total` ×2 без рамки | пусто |
| `R01_total` ×2 + 10 px | 12 символов, `55 564,00 т!` |

Вывод: padding помогает одним ячейкам и вредит другим; универсального «улучшения» нет.
Поэтому **нельзя молча добавлять рамку**. Нужен явный выбор протокола:

1. оригинальные ROI для всех — Windows OCR получает пустые ответы на узких строках;
2. padding для всех моделей — пересоздание входов и повторная проверка;
3. отдельный протокол Windows OCR с пометкой несравнимости;
4. исключение Windows OCR.

Подробнее — [02_PLAN.md](02_PLAN.md#23-этап-8--политика-windows-ocr) и
[06_DECISIONS_AND_RISKS.md](06_DECISIONS_AND_RISKS.md).

## 4.6 Политика хэшей и воспроизводимости

- SHA256 исходных JPEG проверяется до нарезки ROI.
- SHA256 каждой вырезки записывается в `data\cells_manifest.json`.
- Для v6 Tiny SHA256 задан в `config/models.json`; при несовпадении модель
  помечается `HASH_MISMATCH` и не должна участвовать.
- v5 проверен по хэшу родительского файла перед копированием.
- v3 скачан RapidOCR; его фактический SHA256 зафиксирован в этом документе.
  В `config/models.json` ожидаемый хэш для v3 пока не задан — при необходимости
  его можно добавить для ещё более строгой проверки.
- v6 Medium проверен по факту скачивания; `data\models_manifest.json` содержит
  статус `READY`, размер и SHA256.
- Системные файлы Windows OCR не хэшируются: они обновляются вместе с ОС.
  Версия/язык фиксируются в `environment.json` во время прогона.
