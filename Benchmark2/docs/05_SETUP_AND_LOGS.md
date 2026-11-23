# 05. Установка, запуск и логи

## 5.1 Состав установщика

| Файл | Назначение |
|---|---|
| `setup_benchmark2.bat` | Двойной клик: запускает установку и пишет лог |
| `INSTALL_AND_RUN.bat` | Устанавливает окружение при первом запуске, затем запускает бенчмарк |
| `scripts\setup.ps1` | Фактическая установка: venv, snapshot пакетов, доп. OCR-пакеты, проверки |
| `scripts\setup_logged.ps1` | Обёртка: запускает `setup.ps1` и сохраняет **весь** вывод в `logs\` |
| `run_benchmark2.bat` | Проверяет установку, выставляет порт 18018 и запускает прогон |
| `requirements-extra.txt` | Пакеты, которых нет в родительском venv: RapidOCR, tesserocr, WinRT и др. |

## 5.2 Что делает `setup.ps1`

1. Определяет корень Benchmark2 по расположению скрипта.
2. Определяет родительский проект:
   - параметр `-BaseProject`;
   - переменная `BENCHMARK2_BASE_PROJECT`;
   - значение по умолчанию `C:\Work\Projects\Prj_18_OCR_2`.
3. Проверяет наличие родительского `.venv\Scripts\python.exe` и версию Python 3.14.
4. Проверяет маркер `.venv\.benchmark2-root.txt`: нельзя переносить уже созданный venv
   в другую папку без пересоздания.
5. Создаёт **новый** `Benchmark2\.venv` через `python -m venv` (не копирует бинарный venv).
6. Удаляет старый маркер готовности на время обновления.
7. Выполняет `pip freeze` родительского venv (только чтение) и сохраняет snapshot
   в `config\base_requirements_snapshot.txt`.
8. Устанавливает snapshot в `Benchmark2\.venv`.
9. Устанавливает `requirements-extra.txt` (RapidOCR, onnxruntime, tesserocr, WinRT,
   huggingface-hub, psutil, OpenCV и т. д.).
10. Выполняет `pip check` и `python -m benchmark2 validate` из корня Benchmark2.
11. Записывает маркер `.venv\.benchmark2-root.txt` только после успешной проверки.
12. Печатает `[READY]` и путь к venv.

Родительский `.venv` не изменяется: из него только читаются список пакетов и файлы моделей.

### Исправленная ошибка PowerShell 5.1

Раньше проверка версии использовала `python -c` со встроенными двойными кавычками
f-строки. Windows PowerShell 5.1 искажал аргумент, и установка падала с `SyntaxError`.
Теперь используется безопасный вариант:

```powershell
$Version = & $SourcePy -c 'import sys; print(sys.version_info.major, sys.version_info.minor, sep=chr(46))'
```

## 5.3 Как работает логирование `setup_logged.ps1`

- Каталог: `Benchmark2\logs\`.
- Имя файла: `setup_YYYYMMDD_HHMMSS.log`, например
  `setup_20261123_153009.log`.
- Wrapper запускает `setup.ps1` **дочерним процессом** `powershell.exe` и объединяет
  его stdout/stderr через `*>&1 | Tee-Object -FilePath <лог>`.
- В конец лога добавляется строка `[BENCHMARK2] Setup exit code: N`.
- Wrapper возвращает код установки в `.bat`, поэтому ошибка не теряется.

### Почему не `Start-Transcript`

В Windows PowerShell 5.1 `Start-Transcript` записывает только сообщения PowerShell
(`Write-Host` и т. п.), но **не записывает нативный stdout `pip`**. Первая попытка
дала лог, где были шаги установки, но не было ни одной строки `pip install`.
Кроме того, `Tee-Object` в тот же файл конфликтовал с блокировкой транскрипта.
Поэтому транскрипт удалён из `setup.ps1`.

### Почему `setup.ps1` запускается дочерним процессом

Если вызвать `setup.ps1` в том же процессе и объединить потоки через `*>&1`,
stderr от `pip` превращается в `NativeCommandError`. При
`$ErrorActionPreference = 'Stop'` внутри `setup.ps1` это обрывает установку
на `pip install -r snapshot`. Дочерний процесс решает проблему: установщик пишет
в свои stdout/stderr, а объединение и запись выполняет только wrapper.

### Формат лога

- Файл создаётся `Tee-Object` в кодировке **UTF-16LE** (поведение Windows PowerShell 5.1).
  `Get-Content`, Notepad и редакторы читают его нормально; не следует парсить его
  как UTF-8 вслепую.
- В логе есть строки `pip`, предупреждения, `pip check`, `validate`, `[READY]`.
- Успешный лог заканчивается:

```text
[READY] Separate venv installed at C:\...\Benchmark2\.venv
[NEXT] Double-click run_benchmark2.bat. Only Benchmark2 outputs are written.
[BENCHMARK2] Setup exit code: 0
```

### Текущие логи

| Файл | Размер | Что это |
|---|---:|---|
| `logs\setup_20261123_151659_initial.log` | 180 728 | Первая полная установка, снята внешним перенаправлением вывода |
| `logs\setup_20261123_153009.log` | 144 772 | Установка через wrapper, полный `pip`-вывод, exit code 0 |
| `logs\.gitkeep` | 0 | Сохраняет каталог в git; сами логи игнорируются |

`.gitignore` содержит:

```text
logs/*
!logs/.gitkeep
```

## 5.4 Установка

```powershell
cd C:\Work\Projects\Prj_18_OCR_2\Benchmark2
.\setup_benchmark2.bat
```

или:

```powershell
.\INSTALL_AND_RUN.bat
```

Для другого родительского проекта:

```powershell
$env:BENCHMARK2_BASE_PROJECT = 'D:\Work\Prj_18_OCR_2'
.\setup_benchmark2.bat
```

Либо напрямую:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\setup_logged.ps1 `
  -BaseProject 'D:\Work\Prj_18_OCR_2'
```

`-SkipCloneRequirements` пропускает установку snapshot родителя (используется только
для диагностики; обычная установка должна выполняться без него).

## 5.5 Подготовка данных и моделей

```powershell
.\.venv\Scripts\python.exe -m benchmark2 validate
.\.venv\Scripts\python.exe -m benchmark2 prepare
.\.venv\Scripts\python.exe -m benchmark2 models
```

- `prepare` создаёт 17 ROI и проверяет SHA256 исходных JPEG.
- `models` копирует модели из родительского проекта и скачивает отсутствующие
  (v6 Medium — с Hugging Face).
- Офлайн-режим: `--offline` (ничего не скачивать).

## 5.6 Запуск бенчмарка и дэшборд

```powershell
.\run_benchmark2.bat
```

Эквивалент:

```powershell
.\.venv\Scripts\python.exe -m benchmark2 run
```

Дэшборд:

- адрес по умолчанию: **`http://127.0.0.1:18018/`**;
- порт задаётся переменной `BENCHMARK2_PORT`;
- `run_benchmark2.bat` явно выставляет `BENCHMARK2_PORT=18018`, если переменная не задана;
- если порт занят, запуск останавливается с понятной ошибкой
  `Dashboard port 127.0.0.1:18018 is busy...`, без случайного выбора другого порта;
- сервер слушает только `127.0.0.1` (не доступен из локальной сети);
- для закрытия сервера после отчёта нажать Enter в консоли.

## 5.7 Диагностика

| Симптом | Причина / действие |
|---|---|
| `Main project Python missing` | Нет родительского `.venv`; проверить `BENCHMARK2_BASE_PROJECT` |
| `Expected Python 3.14...` | Родительский venv другой версии; бенчмарк рассчитан на 3.14 |
| `Benchmark2 was moved after installation` | Удалить **только** `Benchmark2\.venv` и запустить установку заново |
| `Dashboard port ... is busy` | Освободить 18018 или задать `BENCHMARK2_PORT` |
| Установка упала | Открыть последний `logs\setup_*.log`, найти `Setup exit code` и ошибку выше |
| Windows OCR не видит русский | Установить языковой пакет OCR `ru-RU` в Windows |
| Tesseract не находит `rus` | Проверить `models\tessdata\rus.traineddata` |
| Нет интернета | Модели/пакеты не скачаются; повторить установку с сетью |

## 5.8 Границы изменений

Все создаваемые файлы находятся внутри `Benchmark2`:

- `.venv\` — окружение;
- `models\` — внешние ONNX и tessdata;
- `.venv\Lib\site-packages\rapidocr\models\` — кэш RapidOCR;
- `data\cells\` — вырезки;
- `results\` — результаты;
- `logs\` — журналы установки.

В родительском проекте не создаётся и не изменяется ничего.
