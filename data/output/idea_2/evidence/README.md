# Доказательная база к `../idea.md`

Скрипты диагностические, в production-код не входят и данные проекта не меняют.
Запускать из корня проекта:

```powershell
.venv\Scripts\python.exe data\output\idea_2\evidence\diag_slow_ops.py
.venv\Scripts\python.exe data\output\idea_2\evidence\diag_geometry_stages.py
.venv\Scripts\python.exe data\output\idea_2\evidence\diag_footer_stages.py
.venv\Scripts\python.exe data\output\idea_2\evidence\diag_footer_alt.py
```

По умолчанию используется эталонное задание
`data/output/0251327d5d384287926c40adfb259768` (фото 899×1599). Можно передать
путь к другому заданию первым аргументом:

```powershell
.venv\Scripts\python.exe data\output\idea_2\evidence\diag_geometry_stages.py data\output\<job_id>
```

## Что показывает какой скрипт

| Скрипт | Что измеряет | Ключевой вывод |
|---|---|---|
| `diag_slow_ops.py` | AutoGrid, V2, V3, `build_grid`, footer (`read_line`, `read_full`, ×2) | геометрия ≈1,16 с, footer ≈1,14 с; основной вклад — V3 и `read_full` |
| `diag_geometry_stages.py` | внутренние стадии V3 | V2 ≈307 мс, multiscale LSD ≈326 мс, fit ≈237 мс |
| `diag_footer_stages.py` | детектор vs распознаватель | детектор ≈1,03 с, распознавание ≈37 мс; вход 736×3648 |
| `diag_footer_alt.py` | альтернативный `Det.limit_type='max'` | 28 мс вместо ~1 с; итог читается и парсится |

Абсолютные значения зависят от прогрева моделей, загрузки CPU и версии
библиотек; важны порядки величин и сравнение «до/после» на одном и том же
фото.
