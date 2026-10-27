# Набор 3D-FRONT: форматы методов и проверка

Генератор [front3d-pano](https://github.com/HoloPano-dissertation/front3d-pano) выдаёт
эталонную разметку и COCO. Всё, что нужно только трём методам комплекса (DPC, Im3D-Pano,
Total3D-Pano), и инструменты, которыми набор проверялся перед обучением, лежат здесь.

```text
scripts/front3d/
├── adapters/   разметка генератора → форматы HorizonNet, Pano3D/DPC, LDIF, MGN, вырезки предметов
├── eval/       списки видов для первой стадии, вывода и формы
└── qa/         проверка камер, геометрии, изображений и разметки; отчёт HTML и контактные листы
```

Ниже `$EXP` — папка эксперимента генератора (после `front3d-pano export` и `validate`),
`$TBX` — этот репозиторий, `$DPC` — клон DeepPanoContext.

## Окружения

| Скрипты                                                                                                                        | Что нужно                                                                                                              |
|--------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| `export_crops.py`, `export_horizonnet.py`, `export_sdf.py`, `export_mgn_points.py`, `export_dpc.py`, `validate_dpc.py`, `qa/*` | установленный генератор (`pip install -e <клон front3d-pano>`) с его зависимостями; `export_sdf.py` ещё `scikit-image` |
| `validate_horizonnet.py`                                                                                                       | то же и корень `$DPC` в `PYTHONPATH` (читает цели загрузчиком HorizonNet)                                              |
| `export_dpc_scenes.py`, `complete_dpc_metadata.py`, `check_dpc_scenes.py`, `eval/make_test_split.py`, `eval/index_stage1.py`   | окружение `Pano3D` и корень `$DPC` в `PYTHONPATH`: сцены пишутся и читаются кодом самого DPC                           |
| `eval/make_shape500.py`                                                                                                        | только стандартная библиотека Python                                                                                   |

## `adapters/` — по порядку

```bash
python $TBX/scripts/front3d/adapters/export_crops.py $EXP
python $TBX/scripts/front3d/adapters/export_horizonnet.py $EXP
PYTHONPATH=$DPC python $TBX/scripts/front3d/adapters/validate_horizonnet.py $EXP
python $TBX/scripts/front3d/adapters/export_dpc.py $EXP
python $TBX/scripts/front3d/adapters/validate_dpc.py $EXP
python $TBX/scripts/front3d/adapters/export_sdf.py $EXP
python $TBX/scripts/front3d/adapters/export_mgn_points.py $EXP

export PYTHONPATH=$DPC
conda run -n Pano3D python $TBX/scripts/front3d/adapters/export_dpc_scenes.py $EXP $EXP/dpc_scenes
conda run -n Pano3D python $TBX/scripts/front3d/adapters/complete_dpc_metadata.py $EXP/dpc_scenes \
    --classes <классы через запятую>
conda run -n Pano3D python $TBX/scripts/front3d/adapters/check_dpc_scenes.py $EXP $EXP/dpc_scenes
```

| Скрипт                                           | Что пишет                                                              | Для чего                                                            |
|--------------------------------------------------|------------------------------------------------------------------------|---------------------------------------------------------------------|
| `export_crops.py`                                | `objects/.../crop-<дом>__<комната>__<вид>.png`, `objects/<часть>.json` | вырезки предметов для BEN                                           |
| `export_horizonnet.py`, `validate_horizonnet.py` | `horizonnet/<часть>/`                                                  | обучение HorizonNet (`training/train_horizonnet.py`)                |
| `export_dpc.py`, `validate_dpc.py`               | `dpc_dataset/`                                                         | подача изображений в DPC и соответствие имён видам                  |
| `export_sdf.py`                                  | в папке каждого предмета `objects/<класс>/<id>/`                       | цели LDIF                                                           |
| `export_mgn_points.py`                           | там же                                                                 | цели MGN, нужен результат `export_sdf.py`                           |
| `export_dpc_scenes.py`                           | `dpc_scenes/`                                                          | сцены `data.pkl` для BEN, Scene-GCN и вывода; нужны цели HorizonNet |
| `complete_dpc_metadata.py`                       | статистика размеров классов в `dpc_scenes/`                            | её требует загрузчик сцен DPC                                       |
| `check_dpc_scenes.py`                            | отчёт                                                                  | сверка сцен с эталоном генератора                                   |

`export_crops.py`, `export_sdf.py`, `export_mgn_points.py` и `export_dpc_scenes.py`
шардируются (`--shard-index`, `--shard-count`). Вспомогательные модули: `layout_targets.py`
(цели макета по контуру), `scene_layout.py` (рамки и углы в соглашениях DPC),
`sdf_samples.py` (поле расстояния).

### Форматы

**`horizonnet/<часть>/`**

```
img/<дом>__<комната>__<вид>.png    ссылка на rgb
label_dense/<...>.npz              плотные цели
label_cor/<...>.txt                углы в формате HorizonNet (только для контуров без дыр)
```

| Ключ `npz`         | Форма  | Что                                                               |
|--------------------|--------|-------------------------------------------------------------------|
| `boundary`         | 2×1024 | широта (рад) границы потолок—стена и стена—пол в каждом столбце   |
| `corner`           | 1×1024 | `0.96^(расстояние в пикселях до ближайшего видимого угла)`        |
| `ranges`           | 1024   | расстояние по горизонтали от камеры до контура вдоль луча столбца |
| `visible_corner_x` | K      | столбцы видимых углов                                             |

Цели строятся лучами из камеры до рёбер очищенного контура, поэтому подходят для комнат
любой формы, в том числе с дырами в плане.

**Вырезки предметов.** Плотная рамка маски с полем 10 %, сторона не меньше 8 пикселей.

**`dpc_dataset/`.** `images/<часть>/<дом>__<комната>__<вид>.png` (ссылки на `rgb`),
`<часть>.json`, `test_smoke.json` (одна сцена для пробного запуска) и `sample_map.jsonl`
(соответствие имён DPC идентификаторам видов).

**`dpc_scenes/`.** Сцены в формате `data.pkl`, собранные кодом самого DPC. Соглашения:
`bdb3d.basis = basisᵀ` (в DPC оси — столбцы), `centroid = center + камера`;
`manhattan_world` — 2N углов: N на полу, затем N на потолке, обход по часовой стрелке,
дыры отбрасываются. Виды без видимых предметов в сцены не попадают.

**Цели LDIF и MGN** в папке каждого предмета:

| Файл                                   | Что                                                                  |
|----------------------------------------|----------------------------------------------------------------------|
| `nss_points.sdf`, `uniform_points.sdf` | по 100 000 точек у поверхности и равномерно, со знаковым расстоянием |
| `coarse_grid.grd`                      | грубая сетка 32³ в формате GAPS                                      |
| `mesh_watertight.ply`                  | замкнутая сетка, построенная по полю расстояния                      |
| `orig_to_gaps.txt`                     | матрица 4×4 перехода в нормированную систему LDIF                    |
| `gt_3dpoints.mgn`, `densities.mgn`     | 10 000 точек поверхности и их локальная плотность                    |

Перед расчётом поля сетка умножается на 0.8, чтобы поместиться в куб ±0.7.

### Ошибки форматов, которые были найдены и исправлены

Теперь их ловят тесты (`tests/front3d/`) и `check_dpc_scenes.py`:

1. Порядок углов `manhattan_world` в сценах DPC: нужны блоки — N точек пола, затем
   N точек потолка.
2. `basis` рамки в сценах DPC транспонирован относительно эталона. Без транспонирования
   рамки смещались до 1.7 м.
3. Пиксельные углы HorizonNet сдвигались на один пиксель: план искажался на 3 %.
4. `model_path` сцены DPC должен указывать на папку предмета, а не на файл.
5. Виды без видимых предметов исключаются из сцен DPC: на них ломается подсчёт отношений.

## `qa/` — проверка набора перед обучением

```bash
python $TBX/scripts/front3d/qa/audit.py $EXP/splits/rooms.jsonl $EXP/outputs \
    $EXP/validation/geometry --workers 4
python $TBX/scripts/front3d/qa/review.py $EXP/validation/geometry \
    $EXP/validation/review --all-flagged
python $TBX/scripts/front3d/qa/extract_preview.py $EXP/outputs/<часть>/<дом>/<комната>/0.hdf5 <папка>
```

| Скрипт                      | Что делает                                                                                                                                                                                          |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `audit.py` (+ `quality.py`) | проверяет камеры, геометрию, яркость и долю валидной глубины, строит `review.html` с превью                                                                                                         |
| `review.py`                 | контактные листы: все отмеченные комнаты и фиксированная случайная выборка                                                                                                                          |
| `extract_preview.py`        | цвет, глубину, нормали и экземпляры одного HDF5 в PNG и формы массивов в `datasets.json`                                                                                                            |
| `check_annotations.py`      | сверяет разметку комнаты с отрендеренной глубиной: границы потолка и пола HorizonNet, порядок углов, долю точек предмета внутри его рамки; рисует наложение по четырём видам и пишет `report.jsonl` |

```bash
python $TBX/scripts/front3d/qa/check_annotations.py $EXP <папка отчёта> <дом>/<комната> [...]
```

Рамка отмечается пурпурным, если внутри неё меньше 95 % точек предмета (с полем 3 см);
столбцы, где граница потолка расходится с глубиной больше чем на 2 пикселя, — красными точками.

## `eval/` — списки видов

| Скрипт                       | Что пишет                                                                   | Когда                                                |
|------------------------------|-----------------------------------------------------------------------------|------------------------------------------------------|
| `index_stage1.py EXP STAGE1` | `STAGE1/train.json`, `STAGE1/test.json` (проверочная часть)                 | после первой стадии, перед обучением BEN и Scene-GCN |
| `make_test_split.py EXP`     | `dpc_scenes/test_split.json`, `eval/manifest_test.jsonl`                    | перед выводом на тестовой части                      |
| `make_shape500.py EXP`       | `eval/shape500/`: список 500 видов, эталон с `mesh_watertight.ply`, перечни | перед подсчётом формы                                |

Виды без видимых предметов во все списки не попадают: на них падает подсчёт отношений DPC.
Порядок команд — [docs/front3d.md](../../docs/front3d.md).

Флаг `missing_depth` ставится при доле валидной глубины ниже 95 %: окна и проёмы дают его
и на нормальных комнатах, поэтому отмеченную комнату нужно посмотреть, а не исключать сразу.

Ворота обучения (`state/training_gate.json`) открывает только человек после просмотра этих
отчётов, см. [training/README.md](../../training/README.md).
