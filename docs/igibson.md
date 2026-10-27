# Воспроизведение статьи на iGibson

Здесь описано, как получить заново все числа версии статьи на наборе iGibson: пять
конфигураций методов, две контрольные строки на 11 классах и сравнение формы LDIF
и MGN. Все числа считаются одним и тем же кодом бенчмарка, конфигурации различаются
только настройками вывода DeepPanoContext.

## Три уровня воспроизведения

| Уровень                                  | Что нужно                                                            | Время на 500 сцен             | Зачем                                        |
|------------------------------------------|----------------------------------------------------------------------|-------------------------------|----------------------------------------------|
| **1. Пересчёт по готовым предсказаниям** | предсказания и эталон в JSON                                         | секунды, без GPU, на любой ОС | проверить код метрик и все таблицы           |
| **2. Через окружение модели**            | собранное окружение DeepPanoContext и сохранённый вывод (`data.pkl`) | около минуты                  | пройти путь так, как задуман комплекс        |
| **3. С нуля**                            | набор iGibson, GPU                                                   | часы                          | заново получить вывод и обучить головы формы |

Уровни 1 и 2 проверены на всех 500 сценах: они дают одинаковые числа до последнего знака.

**Зачем окружение модели, если метрикам не нужен GPU.** DeepPanoContext сохраняет каждое
предсказание как `pickle` своих классов. Чтобы его прочитать, нужно импортировать
`utils.*`, а они тянут за собой `gibson2`. Ядро бенчмарка не использует даже numpy и
такой файл прочитать не может, поэтому перевод в общий формат выполняется в окружении
модели. Это единственный шаг, которому окружение нужно. Уровень 1 — тот же перевод,
сделанный один раз заранее и сохранённый в JSON.

## Что понадобится

| Что                                                        | Откуда                                                                                                                                                                              | Для какого уровня                             |
|------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| этот репозиторий                                           | `git clone`, затем `make setup`                                                                                                                                                     | все                                           |
| набор iGibson                                              | официальный сайт iGibson/BEHAVIOR (доступ по заявке, не распространяется), подготовка как в репозитории DeepPanoContext                                                             | 3                                             |
| код и веса DeepPanoContext                                 | `pano3dbench env prepare --env configs/environments/dpc.yaml`                                                                                                                       | 2, 3                                          |
| наши веса голов формы (MGN, LDIF)                          | [выпуск `weights-v1`](https://github.com/HoloPano-dissertation/pano3dbench/releases/tag/weights-v1): `total3d-pano_mgn_igibson.pth`, `im3d-pano_ldif_igibson.pth`, `SHA256SUMS.txt` | 2, 3 для Im3D-Pano (обученной) и Total3D-Pano |
| предсказания и эталон в JSON (`igibson_bench`, около 7 МБ) | получаются на уровне 2, см. ниже. В репозиторий пока не входят                                                                                                                      | 1                                             |

Ниже `$TBX` — путь к этому репозиторию, `$DPC` — к развёрнутому DeepPanoContext
(`$TBX/.pano3dbench/repos/Pano3D` после `env prepare`).

## Какая конфигурация даёт какую строку

Конфигурации лежат в `configs/igibson/runs/`. У каждого метода две: `<метод>.yaml`
проходит через окружение модели (уровень 2), `<метод>_rescore.yaml` пересчитывает готовые
предсказания (уровень 1).

| Строка статьи                          | Веса    | Через окружение     | По готовым предсказаниям     |
|----------------------------------------|---------|---------------------|------------------------------|
| DPC@100 (выпуск авторов)               | авторов | `dpc.yaml`          | `dpc_rescore.yaml`           |
| DPC@20 (стабилизированная)             | авторов | `dpc_s20.yaml`      | `dpc_s20_rescore.yaml`       |
| Im3D-Pano (абляция)                    | авторов | `im3d.yaml`         | `im3d_rescore.yaml`          |
| Im3D-Pano (обученная)                  | наши    | `im3d_trained.yaml` | `im3d_trained_rescore.yaml`  |
| Total3D-Pano                           | наши    | `total3d.yaml`      | `total3d_rescore.yaml`       |
| контрольная строка DPC@100, 11 классов | авторов | —                   | `dpc_pcf.yaml`               |
| контрольная строка DPC@20, 11 классов  | авторов | —                   | `dpc_s20_pcf.yaml`           |
| форма: LDIF против MGN                 | наши    | —                   | `shape_chamfer_rescore.yaml` |

Пять строк методов наследуют общий протокол `configs/protocols/boxes.yaml`, контрольные
строки — протокол `configs/igibson/protocols/igibson_pcf.yaml`.

## Уровень 1: пересчёт по готовым предсказаниям

1. Получить папку `igibson_bench` с эталоном и предсказаниями всех конфигураций
   (её создают шаги уровня 2: `dpc_build_igibson_manifest.py` и `dpc_batch_convert.py`).
2. Заменить `/PATH/TO/...` в нужных `*_rescore.yaml` и `*_pcf.yaml` на путь к этой папке.
3. Загрузчик проверяет, что файлы панорам из перечня существуют, хотя в этом режиме их
   не читает. Если самих панорам нет, достаточно пустых заглушек:
   ```bash
   python - <<'EOF'
   import json, pathlib, sys
   manifest = pathlib.Path(sys.argv[1])
   for line in manifest.read_text().splitlines():
       path = pathlib.Path(json.loads(line)["input"])
       path = path if path.is_absolute() else manifest.parent / path
       path.parent.mkdir(parents=True, exist_ok=True)
       path.touch()
   EOF
   ```
4. Запустить:
   ```bash
   pano3dbench run --config configs/igibson/runs/dpc_rescore.yaml
   cat artifacts/dpc_igibson/report.md
   ```

## Уровень 2: через окружение модели

```bash
cd $TBX && make setup

pano3dbench env prepare --env configs/environments/dpc.yaml --dry-run
pano3dbench env prepare --env configs/environments/dpc.yaml
pano3dbench env prepare --env configs/environments/dpc.yaml --verify

conda run -n Pano3D python $TBX/scripts/igibson/dpc_build_igibson_manifest.py \
    $DPC $DPC/data/igibson_bench $DPC/data/igibson/test.json

pano3dbench run --config configs/igibson/runs/dpc.yaml
pano3dbench run --config configs/igibson/runs/dpc_s20.yaml
pano3dbench run --config configs/igibson/runs/im3d.yaml
pano3dbench run --config configs/igibson/runs/im3d_trained.yaml
pano3dbench run --config configs/igibson/runs/total3d.yaml
```

- `--dry-run` показывает, что уже на месте и что требует сети, ничего не меняя.
- Если окружение собрано вручную или администратором, вместо сборки — `--adopt`.
- `--verify` пересчитывает контрольные суммы весов на диске.
- Перед запуском в конфигурациях нужно указать путь к `$DPC/data/igibson_bench/manifest.jsonl`.
- Итог строки — `artifacts/<имя>/report.md`, происхождение (окружение, коммит, веса) —
  `artifacts/<имя>/run.json`.

Конфигурации отличаются только `runner_args`: какой папкой вывода пользоваться и как
назвать метод. Почему у методов один раннер — в [methods.md](methods.md).

Эталон строится тем же кодом, что и предсказания (`runners/scene_payload.py`,
`dpc_runner.convert_data_pkl`). После любой правки этого кода эталон нужно построить
заново, иначе стороны станут несравнимы.

**Предсказания в JSON для уровня 1** получаются так:

```bash
conda run -n Pano3D python $TBX/scripts/dpc/dpc_batch_convert.py \
    $DPC/data/igibson_bench/manifest.jsonl $DPC/data/igibson_bench/pred \
    --repo $DPC --pred-root out/pano3d
```

Для других методов: `--pred-root out/pano3d_im3d --out-dir .../pred_im3d` и так далее,
чтобы предсказания двух методов не попали в одну строку.

## Уровень 3: вывод и обучение с нуля

Вывод (по одному запуску на GPU для каждой конфигурации, каждый в свою папку):

```bash
cd $DPC
WANDB_MODE=dryrun python main.py configs/pano3d_igibson.yaml \
    --data.split data/igibson --mode qtest --log.path out/pano3d
WANDB_MODE=dryrun python main.py configs/pano3d_igibson.yaml \
    --data.split data/igibson --mode qtest \
    --model.scene_gcn.relation_adjust False --log.path out/pano3d_im3d
WANDB_MODE=dryrun python main.py configs/pano3d_igibson_total3d.yaml \
    --data.split data/igibson --mode qtest --log.path out/pano3d_total3d
```

DPC@20 — тот же запуск, но с `optimize_steps: 20` в настройках оптимизации отношений и
с `--log.path out/pano3d_dpc_s20`. Im3D-Pano (обученная) — с `configs/pano3d_igibson_im3d.yaml`,
`--model.scene_gcn.relation_adjust False` и `--log.path out/pano3d_im3d_trained`.
Имена папок вывода должны совпадать с `--pred-root` в конфигурациях `configs/igibson/runs/`. Конфигурации
`configs/pano3d_igibson*.yaml` лежат
в самом DeepPanoContext; наши `configs/igibson/dpc/inference/pano3d_igibson_im3d.yaml` и
`pano3d_igibson_total3d.yaml` нужно положить рядом с ними.

Обучение голов формы:

```bash
conda run -n Pano3D python $TBX/scripts/igibson/mgn_make_gt.py $DPC --processes 8
conda run -n Pano3D python main.py configs/mgnet_igibson.yaml --mode qtrain
conda run -n Pano3D python main.py configs/ldif_igibson_scratch.yaml --mode qtrain
```

`mgnet_igibson.yaml` и `ldif_igibson_scratch.yaml` берутся из `configs/igibson/dpc/training/` и
кладутся в `configs/` DeepPanoContext. Для обучения расширение ChamferDistance на CUDA
должно действительно собраться (для вывода его можно не собирать).

Правки кода DeepPanoContext, без которых он не работает вне Colab и на этих данных,
применяет `scripts/dpc/dpc_postsetup.py`; повторный запуск ничего не ломает.

## Сравнение формы

Метрики формы и их устройство — в [metrics.md](metrics.md#форма-объектов). Обычный
`qtest` сохраняет только рамки, поэтому для формы нужен отдельный вывод с выгрузкой сеток:

1. Правка `DPC_PATCH_EXPORT_OBJ_MESH` из `scripts/dpc/dpc_postsetup.py` в режиме `test`
   сохраняет каноническую сетку каждого предмета в `obj_mesh/<i>.ply` рядом с `data.pkl`.
2. Вывод с выгрузкой:
   ```bash
   conda run -n Pano3D python -u main.py configs/pano3d_igibson_total3d.yaml \
     --data.split data/igibson --mode test --log.path out/t3d_export_full
   ```
   Для Im3D-Pano (обученной) — `configs/pano3d_igibson_im3d.yaml`, `--log.path out/im3d_export_full`.
3. Эталон с формой: `IGIBSON_OBJ_ROOT` добавляет каждому предмету его замкнутую сетку
   `<корень>/<model_path>/mesh_watertight.ply`:
   ```bash
   cd $DPC && IGIBSON_OBJ_ROOT=data/igibson_obj conda run -n Pano3D python \
     $TBX/scripts/igibson/dpc_build_igibson_manifest.py $DPC $DPC/data/igibson_bench_shape data/igibson/test.json
   ```
4. Предсказания с формой:
   ```bash
   conda run -n Pano3D python $TBX/scripts/dpc/dpc_batch_convert.py \
     $DPC/data/igibson_bench_shape/manifest.jsonl $DPC/data/igibson_bench_shape/pred_total3d \
     --repo $DPC --pred-root out/t3d_export_full --obj-mesh-sibling obj_mesh
   ```
5. Подсчёт без GPU, по одному запуску `shape_chamfer_rescore.yaml` на метод.

Перед полным прогоном стоит сверить на 2–3 предметах размеры предсказанной и эталонной
сетки.

## Результаты

500 тестовых сцен, среднее. Сопоставление предметов по ориентированному 3D-IoU 0.15.

| Показатель                       |  DPC@100 |    DPC@20 | Im3D-Pano (абляция) | Im3D-Pano (обученная) | Total3D-Pano |
|----------------------------------|---------:|----------:|--------------------:|----------------------:|-------------:|
| mAP по набору ↑                  |    0.292 | **0.356** |               0.293 |                 0.224 |        0.224 |
| mAP по сцене ↑                   |    0.314 | **0.381** |               0.279 |                 0.227 |        0.227 |
| IoU макета, среднее ↑            |    0.748 |     0.897 |               0.897 |                 0.897 |        0.897 |
| IoU макета, медиана ↑            |    0.919 |     0.926 |               0.926 |                 0.926 |        0.926 |
| столкновения предметов ↓         |   0.0044 |    0.0077 |              0.0173 |                0.0227 |       0.0227 |
| нарушения макета (допуск 0.05) ↓ | 0.413 \* |     0.501 |               0.377 |                 0.799 |        0.799 |
| проникание в макет ↓             | 0.258 \* |     0.298 |               0.220 |                 0.576 |        0.576 |

\* У DPC@100 эти два показателя занижены: 86 сцен с вырожденными рамками пропускаются
при подсчёте. Честные числа для DPC — в столбце DPC@20 (вырожденных сцен 0).

- **Веса:** DPC и Im3D-Pano (абляция) — авторов; Im3D-Pano (обученная) и Total3D-Pano —
  обучены нами (голова формы).
- **Оптимизация отношений и Scene-GCN:** у DPC включены; у Im3D-Pano (абляции)
  оптимизация выключена, но Scene-GCN присутствует; у обученных строк Scene-GCN нет вовсе.

**Контрольные строки на 11 классах** (`chair, sofa, table, fridge, sink, door, floor_lamp,
bottom_cabinet, top_cabinet, sofa_chair, dryer`), mAP по набору: DPC@100 — 45.21 %,
DPC@20 — 54.26 %. Авторы DPC публикуют 52.69 % на том же наборе классов. Таблица выше
усредняет по всем классам эталона, поэтому сравнивать её с опубликованными числами нельзя.

**Форма** (462 сцены с сопоставленными парами из 500, `normalize: canonical`):

| Метод                           | Чамфер ↓ (среднее / медиана) | F-мера ↑ (среднее / медиана) |
|---------------------------------|-----------------------------:|-----------------------------:|
| Total3D-Pano (MGN, явная сетка) |        **0.00445** / 0.00245 |            **0.939** / 0.983 |
| Im3D-Pano (LDIF, неявная форма) |            0.00492 / 0.00302 |                0.930 / 0.967 |

Рамки у двух методов общие, поэтому сравнение парное, по сценам: Чамфер(MGN − LDIF) =
−0.00047, 95 % ДИ [−0.00070, −0.00023], MGN лучше в 304 из 462 сцен; F-мера(MGN − LDIF) =
+0.0086, 95 % ДИ [+0.0042, +0.0128]. Оба интервала не содержат нуля: MGN значимо точнее.

## Наблюдения

1. **Макет — общий контроль.** IoU макета совпадает у всех устойчивых конфигураций
   до шестого знака (0.897414 / 0.926493): это одна и та же оценка HorizonNet. Побитово
   числа не равны: каждая конфигурация — отдельный запуск, и углы комнаты расходятся
   в шестом знаке из-за недетерминизма GPU. Значение 0.748 у DPC@100 — следствие
   расходимости оптимизации, а не худшая оценка комнаты.
2. **По рамкам LDIF и MGN неразличимы.** У Im3D-Pano (обученной) и Total3D-Pano рамки
   предметов совпадают во всех 500 сценах (проверено по файлам), поэтому все показатели
   рамок одинаковы. Рамки даёт общий BEN, а голова формы на них не влияет. Сравнивать
   эти методы можно только по форме.
3. **Вклад оптимизации отношений** (DPC@20 против абляции): выше детектирование (0.356
   против 0.293), меньше столкновений (0.0077 против 0.0173), но больше проникание
   (0.298 против 0.220).
4. **Роль Scene-GCN.** Без неё (обученные строки) детектирование падает (0.224 против
   0.293), проникание растёт (0.576 против 0.220). Поэтому обученные строки нельзя прямо
   сравнивать с абляцией: чистая пара для формы — обученные строки между собой.
5. **Неустойчивость выпуска DPC.** При 100 шагах оптимизации 86 из 500 макетов
   вырождаются, mAP по набору 0.292 — почти как без оптимизации отношений (0.293). При 20
   шагах вырождений нет, mAP 0.356. Это настройка, а не свойство метода.

## Оговорки

- Im3D-Pano (абляция) — DPC с выключенной оптимизацией отношений на весах авторов;
  Im3D-Pano (обученная) — голова LDIF, обученная с нуля. Это разные строки, и абляцию
  нужно называть абляцией.
- IoU макета — общий вход всех методов, а не различающий показатель.
- Вычисления: NVIDIA A100-40GB, окружение `Pano3D` (torch 1.7.1+cu110, CUDA 11,
  detectron2, gibson2); ядро бенчмарка — Python 3.9, только стандартная библиотека.
  Зерна заданы в конфигурациях, `run.json` записывает конфигурацию, зерно, ревизию git
  и происхождение окружения.
- **Проверить перед новым пересчётом.** Числа выше получены до исправления ориентации
  рамок в `runners/dpc_runner.py` (оси `basis` DPC хранятся столбцами). Если перевести
  предсказания заново исправленным кодом, числа могут измениться: их нужно пересчитать
  и сверить с таблицей.
