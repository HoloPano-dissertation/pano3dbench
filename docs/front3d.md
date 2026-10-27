# Воспроизведение на 3D-FRONT

Набор панорам 3D-FRONT с разметкой собирает отдельный генератор
[front3d-pano](https://github.com/HoloPano-dissertation/front3d-pano). Здесь описано, что
делается дальше: обучение компонент трёх методов на этом наборе, вывод на его тестовой части
и подсчёт показателей. Состав набора и как собрать именно его — в
[front3d_dataset.md](front3d_dataset.md).

Ниже `$EXP` — папка эксперимента генератора, `$DPC` — клон DeepPanoContext,
`$TBX` — этот репозиторий. Во всех конфигурациях `/PATH/TO/front3d` нужно заменить на
`$EXP`, а `/PATH/TO/out/<обучение>/RUN/` — на папку запуска DPC с нужными весами:
конфигурации DPC не раскрывают переменные окружения в путях.

## Что нужно

- набор, собранный генератором `front3d-pano` (папка эксперимента: `coco/`, `objects/`,
  `ground_truth/`, `manifests_gt/`, `splits/`, `state/classes.json`);
- форматы методов, построенные по этому набору скриптами
  [scripts/front3d/adapters/](../scripts/front3d/README.md): `horizonnet/`, вырезки
  и цели LDIF/MGN в `objects/`, `dpc_dataset/`, `dpc_scenes/`;
- окружение DeepPanoContext (`configs/environments/dpc.yaml`,
  см. [igibson.md](igibson.md#уровень-2-через-окружение-модели)) и правки
  `scripts/dpc/dpc_postsetup.py`;
- GPU для обучения и вывода.

Список классов набора задаётся переменной окружения, иначе DPC подставит словарь iGibson
из 56 классов:

```bash
export PANO3D_CLASSES=$(python -c "import json,sys;print(','.join(json.load(open(sys.argv[1]))['classes']))" \
  $EXP/state/classes.json)
```

Все команды DPC ниже запускаются из `$DPC` в окружении `Pano3D` с `PYTHONPATH=$DPC`.

## 1. Обучение компонент

Порядок и конфигурации — в [configs/front3d/dpc/README.md](../configs/front3d/dpc/README.md):

| Порядок | Компонента    | Чем                                                   | Читает                    |
|--------:|---------------|-------------------------------------------------------|---------------------------|
|       1 | детектор      | `training/train_detector.py`                          | `coco/`                   |
|       2 | HorizonNet    | `training/train_horizonnet.py`                        | `horizonnet/`             |
|       3 | первая стадия | `configs/front3d/dpc/first_stage_front3d.yaml`        | `dpc_scenes/`             |
|       4 | BEN           | `configs/front3d/dpc/bdb3d_estimation_front3d.yaml`   | `dpc_stage1/`             |
|       5 | LDIF          | `configs/front3d/dpc/ldif_front3d.yaml`               | `objects/`                |
|       6 | MGN           | `configs/front3d/dpc/mgnet_front3d.yaml`              | `objects/`                |
|       7 | Scene-GCN     | `configs/front3d/dpc/relation_scene_gcn_front3d.yaml` | `dpc_stage1/`, `objects/` |

Детектор и HorizonNet — [training/README.md](../training/README.md). Остальное обучается
кодом DeepPanoContext: `python main.py <конфигурация> --mode train`.

**Первая стадия** — не обучение, а вывод обученного детектора и HorizonNet по обучающей и
проверочной частям. Она сохраняет сцены вместе с найденными предметами в `dpc_stage1/`, и
BEN со Scene-GCN учатся уже на этих детекциях, а не на эталонных рамках:

```bash
python main.py $TBX/configs/front3d/dpc/first_stage_front3d.yaml --mode test
python $TBX/scripts/front3d/eval/index_stage1.py $EXP $EXP/dpc_stage1
```

`index_stage1.py` пишет `dpc_stage1/train.json` (обучающая часть) и `dpc_stage1/test.json`
(проверочная часть: так её называет загрузчик DPC). Камеры без видимых предметов в списки не
попадают. Скрипт завершается с ошибкой, если в первой стадии нашлись виды тестовой части.

BEN обучается с нуля: веса авторов рассчитаны на 56 классов.

Скрипты `training/` начинают полное обучение, только если в `state/training_gate.json`
набора стоит `"training_approved": true` с причиной. Ставит его человек после проверки набора.
Ворота снова закрыты, если после одобрения изменился состав или выгрузки генератора
(`splits/rooms.jsonl`, отчёты этапов в `state/`).

Перед долгим обучением BEN или Scene-GCN полезно прогнать загрузчик без модели: он
перебирает несколько сотен пакетов и сразу показывает, если в каком-то пакете пропали предметы:

```bash
python $TBX/scripts/dpc/dpc_loader_probe.py $TBX/configs/front3d/dpc/bdb3d_estimation_front3d.yaml --steps 300
```

## 2. Вывод на тестовой части

Из обученных компонент собираются четыре конфигурации:

| Конфигурация DPC                                  | Метод                                | Папка вывода                 |
|---------------------------------------------------|--------------------------------------|------------------------------|
| `configs/front3d/dpc/pano3d_front3d.yaml`         | DPC, 100 шагов оптимизации отношений | `out/pano3d_front3d`         |
| `configs/front3d/dpc/pano3d_front3d_s20.yaml`     | DPC, 20 шагов                        | `out/pano3d_front3d_s20`     |
| `configs/front3d/dpc/pano3d_front3d_im3d.yaml`    | Im3D-Pano: без Scene-GCN, LDIF       | `out/pano3d_front3d_im3d`    |
| `configs/front3d/dpc/pano3d_front3d_total3d.yaml` | Total3D-Pano: без Scene-GCN, MGN     | `out/pano3d_front3d_total3d` |

Во всех четырёх первый вес — Scene-GCN: в нём же лежат замороженные BEN и LDIF, с которыми
он учился. Im3D-Pano получает тот же LDIF третьим весом, Total3D-Pano — MGN. Детектор
задаётся отдельно (`model.detector.weight`): в весах Scene-GCN его нет.

1. Список тестовых видов без видов, где не виден ни один предмет:
   ```bash
   python $TBX/scripts/front3d/eval/make_test_split.py $EXP
   ```
   Пишет `dpc_scenes/test_split.json` и перечень `eval/manifest_test.jsonl` — тот же набор
   видов (5707 на нашем наборе, 21 вид без предметов исключён).
2. Вывод каждой конфигурации:
   ```bash
   python main.py $TBX/configs/front3d/dpc/pano3d_front3d.yaml --mode qtest --model.scene_gcn.relation_adjust True
   python main.py $TBX/configs/front3d/dpc/pano3d_front3d_s20.yaml --mode qtest --model.scene_gcn.relation_adjust True
   python main.py $TBX/configs/front3d/dpc/pano3d_front3d_im3d.yaml --mode qtest
   python main.py $TBX/configs/front3d/dpc/pano3d_front3d_total3d.yaml --mode qtest
   ```
   Предсказания пишутся в `out/<папка>/<запуск>/visualization/<дом>/<комната>__<вид>/data.pkl`.
   Долгий вывод можно разбить на части, передав свой список видов: `--data.split <файл>.json`.
   Части одной конфигурации нельзя запускать в одну и ту же секунду: папка запуска DPC
   называется по времени старта, и вторая часть упадёт с `FileExistsError`.

## 3. Подсчёт

1. Перевести предсказания каждой конфигурации в общий формат:
   ```bash
   python $TBX/scripts/dpc/dpc_batch_convert.py $EXP/eval/manifest_test.jsonl $EXP/eval/pred_dpc \
       --repo $DPC --pred-root out/pano3d_front3d
   ```
   и так же для `pred_dpc_s20`, `pred_im3d`, `pred_total3d`. Конвертер сам переводит имена
   вида `<дом>/<комната>/<вид>` в папки DPC `<дом>-<комната>__<вид>` и рамки DPC в систему
   камеры эталона, а рядом пишет перечень `manifest_pred_<имя>.jsonl` из переведённых видов.
2. Посчитать показатели:
   ```bash
   pano3dbench run --config configs/front3d/runs/dpc_rescore.yaml
   ```
   и так же `dpc_s20_rescore.yaml`, `im3d_rescore.yaml`, `total3d_rescore.yaml`. Все они
   наследуют общий протокол `configs/protocols/boxes.yaml`: mAP при ориентированном
   3D-IoU 0.15, IoU макета, столкновения, нарушения макета с допуском 0.05, проникание.

Эталон — `ground_truth/test/...` набора (все предметы, включая заслонённые; система камеры,
метры).

**Проверка цепочки эталоном.** `configs/front3d/runs/gt_sanity.yaml` подаёт эталон вместо
предсказаний: mAP и IoU макета должны быть 1.0. Физические показатели самого эталона — точка
отсчёта для методов: столкновения 0.14, нарушения 0.22, проникание 0.053.

## 4. Форма

Форма считается на 500 тестовых видах. Список зафиксирован в
[configs/front3d/shape500_views.txt](../configs/front3d/shape500_views.txt) и строится заново
так:

```bash
python $TBX/scripts/front3d/eval/make_shape500.py $EXP
cp $EXP/eval/shape500/test_split_shape500.json $EXP/dpc_scenes/
```

Скрипт выбирает виды `random.Random(20261016).sample` из списка тестовых видов и пишет эталон
`eval/shape500/gt/`, в котором форма каждого предмета — `mesh_watertight.ply` из его папки в
`objects/`. Список копируется в `dpc_scenes/`, потому что DPC ищет сцены относительно папки
файла разбиения.

Вывод с выгрузкой сеток (`--mode test`, сетки пишет правка `DPC_PATCH_EXPORT_OBJ_MESH`):

```bash
python main.py $TBX/configs/front3d/dpc/shape500_im3d.yaml --mode test
python main.py $TBX/configs/front3d/dpc/shape500_total3d.yaml --mode test
python $TBX/scripts/dpc/dpc_batch_convert.py $EXP/eval/shape500/manifest_test.jsonl $EXP/eval/shape500/pred_im3d \
    --repo $DPC --pred-root out/shape500_im3d --obj-mesh-sibling obj_mesh
pano3dbench run --config configs/front3d/runs/shape_im3d_rescore.yaml
```

и так же для Total3D-Pano. Подсчёт всегда идёт по `eval/shape500/manifest_gt.jsonl`, а не по
перечню конвертера: тот ссылается на эталон без замкнутых сеток. Сопоставление по
ориентированному 3D-IoU 0.15, `normalize: canonical`, 2048 точек, порог F-меры 0.1.
Устройство показателей — в [metrics.md](metrics.md#форма-объектов).
