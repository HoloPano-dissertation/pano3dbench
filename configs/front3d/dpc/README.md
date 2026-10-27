# Конфигурации DPC для 3D-FRONT

Эти файлы читает `main.py` DeepPanoContext. Полный порядок команд —
[docs/front3d.md](../../../docs/front3d.md).

```text
dpc/
├── training/    обучение компонент: первая стадия, BEN, LDIF, MGN, Scene-GCN
└── inference/   вывод готовых методов: DPC@100, DPC@20, Im3D-Pano, Total3D-Pano, форма на 500 видах
```

Раскладка та же, что у iGibson ([../../igibson/dpc/README.md](../../igibson/dpc/README.md)).
На кластере шаги отправляются в очередь `scripts/slurm/front3d.sh <шаг>`
([scripts/slurm/README.md](../../../scripts/slurm/README.md)): `components` — шаги 1–2,
`stage1` — шаг 3 с указателем, `heads` — шаги 4–6, `gcn` — шаг 7, затем `inference`,
`score` и `shape`.

## Обучение

Детектор и оценщик планировки не зависят ни от масштаба, ни от классов и обучаются первыми.
Первая стадия прогоняет их по обучающей и проверочной частям и сохраняет сцены с найденными
предметами в `dpc_stage1/`; на этих детекциях учатся BEN и Scene-GCN.

| Порядок | Компонента    | Конфигурация                               | Читает                    |
|--------:|---------------|--------------------------------------------|---------------------------|
|       1 | Детектор      | `training/train_detector.py`               | `coco/`                   |
|       2 | HorizonNet    | `training/train_horizonnet.py`             | `horizonnet/`             |
|       3 | первая стадия | `training/first_stage_front3d.yaml`        | `dpc_scenes/`             |
|       4 | BEN           | `training/bdb3d_estimation_front3d.yaml`   | `dpc_stage1/`             |
|       5 | LDIF          | `training/ldif_front3d.yaml`               | `objects/`                |
|       6 | MGN           | `training/mgnet_front3d.yaml`              | `objects/`                |
|       7 | Scene-GCN     | `training/relation_scene_gcn_front3d.yaml` | `dpc_stage1/`, `objects/` |

Шаг 3 — `--mode test`, затем `scripts/front3d/eval/index_stage1.py` строит списки
`dpc_stage1/train.json` и `dpc_stage1/test.json`. Шаг 7 требует готовых весов шагов 4 и 5:
подставьте их пути в `weight`.

## Вывод

| Конфигурация                            | Метод                                                    |
|-----------------------------------------|----------------------------------------------------------|
| `inference/pano3d_front3d.yaml`         | DPC, 100 шагов оптимизации отношений                     |
| `inference/pano3d_front3d_s20.yaml`     | DPC, 20 шагов                                            |
| `inference/pano3d_front3d_im3d.yaml`    | Im3D-Pano: Scene-GCN выключен, форма — LDIF              |
| `inference/pano3d_front3d_total3d.yaml` | Total3D-Pano: Scene-GCN выключен, форма — MGN            |
| `inference/shape500_im3d.yaml`          | Im3D-Pano на 500 видах с выгрузкой сеток (`--mode test`) |
| `inference/shape500_total3d.yaml`       | Total3D-Pano на 500 видах с выгрузкой сеток              |

Первый вес во всех — Scene-GCN (в нём замороженные BEN и LDIF), затем HorizonNet; у Im3D-Pano
третьим весом тот же LDIF, что получил Scene-GCN, у Total3D-Pano — MGN. Детектор задаётся в
`model.detector.weight`.

## Перед запуском

Замените `/PATH/TO/front3d` на папку эксперимента, а `/PATH/TO/out/<обучение>/RUN/` — на
папку запуска DPC: конфигурации DPC не раскрывают переменные окружения в путях.

Список классов задаётся переменной окружения, иначе окружение подставит словарь
iGibson из 56 категорий:

```bash
export PANO3D_CLASSES=$(python -c "import json,sys;print(','.join(json.load(open(sys.argv[1]))['classes']))" \
  /PATH/TO/front3d/state/classes.json)
```

Переменную читает патч из `configs/environments/dpc.yaml`, который применяется при
развёртывании окружения. Без переменной поведение прежнее, воспроизведение на
iGibson не ломается.

BEN обучается с нуля: у предобученных весов размерность классового кода равна 56,
и к нашему списку они не подходят.
