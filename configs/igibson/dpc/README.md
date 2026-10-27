# Конфигурации DPC для iGibson

Эти файлы читает `main.py` DeepPanoContext. Их кладут в `configs/` клона DeepPanoContext
рядом с `configs/pano3d_igibson.yaml` авторов. Полный порядок команд, пути данных и
результаты — [docs/igibson.md](../../../docs/igibson.md).

```text
dpc/
├── training/    обучение голов формы: LDIF с нуля и MGN
└── inference/   вывод методов, у которых нет конфигурации авторов: Im3D-Pano (обученная), Total3D-Pano
```

Раскладка та же, что у 3D-FRONT ([../../front3d/dpc/README.md](../../front3d/dpc/README.md)).
Отличие одно: на iGibson детектор, HorizonNet, BEN и Scene-GCN не обучаются, берутся веса
авторов из окружения `configs/environments/dpc.yaml`.

## Обучение

| Порядок | Компонента | Конфигурация                         | Читает                         |
|--------:|------------|--------------------------------------|--------------------------------|
|       1 | цели MGN   | `scripts/igibson/mgn_make_gt.py`     | сетки предметов iGibson        |
|       2 | MGN        | `training/mgnet_igibson.yaml`        | предметы iGibson и цели шага 1 |
|       3 | LDIF       | `training/ldif_igibson_scratch.yaml` | предметы iGibson               |

Оба обучения — `--mode qtrain`. Для них расширение ChamferDistance на CUDA должно
действительно собраться.

## Вывод

| Конфигурация                            | Метод                                                  | Папка вывода                    |
|-----------------------------------------|--------------------------------------------------------|---------------------------------|
| `configs/pano3d_igibson.yaml` авторов   | DPC@100; с `relation_adjust False` — абляция Im3D-Pano | `out/pano3d`, `out/pano3d_im3d` |
| `inference/pano3d_igibson_im3d.yaml`    | Im3D-Pano (обученная): ствол DPC с головой LDIF шага 3 | `out/pano3d_im3d_trained`       |
| `inference/pano3d_igibson_total3d.yaml` | Total3D-Pano: ствол DPC с головой MGN шага 2           | `out/pano3d_total3d`            |

DPC@20 — та же конфигурация авторов с `optimize_steps: 20`, папка `out/pano3d_dpc_s20`.
Папки вывода должны совпадать с `--pred-root` в `configs/igibson/runs/`. Для формы те же
конфигурации запускаются в режиме `--mode test` с выгрузкой сеток.
