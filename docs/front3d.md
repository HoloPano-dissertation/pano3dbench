# Воспроизведение на 3D-FRONT

Набор панорам 3D-FRONT с разметкой собирает отдельный генератор
[front3d-pano](https://github.com/HoloPano-dissertation/front3d-pano). Здесь описано, что
делается дальше: обучение компонент трёх методов на этом наборе и оценка на его тестовой
части.

> **Рецепт перенесён не полностью.** Конфигурации вывода, конфигурации оценки и скрипты
> запуска на кластере пока лежат только на кластере. Ниже описана цепочка, которой
> получены числа; разделы, где файлов в репозитории ещё нет, помечены.

## Что нужно

- набор, собранный генератором `front3d-pano` (папка эксперимента: `coco/`, `objects/`,
  `ground_truth/`, `manifests_gt/`, `state/classes.json`);
- форматы методов, построенные по этому набору скриптами
  [scripts/front3d/adapters/](../scripts/front3d/README.md): `horizonnet/`, вырезки
  и цели LDIF/MGN в `objects/`, `dpc_dataset/`, `dpc_scenes/`;
- окружение DeepPanoContext (`configs/environments/dpc.yaml`,
  см. [igibson.md](igibson.md#уровень-2-через-окружение-модели));
- GPU для обучения и вывода.

Список классов набора задаётся переменной окружения, иначе DPC подставит словарь iGibson
из 56 классов:

```bash
export PANO3D_CLASSES=$(python -c "import json,sys;print(','.join(json.load(open(sys.argv[1]))['classes']))" \
  /PATH/TO/front3d/state/classes.json)
```

## 1. Обучение компонент

Шесть обучений, порядок и конфигурации — в
[configs/front3d/dpc/README.md](../configs/front3d/dpc/README.md):

| Порядок | Компонента | Чем                                                   | Читает                    |
|--------:|------------|-------------------------------------------------------|---------------------------|
|       1 | детектор   | `training/train_detector.py`                          | `coco/`                   |
|       2 | HorizonNet | `training/train_horizonnet.py`                        | `horizonnet/`             |
|       3 | BEN        | `configs/front3d/dpc/bdb3d_estimation_front3d.yaml`   | `dpc_scenes/`             |
|       4 | LDIF       | `configs/front3d/dpc/ldif_front3d.yaml`               | `objects/`                |
|       5 | MGN        | `configs/front3d/dpc/mgnet_front3d.yaml`              | `objects/`                |
|       6 | Scene-GCN  | `configs/front3d/dpc/relation_scene_gcn_front3d.yaml` | `dpc_scenes/`, `objects/` |

Детектор и HorizonNet — [training/README.md](../training/README.md). BEN, LDIF, MGN и
Scene-GCN обучаются кодом DeepPanoContext: `python main.py <конфигурация> --mode train`,
конфигурации кладутся в `configs/` DeepPanoContext. BEN обучается с нуля: веса авторов
рассчитаны на 56 классов.

Скрипты `training/` начинают полное обучение, только если в `state/training_gate.json`
набора стоит `"training_approved": true` с причиной. Ставит его человек после проверки набора.

## 2. Вывод на тестовой части

Из четырёх конфигураций собираются методы: DPC при 100 и при 20 шагах оптимизации
отношений, Im3D-Pano (без оптимизации, LDIF) и Total3D-Pano (без Scene-GCN, MGN).

1. Список тестовых видов без видов, где не виден ни один предмет: `dpc_scenes/test_split.json`
   и перечень `manifest_test.jsonl`.
2. Вывод DPC для каждой конфигурации: `python main.py <конфигурация> --mode qtest`.
   Предсказания пишутся в `out/<папка конфигурации>/<запуск>/visualization/<дом>/<комната>__<вид>/data.pkl`.

> **Нет в репозитории:** конфигурации вывода (`*.v2.yaml`) и скрипты запуска на кластере.

## 3. Подсчёт

1. Перевести предсказания в общий формат:
   ```bash
   conda run -n Pano3D python scripts/dpc/dpc_batch_convert.py \
       <manifest_test.jsonl> <папка предсказаний JSON> --repo $DPC --pred-root <папка вывода>
   ```
   Конвертер сам переводит имена вида `<дом>/<комната>/<вид>` в папки DPC
   `<дом>-<комната>__<вид>` и рамки DPC в систему камеры эталона.
2. Посчитать показатели с общим протоколом `configs/protocols/boxes.yaml`: mAP при
   ориентированном 3D-IoU 0.15, IoU макета, столкновения, нарушения макета с допуском 0.05,
   проникание.

Эталон — `ground_truth/test/...` набора (все предметы, включая заслонённые; система камеры,
метры); перечень `manifests_gt/test.jsonl` подходит без изменений.

> **Нет в репозитории:** конфигурации оценки. Их место — `configs/front3d/runs/`.

**Проверка цепочки эталоном.** Если подать эталон вместо предсказаний, mAP и IoU макета
должны быть 1.0. Физические показатели самого эталона — точка отсчёта для методов:
столкновения 0.14, нарушения 0.22, проникание 0.053.

## Форма

Форма считается на 500 тестовых видах: вывод с выгрузкой сеток (`--mode test`), эталон —
`mesh_watertight.ply` из `objects/`, сопоставление по ориентированному 3D-IoU 0.15,
`normalize: canonical`, 2048 точек, порог F-меры 0.1, 95 % ДИ бутстрэпом (2000 повторов).
Устройство показателей — в [metrics.md](metrics.md#форма-объектов).

> **Нет в репозитории:** список 500 видов и конфигурации вывода с выгрузкой сеток.
