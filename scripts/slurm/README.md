# Запуск на кластере через Slurm

Шаблоны задач Slurm для обучения, вывода и подсчёта на обоих наборах. Порядок шагов у обоих
наборов один и тот же — обучение компонент → вывод → перевод предсказаний → подсчёт — и
совпадает с таблицами в [configs/front3d/dpc/README.md](../../configs/front3d/dpc/README.md)
и [configs/igibson/dpc/README.md](../../configs/igibson/dpc/README.md). Что делает каждый шаг
и почему — в [docs/front3d.md](../../docs/front3d.md) и [docs/igibson.md](../../docs/igibson.md);
здесь только как отправить его в очередь.

| Файл                        | За что отвечает                                                                                                                                          |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `front3d.sh`                | шаги 3D-FRONT: `components`, `stage1`, `heads`, `gcn`, `inference`, `score`, `shape`                                                                     |
| `igibson.sh`                | шаги iGibson: `heads`, `inference`, `score`                                                                                                              |
| `common.sh`                 | общее для обоих: переменные, отправка задачи с зависимостью, заполнение конфигурации, журнал                                                             |
| `dpc_main.sbatch`           | одна задача `main.py` DeepPanoContext: `<конфигурация> <режим> [аргументы]`, GPU                                                                         |
| `dpc_python.sbatch`         | один скрипт Python в окружении `Pano3D` из папки клона DPC (переводы, списки видов, обучение `training/`)                                                |
| `bench_run.sbatch`          | `pano3dbench run --config <файл>` в окружении комплекса                                                                                                  |
| `fill_config.py`            | копия конфигурации с подставленными путями вместо `/PATH/TO/...`                                                                                         |
| `dpc_site/sitecustomize.py` | `file_system` для обмена тензорами между процессами загрузчика: без него обучение DPC с несколькими процессами падает на «слишком много открытых файлов» |

## Переменные

| Переменная                           | Что это                                                                   | По умолчанию                               |
|--------------------------------------|---------------------------------------------------------------------------|--------------------------------------------|
| `DPC`                                | клон DeepPanoContext                                                      | обязательна                                |
| `DPC_PYTHON`                         | `python` окружения `Pano3D`                                               | обязательна                                |
| `EXP`                                | папка эксперимента 3D-FRONT (только `front3d.sh`)                         | обязательна                                |
| `TBX_PYTHON`                         | `python` окружения комплекса                                              | `.venv/bin/python` репозитория             |
| `LOGS`                               | куда пишутся журналы задач и `submitted.txt`                              | `$EXP/logs`, `$DPC/logs`                   |
| `CONFIGS`                            | куда кладутся заполненные конфигурации                                    | `$EXP/slurm_configs`, `$DPC/slurm_configs` |
| `AFTER`                              | номер задачи, после которой начинать шаг (`afterok`)                      | нет                                        |
| `DETECTOR_INIT`, `DETECTOR_ARGS`     | веса COCO для детектора и его дополнительные аргументы (шаг `components`) | —                                          |
| `SBATCH_PARTITION`, `SBATCH_EXCLUDE` | раздел и исключённые узлы; их читает сам `sbatch`                         | раздел кластера по умолчанию               |

`PANO3D_CLASSES` для 3D-FRONT `front3d.sh` берёт из `$EXP/state/classes.json` сам.

## Пример

```bash
export DPC=~/pano3dbench/.pano3dbench/repos/Pano3D
export DPC_PYTHON=~/.conda/envs/Pano3D/bin/python
export EXP=~/3dfront_midi_v8
export SBATCH_PARTITION=p5gpuA100 SBATCH_EXCLUDE=p7n1

scripts/slurm/front3d.sh components
scripts/slurm/front3d.sh stage1
scripts/slurm/front3d.sh heads
scripts/slurm/front3d.sh gcn
scripts/slurm/front3d.sh inference
AFTER=<номер последней задачи вывода> scripts/slurm/front3d.sh score
scripts/slurm/front3d.sh shape
```

Каждая команда печатает номера отправленных задач и дописывает их в `$LOGS/submitted.txt`.
Внутри шага задачи связаны зависимостями (например, указатель первой стадии ждёт её вывода),
между шагами — нет: следующий шаг запускают, когда предыдущий закончился и проверен.

## Как заполняются конфигурации

`fill_config.py` подставляет `$EXP` вместо `/PATH/TO/front3d`, `$DPC` вместо `/PATH/TO/Pano3D`,
а `/PATH/TO/out/<обучение>/RUN/<файл>` заменяет на **самый новый** запуск
`$DPC/out/<обучение>/<запуск>/`, где этот файл есть. Относительные `extends` и `output_dir`
конфигураций запуска становятся абсолютными. Если что-то не заполнилось, задача не
отправляется. Подстановка идёт в момент отправки, поэтому `gcn` запускают после того, как
BEN и LDIF доучены, а `inference` — после Scene-GCN. Если нужен не самый новый запуск,
поправьте копию в `$CONFIGS` и отправьте задачу вручную.

Конфигурации iGibson путей не содержат: `igibson.sh` копирует их в `configs/` клона DPC,
где их ищет `main.py`.

## Чего здесь нет

- Сборки набора 3D-FRONT и форматов методов: это генератор `front3d-pano` и
  [scripts/front3d/](../front3d/README.md).
- Сравнения формы на iGibson: оно делается командами из
  [docs/igibson.md](../../docs/igibson.md#сравнение-формы).
- Проверки ворот обучения: её делают сами скрипты `training/`.
