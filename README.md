# Pano3DBench

*[English version — README.en.md](README.en.md).*

Единый **воспроизводимый** комплекс для запуска и сравнения методов целостного понимания
трёхмерной сцены помещения по одной панораме 360°. Им сравниваются три метода —
**DeepPanoContext (DPC)**, **Im3D-Pano** и **Total3D-Pano** — по единой методике
оценивания на двух наборах: **iGibson** и собственном наборе панорам **3D-FRONT**.

Ключевая идея: три метода — **конфигурации одной программной базы DeepPanoContext**,
а не три независимых репозитория. У них общие детектор, оценка макета (HorizonNet) и
оценка ориентированных рамок (BEN); различаются они головой формы (LDIF или MGN) и
оптимизацией отношений на графовой сети сцены (Scene-GCN). Поэтому комплекс прогоняет их
по одному пути и приводит любой результат к представлению `<макет, предметы, отношения>`.

## С чего начать

```bash
make setup    # создаёт .venv и ставит пакет с зависимостями для тестов
make test     # тесты, должны закончиться словом "passed"
make smoke    # демонстрационный прогон без GPU и без моделей
```

`make smoke` печатает `Evaluated 1 scene(s). Artifacts: .../artifacts/smoke` и пишет отчёт
в `artifacts/smoke/report.md`.

Без `make` (только стандартная библиотека):

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m pano3dbench run --config configs/demo/smoke.json
```

## Воспроизведение статьи

| Набор    | Документ                           | Что воспроизводится                                                                         |
|----------|------------------------------------|---------------------------------------------------------------------------------------------|
| iGibson  | [docs/igibson.md](docs/igibson.md) | пять конфигураций методов, две контрольные строки на 11 классах, сравнение формы LDIF и MGN |
| 3D-FRONT | [docs/front3d.md](docs/front3d.md) | обучение компонент на собственном наборе и оценка на его тестовой части                     |

Строку статьи можно получить двумя путями, которые дают одинаковые числа:

```bash
pano3dbench run --config configs/igibson/runs/dpc.yaml          # через окружение модели
pano3dbench run --config configs/igibson/runs/dpc_rescore.yaml  # по готовым предсказаниям, без GPU
```

**Веса, обученные нами** (головы формы на iGibson):
[выпуск `weights-v1`](https://github.com/HoloPano-dissertation/pano3dbench/releases/tag/weights-v1) —
`total3d-pano_mgn_igibson.pth` (MGN для Total3D-Pano), `im3d-pano_ldif_igibson.pth`
(LDIF для обученной Im3D-Pano), `SHA256SUMS.txt`, `MANIFEST.md`. Веса DPC авторов
общедоступны в выпуске DeepPanoContext и здесь не распространяются.

## Что делает комплекс

`pano3dbench run --config <cfg>` выполняет цепочку

```text
конфигурация -> набор (перечень + эталон) -> модель -> показатели -> отчёт
```

и пишет в `artifacts/<имя>/` предсказания, значения по сценам, сводку (среднее, медиана,
95 % доверительный интервал), отчёт и `run.json` с конфигурацией, зерном, ревизией git
и сведениями об окружении.

Модель **не импортируется** в процесс комплекса: она работает в своём окружении и
обменивается с ним JSON-файлами, поэтому методы могут требовать несовместимые версии
Python, PyTorch и CUDA. Новый метод подключается раннером и спецификацией окружения,
без изменения ядра.

Протокол сравнения — **ориентированный 3D-IoU предметов при пороге 0.15**, а также IoU
макета, столкновения предметов, выход предметов за макет и проникание. Форму, которая
единственная различает LDIF и MGN, измеряют `mesh_chamfer` и `mesh_fscore`.

## Документация

| Документ                                                               | О чём                                                                     |
|------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [docs/igibson.md](docs/igibson.md)                                     | воспроизведение статьи на iGibson: команды, конфигурации, числа, оговорки |
| [docs/front3d.md](docs/front3d.md)                                     | воспроизведение на 3D-FRONT                                               |
| [docs/front3d_dataset.md](docs/front3d_dataset.md)                     | набор 3D-FRONT версии 1: состав, параметры, как собрать                   |
| [docs/methods.md](docs/methods.md)                                     | три метода как конфигурации DPC, выбор конфигурации запуска               |
| [docs/metrics.md](docs/metrics.md)                                     | каждый показатель, протокол классов, форма, свой показатель               |
| [docs/architecture.md](docs/architecture.md)                           | устройство, команды, окружения, кластер, подключение своей модели         |
| [src/pano3dbench/README.md](src/pano3dbench/README.md)                 | устройство ядра: какой файл за что отвечает, с чего читать код            |
| [src/pano3dbench/metrics/README.md](src/pano3dbench/metrics/README.md) | как по шагам считается каждый показатель                                  |
| [configs/igibson/dpc/README.md](configs/igibson/dpc/README.md)         | обучение голов формы и вывод на iGibson                                   |
| [tests/README.md](tests/README.md)                                     | что проверяет каждая папка тестов                                         |
| [configs/README.md](configs/README.md)                                 | за что отвечает каждая конфигурация                                       |
| [scripts/README.md](scripts/README.md)                                 | за что отвечает каждый скрипт и с какими аргументами он запускается       |
| [scripts/slurm/README.md](scripts/slurm/README.md)                     | шаблоны Slurm: обучение, вывод и подсчёт обоих наборов на кластере        |
| [runners/README.md](runners/README.md)                                 | договор раннера модели                                                    |
| [training/README.md](training/README.md)                               | обучение детектора и HorizonNet на 3D-FRONT                               |
| [configs/front3d/dpc/README.md](configs/front3d/dpc/README.md)         | обучение BEN, LDIF, MGN и Scene-GCN на 3D-FRONT                           |

## Состав репозитория

```text
pano3dbench/
├── configs/
│   ├── environments/        окружения моделей (dpc, echo)
│   ├── protocols/           общий протокол показателей для любого набора
│   ├── demo/                демонстрационные запуски без GPU и данных; scene/ — тестовая сцена
│   ├── igibson/
│   │   ├── protocols/       протокол контрольной строки на 11 классах
│   │   ├── runs/            <метод>.yaml через окружение, <метод>_rescore.yaml по готовым предсказаниям
│   │   └── dpc/             конфигурации самого DPC для iGibson
│   └── front3d/
│       ├── front3d_to_igibson.json  соответствие классов 3D-FRONT словарю iGibson
│       ├── dpc/             конфигурации DPC на 3D-FRONT: обучение компонент и вывод
│       ├── splits/          состав набора версии 1: train/val/test и исключённые комнаты
│       └── runs/            строки статьи по готовым предсказаниям
├── docs/                    документация
├── runners/                 раннеры моделей для изолированных окружений
├── scripts/
│   ├── dpc/                 перевод предсказаний DPC, правки кода DPC
│   ├── front3d/             форматы методов, списки видов для оценки и проверка набора 3D-FRONT
│   └── igibson/             эталон и цели MGN для iGibson
├── src/pano3dbench/         ядро
├── tests/                   тесты: core, runners, environments, front3d
├── training/                обучение детектора и HorizonNet на 3D-FRONT
└── pyproject.toml
```

Набор панорам 3D-FRONT собирается отдельным генератором
[front3d-pano](https://github.com/HoloPano-dissertation/front3d-pano).

## Лицензия

MIT, см. [LICENSE](LICENSE).
