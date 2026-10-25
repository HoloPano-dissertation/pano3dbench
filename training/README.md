# Обучение на наборе панорам из 3D-FRONT

Эта папка содержит только запуск детектора и HorizonNet и необходимые им
модули загрузки/проверки. Набор, разметку и COCO готовит генератор
[front3d-pano](https://github.com/HoloPano-dissertation/front3d-pano), цели HorizonNet —
[scripts/front3d/adapters/export_horizonnet.py](../scripts/front3d/README.md);
генератор от этих файлов не зависит.

Нужно отдельное GPU-окружение проекта Pano3D: PyTorch, совместимые CUDA/cuDNN,
Detectron2 и собранные CUDA-операции. Для HorizonNet корень Pano3D должен быть
в `PYTHONPATH`, чтобы импортировался `external.HorizonNet`.
Зависимости рендерера не устанавливают это окружение.

## Проверка и запуск

```bash
python training/gpu_preflight.py --detectron

python training/train_detector.py \
  /data/front3d-experiment/coco /data/front3d-experiment/training/detector \
  --weights /models/mask_rcnn_R_50_FPN_3x.pkl

PYTHONPATH=/path/to/Pano3D python training/train_horizonnet.py \
  /data/front3d-experiment/horizonnet /data/front3d-experiment/training/horizonnet \
  --label-format dense
```

Используйте согласованный COCO checkpoint для детектора, а не веса, обученные на другом наборе.
HorizonNet инициализирует encoder весами ImageNet, decoder обучает заново.
Первый запуск может потребовать скачивания ImageNet-весов.

Детектор и HorizonNet — отдельные обучения, одно не запускает другое.
BEN, shape heads и Scene-GCN этими двумя командами не переобучаются.

Полный запуск требует отдельного одобрения качества в
`EXPERIMENT/state/training_gate.json` (`training_approved: true` с обоснованием).
Не выставляйте его только потому, что экспорт завершился без ошибок.
Если после одобрения изменились `splits/rooms.jsonl` или отчёты этапов генератора в `state/`,
ворота снова закрыты: набор надо проверить заново и обновить файл.
Сначала проверьте данные, решите отмеченные проблемы и пройдите GPU preflight.

## Короткая техническая проба до одобрения

```bash
python training/train_detector.py \
  /data/front3d-experiment/coco /data/front3d-experiment/training/detector-smoke \
  --weights /models/mask_rcnn_R_50_FPN_3x.pkl \
  --max-iter 2 --eval-period 0 --workers 0 --allow-unapproved-smoke

PYTHONPATH=/path/to/Pano3D python training/train_horizonnet.py \
  /data/front3d-experiment/horizonnet /data/front3d-experiment/training/layout-smoke \
  --epochs 1 --max-train-batches 2 --max-val-batches 1 \
  --workers 0 --allow-unapproved-smoke
```

Smoke проверяет загрузку/вычисления; не даёт разрешения на полное обучение.
Тестовая часть не используется для подбора checkpoint.
`gpu_preflight.py` останавливает запуск при несовместимости архитектуры GPU
со сборкой PyTorch — новый GPU не всегда подходит старому окружению.
