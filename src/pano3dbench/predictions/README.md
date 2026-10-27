# `predictions/` — источники предсказаний

Источник предсказаний получает `SceneSample` и возвращает `SceneOutput`. Это не модель:
модель работает в своём окружении, а источник либо читает её готовый ответ, либо
запускает её раннер.

| Файл                    | За что отвечает                                                                                                                       |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `base.py`               | `PredictionSource` (`predict`, `predict_many`, `provenance`) и реестр `PREDICTION_SOURCES`                                            |
| `saved_predictions.py`  | `SavedPredictionSource`: читает `<prediction_dir>/<sample_id>.json`; в конфигурации `fixture`                                         |
| `runner_predictions.py` | `RunnerPredictionSource`: отдаёт сцены внешнему раннеру и собирает ответы; в конфигурации `subprocess` и `dpc`                        |
| `runner_process.py`     | как запускается раннер: команда из спецификации окружения или из `command`, запуск процесса, `request.json` / `output.json`, тайм-аут |

`RunnerPredictionSource` работает двумя способами:

- `environment: <спецификация>` — команда собирается из окружения (`env prepare` должен быть
  выполнен заранее), `runner_args` дописываются после аргументов спецификации;
- `command: [...]` — готовая команда с подстановками `{request}` и `{output}`.

Если раннер пакетный (`runner.batch: true` в спецификации или `batch: true`), все сцены
уходят одним запуском, а ответы сопоставляются по `sample_id`. Договор раннера —
[runners/README.md](../../../runners/README.md).
