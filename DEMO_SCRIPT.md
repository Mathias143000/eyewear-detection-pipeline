# Demo Script (2 Minutes)

Цель: быстро показать на собеседовании, что проект рабочий как ML pipeline и как сервис.

## 0) Подготовка (до звонка)
1. Активировать окружение и убедиться, что есть модель и threshold:
```bash
set PYTHONPATH=src
python -m eyewear_pipeline.evaluate --model-type torch --model-path models/torch_efficientnet_b0_e8.pt --test-csv data/splits_hf/test.csv --threshold 0.66
```
2. Проверить, что есть:
- `models/torch_efficientnet_b0_e8.pt`
- `artifacts/threshold_torch_hf.json`
- `reports/experiment_table_hf.csv`

## 1) Показ качества (30-40 сек)
Показать:
- `reports/experiment_table_hf.csv`
- `reports/metrics.json`
- `reports/confusion_matrix.png`

Коротко сказать:
- сравнили baseline и torch-модели;
- выбрали лучшую (`efficientnet_b0`);
- откалибровали порог по валидации;
- получили целевые метрики на test.

## 2) Показ инференса на изображении (30 сек)
```bash
python -m eyewear_pipeline.predict_image --image data/hf_celeba/glasses/g_000010.jpg --output artifacts/demo_prediction.jpg --model-type torch --model-path models/torch_efficientnet_b0_e8.pt
```
Показать `artifacts/demo_prediction.jpg` и JSON с `bbox`, `label_name`, `confidence`.

## 3) Показ API (40-50 сек)
Запуск:
```bash
set EYEWEAR_MODEL_TYPE=torch
set EYEWEAR_MODEL_PATH=models/torch_efficientnet_b0_e8.pt
set EYEWEAR_THRESHOLD_FILE=artifacts/threshold_torch_hf.json
uvicorn eyewear_pipeline.api.main:app --host 0.0.0.0 --port 8000
```

В другом терминале:
```bash
curl http://127.0.0.1:8000/model-info
curl -X POST "http://127.0.0.1:8000/predict/image" -F "file=@data/hf_celeba/glasses/g_000010.jpg"
```

## 4) Что проговорить в конце (10 сек)
- проект воспроизводим (скрипты данных, train, eval, experiments);
- есть отдельный baseline и error analysis;
- есть путь до продового API и Docker.
