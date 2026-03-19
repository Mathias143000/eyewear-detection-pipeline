# Eyewear Detection Pipeline

End-to-end ML/CV проект для задачи `glasses` / `no_glasses`:
`face detection -> face crop -> classifier -> API/CLI prediction`.

Проект ориентирован на портфолио Junior/Junior+ и покрывает полный цикл:
- подготовка данных и сплитов,
- baseline и нейросетевое обучение,
- оценка метрик и confusion matrix,
- подбор production threshold по `F1`,
- пакетный раннер экспериментов с итоговой таблицей,
- инференс на изображении и видео,
- FastAPI-сервис и Docker.

## Архитектура
1. Вход: изображение или видео.
2. Детекция лица: OpenCV Haar cascade.
3. Классификация кропа лица:
- `baseline` (логистическая регрессия на handcrafted фичах),
- `torch` (MobileNetV3 через `timm`/`torchvision` fallback).
4. Выход: `bbox`, `label`, `confidence`.

## Структура
```text
eyewear-detection-pipeline/
  src/eyewear_pipeline/
    api/main.py
    baseline.py
    config.py
    data.py
    evaluate.py
    face.py
    inference.py
    metrics.py
    predict_image.py
    predict_video.py
    torch_model.py
    train.py
  scripts/
    create_synthetic_dataset.py
    prepare_data.py
  tests/
  data/
  models/
  artifacts/
  reports/
```

## Быстрый старт
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
set PYTHONPATH=src
```

## 1) Подготовка данных
Вариант A: synthetic данные для быстрого MVP:
```bash
python scripts/create_synthetic_dataset.py --output-dir data/raw --samples-per-class 120
python scripts/prepare_data.py --dataset-root data/raw --manifest data/manifest.csv --splits-dir data/splits
```

Вариант B: свой датасет:
- положить изображения в `data/raw/glasses` и `data/raw/no_glasses`,
- затем запустить `scripts/prepare_data.py`.

Вариант C: CelebA (`Eyeglasses`) для боевого апгрейда:
```bash
python scripts/prepare_celeba_eyeglasses.py --images-dir D:\datasets\celeba\img_align_celeba --attr-file D:\datasets\celeba\list_attr_celeba.txt --manifest data/manifest_celeba.csv --splits-dir data/splits
```

Вариант D: CelebA через Hugging Face (без ручного скачивания):
```bash
python scripts/prepare_hf_celeba_eyeglasses.py --repo-id flwrlabs/celeba --split train --output-dir data/hf_celeba --manifest data/manifest_hf_celeba.csv --splits-dir data/splits_hf --max-samples-per-class 800
```

## 2) Обучение
Baseline (быстро и стабильно):
```bash
python -m eyewear_pipeline.train --model-type baseline --train-csv data/splits/train.csv
```

Torch-модель:
```bash
python -m eyewear_pipeline.train --model-type torch --train-csv data/splits/train.csv --val-csv data/splits/val.csv --epochs 5
```

## 3) Оценка
```bash
python -m eyewear_pipeline.evaluate --model-type baseline --test-csv data/splits/test.csv
```

Артефакты:
- `reports/metrics.json`
- `reports/confusion_matrix.png`

## 3.1) Калибровка порога
```bash
python scripts/calibrate_threshold.py --val-csv data/splits/val.csv --model-type baseline --output artifacts/threshold.json
```

Это позволяет валидно выбрать порог для production, вместо жесткого `0.5`.

## 3.2) Пакет экспериментов
```bash
python scripts/run_experiments.py --train-csv data/splits/train.csv --val-csv data/splits/val.csv --test-csv data/splits/test.csv --output-csv reports/experiment_table.csv
```

Результат: таблица `reports/experiment_table.csv` c метриками и статусом каждого эксперимента.

## Real-world run (пример)
На выборке HF CelebA (`1600` изображений, баланс 800/800):
- модель: `efficientnet_b0` (через `--model-name efficientnet_b0`);
- калиброванный порог: `0.66`;
- тестовые метрики:
  - `precision`: `0.9160`
  - `recall`: `0.9083`
  - `f1`: `0.9121`
  - `roc_auc`: `0.9835`

## 4) Инференс
Изображение:
```bash
python -m eyewear_pipeline.predict_image --image data/raw/glasses/glasses_0001.png --output artifacts/prediction.jpg --model-type baseline
```

Видео:
```bash
python -m eyewear_pipeline.predict_video --video demo/input.mp4 --output artifacts/prediction.mp4 --model-type baseline
```

## 5) API
```bash
uvicorn eyewear_pipeline.api.main:app --reload
```

Endpoints:
- `GET /health`
- `GET /model-info`
- `POST /predict/image` (multipart file)

API может автоматически подхватывать calibrated threshold:
- приоритет 1: `EYEWEAR_CONFIDENCE_THRESHOLD` (явное значение),
- приоритет 2: `EYEWEAR_THRESHOLD_FILE` (json),
- приоритет 3: `artifacts/threshold_torch_hf.json` для `torch`,
- fallback: `artifacts/threshold.json`.

## Docker
```bash
docker compose up --build
```

## Что показывать на собеседовании
- Почему выбран двухшаговый pipeline.
- Как выбирался threshold и какие ошибки (`FP/FN`) наиболее частые.
- Что улучшалось экспериментами (архитектура, эпохи, баланс классов, calibration).
- Как проект упакован для реального использования (CLI/API/Docker/tests).
- Готовый сценарий: `DEMO_SCRIPT.md`.

## Ограничения текущего MVP
- Haar face detector может ошибаться на сложных ракурсах.
- Synthetic данные подходят только для smoke-проверок.
- Для финальной версии портфолио важно запускать пайплайн на real-world данных (например, CelebA) и фиксировать bias/ограничения.
