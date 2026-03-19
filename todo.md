# Eyewear Detection Pipeline - Execution Status

## Scope
Проект реализован как end-to-end MVP для портфолио Junior/Junior+:
- задача `glasses/no_glasses`,
- face detection + classification,
- train/eval/infer,
- FastAPI + Docker + tests + docs.

## Выполнено

### 1. Задача и критерии
- [x] Зафиксирован сценарий `glasses/no_glasses`.
- [x] Зафиксированы ключевые метрики `precision/recall/F1/ROC-AUC`.
- [x] Зафиксирован минимальный результат: reproducible pipeline + API + отчеты.
- [x] MVP ограничен двухшаговым pipeline без лишнего усложнения.

### 2. Данные
- [x] Добавлен формат датасета `data/raw/glasses|no_glasses`.
- [x] Добавлен генератор synthetic-датасета (`scripts/create_synthetic_dataset.py`) для быстрых проверок.
- [x] Добавлены manifest и split инструменты (`scripts/prepare_data.py`).
- [x] Добавлен `dataset_card.md`.

### 3. Baseline и обучение
- [x] Реализован baseline (логистическая регрессия на handcrafted features).
- [x] Реализовано обучение torch-модели (MobileNetV3).
- [x] Реализован единый train entrypoint.
- [x] Добавлена сериализация моделей и train report.

### 4. Эксперименты и оценка
- [x] Реализован evaluation script с метриками.
- [x] Добавлено построение confusion matrix.
- [x] Добавлены шаблоны для error analysis.
- [x] Проведен real-world прогон на HF CelebA с калибровкой порога и сохранением метрик.

### 5. Инференс
- [x] Реализован face detector (OpenCV Haar).
- [x] Реализован inference для изображения.
- [x] Реализован inference для видео.
- [x] Поддержан CPU-only сценарий.

### 6. API и демо
- [x] Реализован FastAPI сервис:
  - [x] `GET /health`
  - [x] `GET /model-info`
  - [x] `POST /predict/image`
- [x] Добавлены переменные окружения для модели.
- [x] Добавлены Dockerfile и docker-compose.

### 7. Инженерная дисциплина
- [x] Добавлены `requirements.txt` и `requirements-dev.txt`.
- [x] Добавлен `pyproject.toml` и конфигурация pytest.
- [x] Добавлены тесты на data/inference/api.
- [x] Подготовлена структура проекта под рост.

### 8. Портфельная упаковка
- [x] Написан `README.md` с полным runbook.
- [x] Добавлен `INTERVIEW_PREP.md`.
- [x] Добавлен `error_analysis.md`.
- [x] Описаны ограничения и roadmap.

## Post-MVP backlog
- [x] Добавить поддержку real-world публичного датасета (CelebA adapter script + split pipeline).
- [x] Добавить раннер для 3+ экспериментов с выгрузкой таблицы результатов (`reports/experiment_table.csv`).
- [x] Добавить калибровку threshold по валидационной выборке.
- [x] Подготовить готовый собеседный сценарий demo (2 минуты, CLI + API + метрики).
- [x] Добавить автоподхват calibrated threshold в API.
- [ ] Расширить на класс `sunglasses`.
- [ ] Добавить ONNX export и benchmark latency/FPS.
- [ ] Добавить visual demo (streamlit/gradio или web UI).
