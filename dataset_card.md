# Dataset Card: Eyewear Classification

## Task
Бинарная классификация лица:
- `0: no_glasses`
- `1: glasses`

## Рекомендованные источники real-world данных
- CelebA (атрибут `Eyeglasses`) с корректной фильтрацией и валидацией лицензии.
- MAFA / WiderFace + собственная разметка по задаче очков.
- Внутренний датасет (при наличии прав на использование изображений).

## Текущая реализация в репозитории
- Для smoke-тестов используется synthetic dataset (`scripts/create_synthetic_dataset.py`).
- Для real-world сценария добавлен адаптер CelebA (`scripts/prepare_celeba_eyeglasses.py`).
- Для автоматического real-world прогона без ручного скачивания добавлен HF адаптер (`scripts/prepare_hf_celeba_eyeglasses.py`).
- Формат хранения:
  - `data/raw/glasses/*.png`
  - `data/raw/no_glasses/*.png`
- Manifest:
  - `data/manifest.csv` с колонками `image_path,label`.
- Сплиты:
  - `data/splits/train.csv`
  - `data/splits/val.csv`
  - `data/splits/test.csv`

## Риски и ограничения
- Synthetic dataset не отражает real-world вариативность.
- Возможный class imbalance при сборе реальных данных.
- Потенциальный bias по демографии, освещению, типу камеры.
- Артефакты сжатия и качество разметки влияют на метрики.

## Mitigation
- Стратифицированный split.
- Отдельный error analysis по подгруппам условий (освещение, ракурс, окклюзия).
- Ручная проверка шумных примеров.
- Консервативная интерпретация метрик и честное описание ограничений.
- Подбор production threshold по валидационной выборке (`scripts/calibrate_threshold.py`).

## Последний реальный прогон (2026-03-14)
- Источник: `flwrlabs/celeba` (Hugging Face).
- Размер: 1600 (800/800 по классам).
- Модель: EfficientNet-B0.
- Threshold: 0.66 (калибровка по validation split).
- Тест:
  - precision: 0.9160
  - recall: 0.9083
  - f1: 0.9121
  - roc_auc: 0.9835
