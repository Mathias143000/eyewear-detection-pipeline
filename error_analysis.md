# Error Analysis Template

Этот файл заполняется после запуска `python -m eyewear_pipeline.evaluate`.

## Summary
- Model type:
- Dataset version:
- Test samples:
- Precision:
- Recall:
- F1:
- ROC-AUC:

## FP Cases (No glasses -> Predicted glasses)
1. Причина:
2. Причина:
3. Причина:

## FN Cases (Glasses -> Predicted no_glasses)
1. Причина:
2. Причина:
3. Причина:

## Observed failure modes
- Сильные блики на линзах.
- Боковой ракурс лица.
- Частичная окклюзия (волосы, рука, маска).
- Низкая резкость / смаз.

## Actions
1. Добавить hard negative примеры.
2. Увеличить долю примеров с профильными ракурсами.
3. Подобрать threshold по PR-кривой.
4. Проверить более сильный face detector.

