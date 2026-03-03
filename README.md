# diabetes_predictors_deep_neural_network
PyTorch-based neural network for predicting diabetes from clinical features. Includes preprocessing, training, evaluation. Designed for reproducible experiments on tabular health data; for research use only, not clinical decision-making.

A small, practical PyTorch project for training a neural network to **predict diabetes risk** from tabular clinical features and to **evaluate which predictors matter** using common ML diagnostics.

> **Disclaimer**
> This project is for educational/research purposes only and is **not** medical advice or a clinical device. Model outputs should not be used to make medical decisions.

---

## Project Goals

- Train a PyTorch neural network on tabular data to predict diabetes (binary classification).
- Evaluate performance with metrics suited to imbalanced clinical datasets.
- Interpret predictors using feature importance methods (permutation importance, SHAP, etc.).
- Provide a reproducible pipeline: data prep → training → evaluation → interpretation.

## Expected Data Format

Input data should be in a CSV (or Parquet) with:
- One row per patient/visit
- Columns = predictors (features)
- A binary target label column such as `diabetes` (0/1)

Example columns (you can adapt):
- `age`, `bmi`, `glucose`, `insulin`, `blood_pressure`, `pregnancies`, `skin_thickness`, `dpf` (diabetes pedigree function), etc.

### Example (CSV)
```csv
age,bmi,glucose,blood_pressure,insulin,diabetes
45,31.2,155,72,130,1
29,24.8,95,66,85,0
...
