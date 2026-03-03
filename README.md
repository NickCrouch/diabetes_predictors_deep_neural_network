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

### Example (CSV)
```csv
age,bmi,glucose,blood_pressure,insulin,diabetes
45,31.2,155,72,130,1
29,24.8,95,66,85,0
```

Data for this analysis was taken from the CDC's 2024 Behavioral Risk Factor Surveillance System (BRFSS) collection. 

# Training output
Epoch 10/150 | loss=1.1334 | val PR AUC=0.2989 \
Epoch 20/150 | loss=1.0896 | val PR AUC=0.3246 \
Epoch 30/150 | loss=1.0466 | val PR AUC=0.3384 \
Epoch 40/150 | loss=1.0109 | val PR AUC=0.3492 \
Epoch 50/150 | loss=0.9872 | val PR AUC=0.3534 \
Epoch 60/150 | loss=0.9744 | val PR AUC=0.3519 \
Epoch 70/150 | loss=0.9677 | val PR AUC=0.3504 \
Epoch 80/150 | loss=0.9638 | val PR AUC=0.3513 \
Epoch 90/150 | loss=0.9601 | val PR AUC=0.3530 \
Epoch 100/150 | loss=0.9562 | val PR AUC=0.3546 \
Epoch 110/150 | loss=0.9551 | val PR AUC=0.3559 \
Epoch 120/150 | loss=0.9537 | val PR AUC=0.3571 \
Epoch 130/150 | loss=0.9512 | val PR AUC=0.3584 \
Epoch 140/150 | loss=0.9501 | val PR AUC=0.3596 \
Epoch 150/150 | loss=0.9505 | val PR AUC=0.3605 \
Best val PR AUC: 0.36049848754738567 \
Training finished.


<img width="500" height="350" alt="loss_training" src="https://github.com/user-attachments/assets/add73e44-0910-428f-abe9-5f7a710aa32a" />


ROC AUC: 0.79 \
PR AUC: 0.36 \
Performing permutation importance \
Baseline AUC: 0.79 \
Feature 0: ΔAUC = 0.1634 ± 0.0029 \
Feature 2: ΔAUC = 0.0669 ± 0.0016 \
Feature 5: ΔAUC = 0.0184 ± 0.0007 \
Feature 4: ΔAUC = 0.0102 ± 0.0004 \
Feature 1: ΔAUC = 0.0068 ± 0.0006 \
Feature 6: ΔAUC = 0.0021 ± 0.0004 \
Feature 3: ΔAUC = 0.0005 ± 0.0002


The model was trained as a feedforward neural network with two hidden layers (64 and 32 units) using ReLU activations and dropout regularization. To address class imbalance (positive prevalence ≈15%), the loss function was specified as BCEWithLogitsLoss with a class weighting term (pos_weight = 5.63) computed from the training data only. The dataset was divided into training, validation, and test sets using stratified sampling to preserve class proportions. Feature scaling was performed using a StandardScaler fit exclusively on the training data to avoid data leakage. Model selection was conducted using the validation set: the optimal training epoch was chosen based on validation PR AUC, and the classification threshold was tuned to maximize the F1 score rather than relying on the default 0.5 cutoff. Final model performance was evaluated once on the held-out test set.

On the test data, the model achieved a ROC AUC of 0.79 and a PR AUC of 0.36. The ROC AUC (Area Under the Receiver Operating Characteristic Curve) measures the model’s ability to discriminate between positive and negative cases across all possible decision thresholds; it can be interpreted as the probability that a randomly selected positive instance is ranked higher than a randomly selected negative instance. A value of 0.78 indicates good, though not perfect, discriminative ability. The PR AUC (Area Under the Precision–Recall Curve) focuses specifically on performance for the positive class and is particularly informative under class imbalance. Because the baseline PR AUC equals the positive class prevalence (≈0.15 in this dataset), a PR AUC of 0.36 represents approximately a 2.4-fold improvement over chance. At the validation-optimized decision threshold (≈0.60), the model achieves high precision for the negative class (0.93) and moderate precision for the positive class (0.32), reflecting meaningful but incomplete identification of minority-class cases.

Overall, the model demonstrates solid ranking performance and measurable predictive lift over baseline, though performance for the minority class remains constrained. Future improvements could focus on additional feature engineering, systematic tuning of class weighting and regularization parameters, exploration of alternative architectures or tree-based models, or probability calibration techniques to improve threshold stability and interpretability.
