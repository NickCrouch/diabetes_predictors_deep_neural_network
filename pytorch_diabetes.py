import pandas as pd
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score, f1_score
from plotnine import ggplot, aes, geom_line, theme_bw, ggtitle
import numpy as np


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1. Data Loading and Preprocessing ---
def load_and_preprocess_data(csv_file_path):
    # Load data using pandas
    data = pd.read_csv(csv_file_path)
    
    # Assuming the last column is the target (binary outcome)
    X = data.iloc[:, :-1].values # Features (first 7 columns for an 8-col total)
    y = data.iloc[:, -1].values  # Target (last column)
    
    # Split into training and testing sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Split 2: train vs val (val_size is fraction of temp)
    val_size=0.2
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )

    # Fit scaler on TRAIN ONLY (no leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val   = torch.tensor(X_val, dtype=torch.float32)
    y_val   = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    y_test  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

# --- 2. Model Definition ---
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),          # First Linear layer
            nn.ReLU(),                         # First ReLU activation
            nn.Dropout(0.2),                   # First Dropout layer (e.g., 20% dropout rate)
            nn.Linear(64, 32),                 # Second Linear layer
            nn.ReLU(),                         # Second ReLU activation
            nn.Dropout(0.2),                   # Second Dropout layer
            nn.Linear(32, 1),                  # Output Linear layer
            # nn.Sigmoid()                       # Sigmoid activation for binary outcome
        )

    def forward(self, x):
        return self.model(x)

# --- 3. Training Functions ---
def pr_auc_from_logits(logits, y_true):
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    y_np  = y_true.squeeze().cpu().numpy()
    return average_precision_score(y_np, probs)

def train_model(model, X_train, y_train, X_val, y_val, criterion, optimizer, num_epochs=100):
    losses = []
    best_val_pr = -1.0
    best_state = None

    for epoch in range(num_epochs):
        # ---- TRAIN ----
        model.train()
        logits = model(X_train)
        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # ---- VALIDATION (no gradients) ----
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)                 # ### <-- validation used here
            val_pr = pr_auc_from_logits(val_logits, y_val)  # ### <-- and here

        # Keep best model by validation PR AUC
        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_state = copy.deepcopy(model.state_dict())  # ### <-- best checkpoint based on VAL

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | loss={loss.item():.4f} | val PR AUC={val_pr:.4f}")

    # Restore best model before returning
    if best_state is not None:
        model.load_state_dict(best_state)  # ### <-- best epoch chosen by validation

    return losses, best_val_pr

# --- 4. Evaluation Functions ---
def find_best_threshold_f1(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_val)).squeeze().cpu().numpy()
    y = y_val.squeeze().cpu().numpy()

    thresholds = np.linspace(0.01, 0.99, 200)
    f1s = [f1_score(y, probs > t) for t in thresholds]
    best_t = thresholds[int(np.argmax(f1s))]
    return best_t, max(f1s)

def evaluate_model(model, X_test, y_test):
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation for evaluation
        outputs = model(X_test)
        predicted = (outputs > 0.5).float() # Threshold at 0.5 for binary classification
        
        accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
        print(f'\nAccuracy on test set: {accuracy:.4f}')
        print('Classification Report on test set:')
        print(classification_report(y_test.numpy(), predicted.numpy()))

        probs_test = torch.sigmoid(model(X_test)).squeeze().cpu().numpy()
    y_test_np = y_test.squeeze().cpu().numpy()
    print("ROC AUC:", roc_auc_score(y_test_np, probs_test))
    print("PR AUC:", average_precision_score(y_test_np, probs_test))


def evaluate_on_test(model, X_test, y_test, threshold):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    y = y_test.squeeze().cpu().numpy()
    preds = (probs > threshold).astype(int)

    print("ROC AUC:", roc_auc_score(y, probs))
    print("PR AUC:", average_precision_score(y, probs))
    print("Threshold:", threshold)
    print(classification_report(y, preds))

# --- 5. ROC AUC permutation importance. ---
def permutation_importance_auc(model, X, y, n_repeats=10, seed=42):
    model.eval()
    y_np = y.detach().cpu().numpy().reshape(-1)

    with torch.no_grad():
        base_probs = torch.sigmoid(model(X)).squeeze().cpu().numpy().reshape(-1)
    baseline = roc_auc_score(y_np, base_probs)

    rng = np.random.default_rng(seed)
    importances = np.zeros((X.shape[1], n_repeats), dtype=float)

    for j in range(X.shape[1]):
        for r in range(n_repeats):
            Xp = X.clone()
            idx = rng.permutation(X.shape[0])
            Xp[:, j] = Xp[idx, j]

            with torch.no_grad():
                probs_p = torch.sigmoid(model(Xp)).squeeze().cpu().numpy().reshape(-1)

            score_p = roc_auc_score(y_np, probs_p)
            importances[j, r] = baseline - score_p

    return importances.mean(axis=1), importances.std(axis=1), baseline

# --- Main Execution ---
if __name__ == '__main__':
    CSV_FILE = 'diabetes_data_2024_brfss.csv' 
    INPUT_SIZE = 7 # 8 columns total, last is target
    NUM_EPOCHS = 150

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, scale = load_and_preprocess_data(CSV_FILE)

    # Initialize model, loss, and optimizer
    model = BinaryClassifier(input_size=X_train.shape[1])

    # Loss with pos_weight (computed on TRAIN ONLY)
    pos = y_train.sum()
    neg = (y_train == 0).sum()
    pos_weight = (neg / pos).float()  # > 1 when positives are rarer
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimizer

    # Train the model
    print("Starting training...")
    # 4) Train, selecting best epoch by VAL PR AUC   ### <-- validation affects selection
    losses, best_val_pr = train_model(model, X_train, y_train, X_val, y_val, criterion, optimizer, num_epochs=NUM_EPOCHS)
    print("Best val PR AUC:", best_val_pr)

    # Show training loss
    dat = pd.DataFrame(losses, columns=['Loss'])
    dat['Epoch'] = range(0, NUM_EPOCHS)
    p = (ggplot(dat, aes(x='Epoch', y='Loss')) + geom_line() + theme_bw() + ggtitle("Training Loss"))
    p.save("loss_training.png", width=6, height=4, dpi=300, verbose=False)

    print("Training finished.")

    best_t, best_val_f1 = find_best_threshold_f1(model, X_val, y_val)
    print("Best val threshold:", best_t, "Best val F1:", best_val_f1)

    evaluate_model(model, X_test, y_test)

    print("Performing permutation importance")
    # permutation importance (on test or validation set)
    mean_imp, std_imp, base_auc = permutation_importance_auc(model, X_test, y_test, n_repeats=10)

    order = np.argsort(-mean_imp)
    print("Baseline AUC:", base_auc)
    for k in order:
        print(f"Feature {k}: ΔAUC = {mean_imp[k]:.4f} ± {std_imp[k]:.4f}")

    evaluate_on_test(model, X_test, y_test, threshold=best_t)
