import numpy as np
import pandas as pd
# import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------
# Load and preprocess data
# -------------------------
# data = pd.read_excel(r"C:\Users\hp\Desktop\VS Code Projects\VS_AllCodes\Scratch - Perceptron\student_maths.xlsb")
data = pd.read_csv(r"C:\Users\hp\Desktop\VS Code Projects\VS_AllCodes\IPL Predication Code-\Scratch_Decision_Tree_\california_housing_test_1.csv", sep=",")

# X: use columns 26:29 (3 features); y: column 11 (make sure it's 0/1 for classification)
# X = data.iloc[:, 26:29].values.astype(float)   # shape (N, 3)
# y = data.iloc[:, 11:12].values.astype(float)   # shape (N, 1)

X = data.iloc[:,0:-1].values.astype(float)
y = data.iloc[:,-1].values.astype(float)

X = np.asarray(X)
y = np.asarray(y)

# Standardize per feature: (x - mean) / std
# X_mean = X.mean(axis=0, keepdims=True)
# X_std  = X.std(axis=0, keepdims=True) + 1e-8   # avoid div-by-zero
# X = (X - X_mean) / X_std

# Ensure y is in {0,1}. If it's not, you must map it.
# Example: if it's marks, this won't be classification. For now assuming it's already 0/1.
# If needed: y = (y > threshold).astype(float)

# -------------------------
# MLP with ReLU hidden, Sigmoid output
# BCE Loss (stable and simple gradients)
# -------------------------
class MLP:
    def __init__(self, in_features, hidden_units=4, out_features=1, seed=42):
        in_features = X.shape[1]
        rng = np.random.default_rng(seed)
        self.w1 = rng.standard_normal((in_features, hidden_units)) * 0.5
        self.b1 = np.zeros((1, hidden_units))
        self.w2 = rng.standard_normal((hidden_units, out_features)) * 0.5
        self.b2 = np.zeros((1, out_features))
        # gradient buffers
        self.gw1 = np.zeros_like(self.w1)
        self.gb1 = np.zeros_like(self.b1)
        self.gw2 = np.zeros_like(self.w2)
        self.gb2 = np.zeros_like(self.b2)
        self.cache = {}

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def drelu(z):
        return (z > 0).astype(z.dtype)

    @staticmethod
    def sigmoid(z):
        # clip for numerical stability
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def bce_loss(y, y_hat, eps=1e-10):
        """
        Binary cross-entropy: mean( -[ y*log(y_hat) + (1-y)*log(1-y_hat) ] )
        y, y_hat shape (N,1). y in {0,1}
        """
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return np.mean(- (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

    def forward(self, X):
        z1 = X * self.w1 + self.b1   # (N, H)
        h  = self.relu(z1)           # (N, H)
        z2 = h * self.w2 + self.b2   # (N, 1)
        y_hat = self.sigmoid(z2)     # (N, 1)
        self.cache = {"X": X, "z1": z1, "h": h, "z2": z2, "y_hat": y_hat}
        return y_hat

    def backward(self, y):
        """
        With sigmoid + BCE, dL/dz2 = (y_hat - y)
        """
        X   = self.cache["X"]
        z1  = self.cache["z1"]
        h   = self.cache["h"]
        y_hat = self.cache["y_hat"]

        # output layer grad
        dL_dz2 = (y_hat - y)                     # (N,1)

        self.gw2 = h.T * dL_dz2                  # (H,1)
        self.gb2 = np.sum(dL_dz2, axis=0, keepdims=True)  # (1,1)

        # backprop to hidden
        dL_dh = dL_dz2 * self.w2.T              # (N,H)
        dh_dz1 = self.drelu(z1)                 # (N,H)
        dL_dz1 = dL_dh * dh_dz1                 # (N,H)

        self.gw1 = X.T * dL_dz1                 # (in,H)
        self.gb1 = np.sum(dL_dz1, axis=0, keepdims=True)  # (1,H)

    def step(self, lr=0.01):
        self.w2 -= lr * self.gw2
        self.b2 -= lr * self.gb2
        self.w1 -= lr * self.gw1
        self.b1 -= lr * self.gb1

    def fit(self, X, y, epochs=2000, lr=0.01, verbose_every=200):
        for ep in range(epochs):
            y_hat = self.forward(X)
            loss = self.bce_loss(y, y_hat)
            self.backward(y)
            self.step(lr)
            if verbose_every and (ep % verbose_every == 0):
                print(f"Epoch {ep:4d} | BCE Loss: {loss:.6f}")

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int), proba

# -------------------------
# Metrics helper (fixed)
# -------------------------
class Measures:
    def confusion(self, y_true, y_pred):
        """
        y_true, y_pred: arrays of shape (N,1) or (N,)
        Returns TP, TN, FP, FN
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        TP = np.sum((y_pred == 1) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        return TP, TN, FP, FN

    def accuracy(self, y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        return np.mean(y_true == y_pred)

    def precision(self, y_true, y_pred, eps=1e-12):
        TP, TN, FP, FN = self.confusion(y_true, y_pred)
        return TP / (TP + FP + eps)

    def recall(self, y_true, y_pred, eps=1e-12):
        TP, TN, FP, FN = self.confusion(y_true, y_pred)
        return TP / (TP + FN + eps)

    def f1(self, y_true, y_pred, eps=1e-12):
        p = self.precision(y_true, y_pred, eps)
        r = self.recall(y_true, y_pred, eps)
        return 2 * p * r / (p + r + eps)

# -------------------------
# Train & evaluate
# -------------------------
in_features = X.shape[1]
mlp = MLP(in_features=in_features, hidden_units=4, out_features=1, seed=7)

# IMPORTANT: y must be (N,1) and in {0,1} for BCE
mlp.fit(X, y, epochs=1000, lr=0.001, verbose_every=200)

preds, probs = mlp.predict(X, threshold=0.5)

m = Measures()
TP, TN, FP, FN = m.confusion(y, preds)
acc = m.accuracy(y, preds)
prec = m.precision(y, preds)
rec  = m.recall(y, preds)
f1   = m.f1(y, preds)

print("\n--- Results on training data ---")
print("Confusion Matrix (TP, TN, FP, FN):", TP, TN, FP, FN)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

print("X (first 10):", X[:10].ravel())
print("\nSample probs (first 10):", np.round(probs[:10].ravel(), 4))
print("Sample preds (first 10):", preds[:10].ravel())