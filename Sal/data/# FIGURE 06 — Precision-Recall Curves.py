# FIGURE 06 — Precision-Recall Curves
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris          # ← replace with your actual data
from sklearn.model_selection import train_test_split

# ── Load & split data ───────────────────────────────────────────────────────
data = load_iris()                              # ← replace with your actual data
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Train models ────────────────────────────────────────────────────────────
lr_model  = LogisticRegression(max_iter=1000).fit(X_train, y_train)
svm_model = SVC(kernel="rbf", probability=True).fit(X_train, y_train)
dt_model  = DecisionTreeClassifier().fit(X_train, y_train)
mlp_model = MLPClassifier(max_iter=1000).fit(X_train, y_train)

models = {
    "Logistic Regression": lr_model,
    "SVM (RBF)":           svm_model,
    "Decision Tree":       dt_model,
    "ANN (MLP)":           mlp_model,
}

# ── Precision-Recall Curves ─────────────────────────────────────────────────
classes    = np.unique(y_test)
n_classes  = len(classes)
y_test_bin = label_binarize(y_test, classes=classes)

plt.figure(figsize=(8, 6))

for name, model in models.items():
    y_prob = model.predict_proba(X_test)

    if n_classes == 2:
        precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
        ap = average_precision_score(y_test, y_prob[:, 1])
        plt.plot(recall, precision, label=f"{name} (AP = {ap:.2f})")
    else:
        # Multiclass — macro-average OvR
        precision_all, recall_all = [], []
        for i in range(n_classes):
            p, r, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
            precision_all.append(p)
            recall_all.append(r)

        mean_recall    = np.linspace(0, 1, 100)
        mean_precision = np.mean(
            [np.interp(mean_recall, np.flip(r), np.flip(p))
             for p, r in zip(precision_all, recall_all)], axis=0
        )
        ap = average_precision_score(y_test_bin, y_prob, average="macro")
        plt.plot(mean_recall, mean_precision, label=f"{name} (AP = {ap:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("06_pr_curves.png")
plt.show()