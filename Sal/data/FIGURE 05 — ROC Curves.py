# FIGURE 05 — ROC Curves
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
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

# ── ROC Curves ──────────────────────────────────────────────────────────────
classes    = np.unique(y_test)
n_classes  = len(classes)
y_test_bin = label_binarize(y_test, classes=classes)

plt.figure(figsize=(8, 6))

for name, model in models.items():
    y_prob = model.predict_proba(X_test)

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    else:
        fpr_all, tpr_all = [], []
        for i in range(n_classes):
            fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            fpr_all.append(fpr_i)
            tpr_all.append(tpr_i)

        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(mean_fpr, f, t) for f, t in zip(fpr_all, tpr_all)], axis=0)
        roc_auc  = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("05_roc_curves.png")
plt.show()