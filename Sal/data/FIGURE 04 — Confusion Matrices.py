# FIGURE 04 — Confusion Matrices
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris  # Replace with your actual dataset
from sklearn.model_selection import train_test_split

# ── Load & split your data ──────────────────────────────────────────────────
# Replace this block with your actual data loading logic
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Train models ────────────────────────────────────────────────────────────
lr_model  = LogisticRegression(max_iter=1000).fit(X_train, y_train)
svm_model = SVC(kernel="rbf").fit(X_train, y_train)
dt_model  = DecisionTreeClassifier().fit(X_train, y_train)
mlp_model = MLPClassifier(max_iter=1000).fit(X_train, y_train)

# ── Plot confusion matrices ─────────────────────────────────────────────────
models = {
    "Logistic Regression": lr_model,
    "SVM (RBF)":           svm_model,
    "Decision Tree":       dt_model,
    "ANN (MLP)":           mlp_model,
}

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for ax, (name, model) in zip(axes.ravel(), models.items()):
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(name)

plt.tight_layout()
plt.savefig("04_confusion_matrices.png")
plt.show()