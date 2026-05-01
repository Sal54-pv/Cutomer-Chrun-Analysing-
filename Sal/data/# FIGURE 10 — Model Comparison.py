# FIGURE 10 — Model Comparison
import pandas as pd
import matplotlib.pyplot as plt

# ── Data ────────────────────────────────────────────────────────────────────
results = pd.DataFrame({
    "Model":   ["LR", "SVM", "DT", "ANN", "NB"],
    "Recall":  [0.633, 0.011, 0.758, 0.109, 0.250],
    "F1":      [0.402, 0.021, 0.398, 0.175, 0.327],
    "ROC-AUC": [0.721, 0.710, 0.708, 0.711, 0.704]
})

# ── Plot ─────────────────────────────────────────────────────────────────────
results.set_index("Model").plot(kind="bar", figsize=(8, 5))
plt.title("Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("10_model_comparison.png")
plt.show()