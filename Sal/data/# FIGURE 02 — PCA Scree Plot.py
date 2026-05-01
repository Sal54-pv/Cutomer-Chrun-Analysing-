# FIGURE 02 — PCA Scree Plot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris          # ← replace with your actual data
from sklearn.model_selection import train_test_split

# ── Load & scale data ───────────────────────────────────────────────────────
data = load_iris()                              # ← replace with your actual 33-column dataset
X = data.data
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# ── Scale — fit on train only, transform both (matches argument) ────────────
scaler   = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)      # no data leakage

# ── PCA — fit on train only (matches argument: "fitted exclusively on
#    the training partition") ────────────────────────────────────────────────
pca = PCA()
pca.fit(X_train_scaled)

# ── Cumulative explained variance ───────────────────────────────────────────
cum_var    = np.cumsum(pca.explained_variance_ratio_)
n_comp_90  = np.argmax(cum_var >= 0.90) + 1   # argument says 8
n_comp_95  = np.argmax(cum_var >= 0.95) + 1   # argument says 26

# ── Diagnostics — verify against argument ───────────────────────────────────
print(f"Components for 90% variance : {n_comp_90}")   # expect 8
print(f"Components for 95% variance : {n_comp_95}")   # expect 26
print(f"Total features (columns)    : {X.shape[1]}")  # expect 33

# ── Build 26-component PCA input space for future experiments ───────────────
pca_26          = PCA(n_components=n_comp_95)
X_train_pca     = pca_26.fit_transform(X_train_scaled)
X_test_pca      = pca_26.transform(X_test_scaled)     # no data leakage

# ── Plot ─────────────────────────────────────────────────────────────────────
plt.figure(figsize=(9, 5))
plt.plot(range(1, len(cum_var) + 1), cum_var,
         marker='o', markersize=4, linewidth=1.5, color="steelblue")

# 90% threshold
plt.axhline(y=0.90, linestyle='--', color='orange', linewidth=1.2)
plt.axvline(x=n_comp_90, linestyle=':', color='orange', linewidth=1.2)
plt.annotate(f"90% — {n_comp_90} components",
             xy=(n_comp_90, 0.90),
             xytext=(n_comp_90 + 0.5, 0.87),
             color='orange', fontsize=9)

# 95% threshold
plt.axhline(y=0.95, linestyle='--', color='red', linewidth=1.2)
plt.axvline(x=n_comp_95, linestyle=':', color='red', linewidth=1.2)
plt.annotate(f"95% — {n_comp_95} components",
             xy=(n_comp_95, 0.95),
             xytext=(n_comp_95 + 0.5, 0.92),
             color='red', fontsize=9)

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Scree Plot (fitted on training set only)")
plt.xlim(1, len(cum_var))
plt.ylim(0, 1.02)
plt.tight_layout()
plt.savefig("02_pca_scree.png", dpi=150)
plt.show()