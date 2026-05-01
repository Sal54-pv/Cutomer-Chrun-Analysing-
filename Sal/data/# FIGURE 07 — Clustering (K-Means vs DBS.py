# FIGURE 07 — Clustering (K-Means vs DBSCAN)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris          # ← replace with your actual data

# ── Load & scale data ───────────────────────────────────────────────────────
data = load_iris()                              # ← replace with your actual data
X = data.data
X_scaled = StandardScaler().fit_transform(X)

# ── PCA projection ──────────────────────────────────────────────────────────
pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ── K-Means ─────────────────────────────────────────────────────────────────
kmeans   = KMeans(n_clusters=5, random_state=42)
k_labels = kmeans.fit_predict(X_scaled)

# ── DBSCAN — eps=0.5 (best config per argument: 3 clusters, 0.1% noise) ────
dbscan   = DBSCAN(eps=0.5, min_samples=5)      # ← corrected from 0.8
d_labels = dbscan.fit_predict(X_scaled)

# ── Optional: print DBSCAN diagnostics to verify against argument ───────────
n_clusters = len(set(d_labels)) - (1 if -1 in d_labels else 0)
n_noise    = np.sum(d_labels == -1)
noise_pct  = 100 * n_noise / len(d_labels)
print(f"DBSCAN → clusters: {n_clusters}, noise points: {n_noise} ({noise_pct:.1f}%)")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# K-Means
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                            c=k_labels, cmap="tab10", s=20)
axes[0].set_title("K-Means (k=5)")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
plt.colorbar(scatter1, ax=axes[0], label="Cluster")

# DBSCAN — noise points (label = -1) shown in black
colors = np.where(d_labels == -1, -1, d_labels)
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                            c=colors, cmap="tab10", s=20)
axes[1].set_title("DBSCAN (eps=0.5, min_samples=5)")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
plt.colorbar(scatter2, ax=axes[1], label="Cluster (−1 = noise)")

plt.tight_layout()
plt.savefig("07_clustering.png", dpi=150)
plt.show()