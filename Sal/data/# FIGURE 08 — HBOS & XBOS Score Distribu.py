# FIGURE 08 — HBOS & XBOS Score Distributions
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.hbos import HBOS
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris          # ← replace with your actual data

# ── Load & scale data ───────────────────────────────────────────────────────
data = load_iris()                              # ← replace with your actual data
X = data.data
X_scaled = StandardScaler().fit_transform(X)

# ── HBOS (single scorer, default bins=10) ───────────────────────────────────
hbos = HBOS(n_bins=10)
hbos.fit(X_scaled)
hbos_scores = hbos.decision_scores_           # raw anomaly scores

# ── XBOS — average of 3 HBOS scorers with bins=8, 12, 16 ───────────────────
# (matches argument: "averaging scores from three independent HBOS
#  scorers using different bin counts (8, 12, and 16 bins)")
xbos_scores = np.mean([
    HBOS(n_bins=8).fit(X_scaled).decision_scores_,
    HBOS(n_bins=12).fit(X_scaled).decision_scores_,
    HBOS(n_bins=16).fit(X_scaled).decision_scores_,
], axis=0)

# ── Thresholds at 95th percentile ───────────────────────────────────────────
hbos_thresh = np.percentile(hbos_scores, 95)
xbos_thresh = np.percentile(xbos_scores, 95)

# ── Diagnostics — verify against argument ───────────────────────────────────
hbos_flags = hbos_scores >= hbos_thresh
xbos_flags = xbos_scores >= xbos_thresh

n_xbos_anomalies = np.sum(xbos_flags)
agreement        = np.sum(hbos_flags & xbos_flags) / np.sum(hbos_flags | xbos_flags) * 100

print(f"HBOS  anomalies : {np.sum(hbos_flags)}")
print(f"XBOS  anomalies : {n_xbos_anomalies}")          # argument says 109
print(f"Agreement (IoU) : {agreement:.1f}%")            # argument says 98.4%

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# HBOS
axes[0].hist(hbos_scores, bins=30, color="steelblue", edgecolor="white")
axes[0].axvline(hbos_thresh, color="red", linestyle="dashed", linewidth=1.5,
                label=f"95th pct = {hbos_thresh:.2f}")
axes[0].set_title("HBOS Score Distribution")
axes[0].set_xlabel("Anomaly Score")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# XBOS
axes[1].hist(xbos_scores, bins=30, color="darkorange", edgecolor="white")
axes[1].axvline(xbos_thresh, color="red", linestyle="dashed", linewidth=1.5,
                label=f"95th pct = {xbos_thresh:.2f}")
axes[1].set_title("XBOS Score Distribution\n(avg of bins=8,12,16)")
axes[1].set_xlabel("Anomaly Score")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
plt.savefig("08_hbos_xbos.png", dpi=150)
plt.show()