# FIGURE 09 — Churn Rate by Tenure Group
import pandas as pd
import matplotlib.pyplot as plt

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\SAl\Desktop\Telco-Customer-Churn.csv")

# ── Ensure Churn is numeric ──────────────────────────────────────────────────
df['Churn'] = df['Churn'].astype(str).str.strip()
print("Unique Churn values:", df['Churn'].unique())

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0,
                                'True': 1, 'False': 0,
                                '1': 1, '0': 0,
                                '1.0': 1, '0.0': 0})

if df['Churn'].isnull().any():
    print("WARNING: unmapped Churn values — check printout above")
else:
    print("Churn mapping successful")

# ── Tenure groups ────────────────────────────────────────────────────────────
df['tenure_group'] = pd.cut(
    df['tenure'],
    bins=[0, 12, 24, 48, 72],
    labels=["New", "Early", "Mid", "Loyal"]
)

# ── Churn rate per group as percentage ───────────────────────────────────────
churn_rate = df.groupby('tenure_group', observed=True)['Churn'].mean() * 100

# ── Diagnostics ──────────────────────────────────────────────────────────────
print("\nChurn rate by tenure group:")
for group, rate in churn_rate.items():
    print(f"  {group:6s}: {rate:.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

churn_rate.plot(
    kind='bar',
    ax=ax,
    color=["#d73027", "#fc8d59", "#91bfdb", "#4575b4"],
    edgecolor='white',
    width=0.6
)

for i, val in enumerate(churn_rate):
    ax.text(i, val + 0.5, f"{val:.1f}%",
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel("Tenure Group", labelpad=10)
ax.set_ylabel("Churn Rate (%)")
ax.set_title("Churn Rate by Tenure Group")
ax.set_xticklabels(["New\n(0–12 mo)", "Early\n(13–24 mo)",
                     "Mid\n(25–48 mo)", "Loyal\n(49–72 mo)"],
                    rotation=0)
ax.set_ylim(0, churn_rate.max() + 8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

plt.tight_layout()
plt.savefig("09_temporal_churn.png", dpi=150)
plt.show()