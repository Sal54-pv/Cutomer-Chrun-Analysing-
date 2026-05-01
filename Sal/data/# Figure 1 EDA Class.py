
#Sal: Churn Payton code:


# Figure 1: EDA Class

import sys
print(sys.executable)

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("C:/Users/SAl/Desktop/Telco-Customer-Churn.csv")

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Convert Churn to readable labels
df['ChurnLabel'] = df['Churn'].map({'Yes': 'Churn', 'No': 'No Churn'})

# Create figure
plt.figure(figsize=(12, 5))

# -------------------------------
# Plot 1: Class Distribution
# -------------------------------
plt.subplot(1, 2, 1)
df['ChurnLabel'].value_counts().plot(kind='bar')
plt.title("Class Distribution (Churn vs No Churn)")
plt.xlabel("Class")
plt.ylabel("Count")

# -------------------------------
# Plot 2: Monthly Charges Histogram
# -------------------------------
plt.subplot(1, 2, 2)
plt.hist(df['MonthlyCharges'], bins=30)
plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Frequency")

# Layout and save
plt.tight_layout()
plt.savefig("01_eda_overview.png")
plt.show()

