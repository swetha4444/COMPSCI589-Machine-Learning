import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Style
sns.set(style="whitegrid")
palette = sns.color_palette("Set2")

# Data
model_details = [
    "[11, 5, 1]\nreg=0.01, step=0.05, batch=10",
    "[11, 5, 10, 1]\nreg=0.01, step=0.05, batch=10",
    "[11, 10, 5, 8, 1]\nreg=0.01, step=0.05, batch=32",
    "[11, 2, 4, 8, 16, 1]\nreg=0.01, step=0.05, batch=32",
    "[11, 10, 5, 8, 1]\nreg=0.01, step=0.05, batch=20",
    "[11, 16, 8, 1]\nreg=0.01, step=0.01, batch=30",
    "[11, 4, 2, 4, 1]\nreg=0.1, step=0.1, batch=10, eps=1e-5"
]
accuracy = [80.378, 80.798, 77.008, 73.64, 78.482, 74.062, 69.176]
f1 = [87.384, 87.616, 85.626, 84.0, 86.414, 84.152, 81.782]

# Create DataFrame
df = pd.DataFrame({
    'Model Details': model_details,
    'Accuracy': accuracy,
    'F1 Score': f1
})

# Plot
plt.figure(figsize=(14, 6))
ax = sns.lineplot(data=df, x='Model Details', y='Accuracy', marker='o', label='Accuracy', palette=palette)
sns.lineplot(data=df, x='Model Details', y='F1 Score', marker='s', label='F1 Score', palette=palette)

# Annotate acc and f1 values
for i in range(len(df)):
    plt.text(i, accuracy[i] + 0.4, f"{accuracy[i]:.2f}", ha='center', fontsize=9)
    plt.text(i, f1[i] + 0.4, f"{f1[i]:.2f}", ha='center', fontsize=9)

# Final tweaks
plt.title("Loan Dataset: Accuracy and F1 Score Across Model Configurations")
plt.xticks(rotation=30, ha='right')
plt.ylim(65, 90)
plt.ylabel("Score (%)")
plt.xlabel("Model (Architecture + Hyperparameters)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
