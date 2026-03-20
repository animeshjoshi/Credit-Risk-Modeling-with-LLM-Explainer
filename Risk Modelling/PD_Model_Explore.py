import pandas as pd
import numpy as np
import statsmodels.api as sm
np.random.seed(42)
X_data = pd.read_csv('Data/Credit Risk Training Dataset.csv').drop('Unnamed: 0', axis = 1)
y_data = pd.read_csv('Data/Credit Risk Full Dataset.csv')['Default']

X = sm.add_constant(X_data)
y = y_data

model = sm.Logit(y,X)

result = model.fit()

print(result.summary())

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, auc

# --- Get predictions (from your fitted model) ---
y_prob = result.predict(X)
fpr, tpr, thresholds = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

# --- Plot ---
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor("#0f0f1a")
ax.set_facecolor("#0f0f1a")

# Gradient fill under curve
ax.fill_between(fpr, tpr, alpha=0.15, color="#7c3aed")

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5,
        color="#4b5563", label="Random Classifier (AUC = 0.50)")

# ROC curve with gradient color
points = np.array([fpr, tpr]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list("auc_cmap", ["#7c3aed", "#06b6d4", "#10b981"])
lc = LineCollection(segments, cmap=cmap, linewidth=3, zorder=3)
lc.set_array(np.linspace(0, 1, len(fpr)))
ax.add_collection(lc)

# Optimal threshold point
optimal_idx = np.argmax(tpr - fpr)
ax.scatter(fpr[optimal_idx], tpr[optimal_idx],
           color="#f59e0b", s=120, zorder=5, label=f"Optimal Threshold = {thresholds[optimal_idx]:.2f}")
ax.annotate(f"  Best Threshold\n  ({fpr[optimal_idx]:.2f}, {tpr[optimal_idx]:.2f})",
            xy=(fpr[optimal_idx], tpr[optimal_idx]),
            color="#f59e0b", fontsize=9, va="center")

# AUC score badge
ax.text(0.60, 0.12, f"AUC = {roc_auc:.4f}", transform=ax.transAxes,
        fontsize=22, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#7c3aed", alpha=0.85, edgecolor="#a78bfa"))

# Styling
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.set_xlabel("False Positive Rate", fontsize=13, color="#d1d5db", labelpad=10)
ax.set_ylabel("True Positive Rate", fontsize=13, color="#d1d5db", labelpad=10)
ax.set_title("ROC Curve — Logistic Regression", fontsize=16,
             fontweight="bold", color="white", pad=20)

ax.tick_params(colors="#6b7280")
for spine in ax.spines.values():
    spine.set_edgecolor("#1f2937")

ax.grid(True, color="#1f2937", linewidth=0.8, linestyle="--")

# Legend
legend = ax.legend(loc="lower right", fontsize=10,
                   facecolor="#1f2937", edgecolor="#374151", labelcolor="white")

plt.tight_layout()
plt.savefig("Logistic Regression PD AUC Plot.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("Logistic Regression PD AUC Plot.png")