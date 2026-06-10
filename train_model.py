import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0a0f1e"
AX_BG   = "#0f1e2e"
ACCENT  = "#38bdf8"
RED     = "#f87171"
GREEN   = "#34d399"
YELLOW  = "#fbbf24"
ORANGE  = "#fb923c"

def style_ax(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color(ACCENT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("data/creditcard.csv")
print(f"Raw shape: {df.shape}")

# Drop 'Time' column (not useful for anomaly detection)
df = df.drop(columns=["Time"])

y = df["Class"].values          # 0 = normal, 1 = fraud
X = df.drop(columns=["Class"]).values.astype(np.float32)

n_total    = len(y)
n_fraud    = int(y.sum())
n_normal   = n_total - n_fraud
fraud_pct  = n_fraud / n_total * 100

print(f"Features: {X.shape[1]}  |  Samples: {n_total:,}")
print(f"Normal: {n_normal:,}  |  Fraud: {n_fraud:,}  ({fraud_pct:.4f}%)")

# ── 2. Scale ──────────────────────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nFeatures scaled with StandardScaler.")

# ── 3. Train Isolation Forest ─────────────────────────────────────────────────
# contamination = true fraud proportion
contamination = round(n_fraud / n_total, 4)
print(f"\nTraining Isolation Forest (contamination={contamination})...")

iso = IsolationForest(
    n_estimators   = 200,
    contamination  = contamination,
    max_samples    = "auto",
    max_features   = 1.0,
    bootstrap      = False,
    random_state   = 42,
    n_jobs         = -1
)
iso.fit(X_scaled)

# Predictions: 1 = normal, -1 = anomaly (convert to 0=normal, 1=fraud)
raw_pred  = iso.predict(X_scaled)
y_pred    = np.where(raw_pred == -1, 1, 0)
scores    = iso.decision_function(X_scaled)   # higher = more normal

print("Training complete.")

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
cm        = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()
precision = precision_score(y, y_pred, zero_division=0)
recall    = recall_score(y, y_pred, zero_division=0)
f1        = f1_score(y, y_pred, zero_division=0)
roc_auc   = roc_auc_score(y, -scores)   # negate: lower score = more anomalous

print(f"\n  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")

# ── 5. PCA 2D for visualisation ───────────────────────────────────────────────
print("\nRunning PCA 2D for visualisation...")
N_VIZ = 5000
np.random.seed(42)
idx_viz  = np.random.choice(len(X_scaled), size=N_VIZ, replace=False)
X_viz    = X_scaled[idx_viz]
y_viz    = y[idx_viz]
pred_viz = y_pred[idx_viz]

pca2 = PCA(n_components=2, random_state=42)
X_pca = pca2.fit_transform(X_viz)

# ── 6. Anomaly Scatter Plot ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(BG)
for ax in axes:
    style_ax(ax)

# Ground truth
for cls, color, label in [(0, GREEN, "Normal"), (1, RED, "Fraud")]:
    m = y_viz == cls
    axes[0].scatter(X_pca[m, 0], X_pca[m, 1],
                    c=color, label=label, alpha=0.5, s=10, edgecolors="none")
axes[0].set_title("Ground Truth Labels (PCA 2D)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("PC 1"); axes[0].set_ylabel("PC 2")
axes[0].legend(facecolor=AX_BG, labelcolor="white", fontsize=10, markerscale=3)
axes[0].grid(True, linestyle="--", alpha=0.15)

# Isolation Forest predictions
for cls, color, label in [(0, GREEN, "Predicted Normal"), (1, RED, "Predicted Fraud")]:
    m = pred_viz == cls
    axes[1].scatter(X_pca[m, 0], X_pca[m, 1],
                    c=color, label=label, alpha=0.5, s=10, edgecolors="none")
axes[1].set_title("Isolation Forest Predictions (PCA 2D)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("PC 1"); axes[1].set_ylabel("PC 2")
axes[1].legend(facecolor=AX_BG, labelcolor="white", fontsize=10, markerscale=3)
axes[1].grid(True, linestyle="--", alpha=0.15)

plt.suptitle("Anomaly Detection: Ground Truth vs Predictions",
             color="white", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("static/anomaly_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved anomaly_scatter.png")

# ── 7. Anomaly Score Distribution ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG)
style_ax(ax)

scores_normal = scores[y == 0]
scores_fraud  = scores[y == 1]

ax.hist(scores_normal, bins=80, color=GREEN, alpha=0.6, label="Normal", density=True)
ax.hist(scores_fraud,  bins=80, color=RED,   alpha=0.8, label="Fraud",  density=True)

threshold = iso.offset_
ax.axvline(threshold, color=YELLOW, linewidth=2, linestyle="--",
           label=f"Decision Threshold ({threshold:.4f})")

ax.set_title("Anomaly Score Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Anomaly Score (higher = more normal)")
ax.set_ylabel("Density")
ax.legend(facecolor=AX_BG, labelcolor="white", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.15)

plt.tight_layout()
plt.savefig("static/score_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved score_distribution.png")

# ── 8. Confusion Matrix ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor(BG)
style_ax(ax)

cm_display = np.array([[tn, fp], [fn, tp]])
im = ax.imshow(cm_display, interpolation="nearest",
               cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)

classes = ["Normal", "Fraud"]
tick_marks = np.arange(2)
ax.set_xticks(tick_marks); ax.set_xticklabels(classes, color="white")
ax.set_yticks(tick_marks); ax.set_yticklabels(classes, color="white")
ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

thresh_cm = cm_display.max() / 2.0
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm_display[i, j]:,}",
                ha="center", va="center",
                color="white" if cm_display[i, j] < thresh_cm else "black",
                fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("static/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved confusion_matrix.png")

# ── 9. Feature Importance (Mean Anomaly Score Shift per Feature) ───────────────
print("\nComputing feature importances...")
feature_names = [c for c in df.columns if c != "Class"]

# Permutation-style: shuffle one feature, measure score change on fraud subset
fraud_mask = y == 1
X_fraud    = X_scaled[fraud_mask]
base_scores = iso.decision_function(X_fraud)
base_mean   = base_scores.mean()

importances = []
np.random.seed(42)
for i in range(X_fraud.shape[1]):
    X_perm = X_fraud.copy()
    X_perm[:, i] = np.random.permutation(X_perm[:, i])
    perm_mean = iso.decision_function(X_perm).mean()
    importances.append(abs(base_mean - perm_mean))

importances = np.array(importances)
top_idx     = np.argsort(importances)[::-1][:15]
top_names   = [feature_names[i] for i in top_idx]
top_scores  = importances[top_idx]

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor(BG)
style_ax(ax)

colors = [ACCENT if i == 0 else "#0284c7" for i in range(len(top_names))]
bars = ax.barh(range(len(top_names)), top_scores[::-1], color=colors[::-1],
               edgecolor="none", height=0.7)
ax.set_yticks(range(len(top_names)))
ax.set_yticklabels(top_names[::-1], color="white")
ax.set_xlabel("Mean Score Shift on Fraud Samples (Permutation)")
ax.set_title("Top 15 Features by Anomaly Importance", fontsize=14, fontweight="bold")
ax.grid(True, axis="x", linestyle="--", alpha=0.2)

plt.tight_layout()
plt.savefig("static/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved feature_importance.png")

# ── 10. Threshold Analysis ────────────────────────────────────────────────────
thresholds  = np.linspace(scores.min(), scores.max(), 200)
precisions  = []
recalls     = []
f1s         = []

for t in thresholds:
    y_t = (scores < t).astype(int)   # below threshold = anomaly
    precisions.append(precision_score(y, y_t, zero_division=0))
    recalls.append(recall_score(y, y_t, zero_division=0))
    f1s.append(f1_score(y, y_t, zero_division=0))

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG)
style_ax(ax)

ax.plot(thresholds, precisions, color=GREEN,  linewidth=2, label="Precision")
ax.plot(thresholds, recalls,    color=RED,    linewidth=2, label="Recall")
ax.plot(thresholds, f1s,        color=YELLOW, linewidth=2, label="F1 Score")
ax.axvline(threshold, color=ORANGE, linewidth=1.5, linestyle="--",
           label=f"Fitted Threshold ({threshold:.4f})")

ax.set_title("Precision / Recall / F1 vs Decision Threshold",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.legend(facecolor=AX_BG, labelcolor="white", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.15)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig("static/threshold_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved threshold_analysis.png")

# ── 11. Save Artifacts ────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

feature_importance_dict = {
    name: round(float(score), 6)
    for name, score in zip(top_names, top_scores)
}

meta = {
    "n_samples_total":   n_total,
    "n_features":        X.shape[1],
    "n_normal":          n_normal,
    "n_fraud":           n_fraud,
    "fraud_pct":         round(fraud_pct, 4),
    "contamination":     contamination,
    "n_estimators":      200,
    "decision_threshold":round(float(threshold), 6),
    "precision":         round(precision, 4),
    "recall":            round(recall, 4),
    "f1_score":          round(f1, 4),
    "roc_auc":           round(roc_auc, 4),
    "confusion_matrix":  {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "feature_names":     feature_names,
    "top_features":      feature_importance_dict,
    "score_normal_mean": round(float(scores_normal.mean()), 6),
    "score_fraud_mean":  round(float(scores_fraud.mean()), 6),
}

pickle.dump(iso,     open("models/iso_forest.pkl", "wb"))
pickle.dump(scaler,  open("models/scaler.pkl",     "wb"))
pickle.dump(meta,    open("models/meta.pkl",        "wb"))

print("\nAll artifacts saved!")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")
