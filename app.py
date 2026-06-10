import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Isolation Forest - Anomaly Detection",
    page_icon=None,
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
with open("static/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Load Artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    iso    = pickle.load(open("models/iso_forest.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl",     "rb"))
    meta   = pickle.load(open("models/meta.pkl",        "rb"))
    return iso, scaler, meta

iso, scaler, meta = load_artifacts()

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 class='main-title'>Isolation Forest - Anomaly Detection</h1>",
    unsafe_allow_html=True
)
st.markdown(
    """
    <div class='info-box'>
    Detect fraudulent credit card transactions using
    <b>Isolation Forest</b> (Unsupervised Anomaly Detection)
    &nbsp;|&nbsp; Dataset: Credit Card Fraud Detection
    </div>
    """,
    unsafe_allow_html=True
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Algorithm Info")
st.sidebar.success("Algorithm: Isolation Forest")
st.sidebar.info(f"N Estimators: {meta['n_estimators']}")
st.sidebar.info(f"Contamination: {meta['contamination']:.4f}")
st.sidebar.info(f"Decision Threshold: {meta['decision_threshold']:.4f}")
st.sidebar.success("Dataset: Credit Card Fraud Detection")
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Stats")
st.sidebar.metric("Total Transactions", f"{meta['n_samples_total']:,}")
st.sidebar.metric("Normal",  f"{meta['n_normal']:,}")
st.sidebar.metric("Fraud",   f"{meta['n_fraud']:,}")
st.sidebar.metric("Fraud %", f"{meta['fraud_pct']:.4f}%")
st.sidebar.markdown("---")
st.sidebar.subheader("Model Metrics")
st.sidebar.metric("Precision", f"{meta['precision']*100:.2f}%")
st.sidebar.metric("Recall",    f"{meta['recall']*100:.2f}%")
st.sidebar.metric("F1 Score",  f"{meta['f1_score']*100:.2f}%")
st.sidebar.metric("ROC-AUC",   f"{meta['roc_auc']:.4f}")

# ── Top Metric Cards ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Model Performance Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Precision",  f"{meta['precision']*100:.2f}%")
c2.metric("Recall",     f"{meta['recall']*100:.2f}%")
c3.metric("F1 Score",   f"{meta['f1_score']*100:.2f}%")
c4.metric("ROC-AUC",    f"{meta['roc_auc']:.4f}")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Confusion Matrix")
if os.path.exists("static/confusion_matrix.png"):
    st.image("static/confusion_matrix.png", width="stretch")
    cm = meta["confusion_matrix"]
    st.caption(
        f"TN={cm['tn']:,} (correctly identified normal) | "
        f"FP={cm['fp']:,} (normal flagged as fraud) | "
        f"FN={cm['fn']:,} (fraud missed) | "
        f"TP={cm['tp']:,} (correctly caught fraud)"
    )
else:
    st.warning("Run train_model.py first.")

# ── Anomaly Scatter ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Anomaly Detection Visualisation (PCA 2D Projection)")
if os.path.exists("static/anomaly_scatter.png"):
    st.image("static/anomaly_scatter.png", width="stretch")
    st.caption(
        "Left: true labels. Right: Isolation Forest predictions. "
        "PCA reduces 29 V-features to 2D for visualisation."
    )
else:
    st.warning("Run train_model.py first.")

# ── Score Distribution ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Anomaly Score Distribution")
if os.path.exists("static/score_distribution.png"):
    st.image("static/score_distribution.png", width="stretch")
    st.caption(
        f"Normal avg score: {meta['score_normal_mean']:.4f} | "
        f"Fraud avg score: {meta['score_fraud_mean']:.4f}. "
        "Fraud transactions have consistently lower anomaly scores, "
        "confirming the model separates the two classes."
    )
else:
    st.warning("Run train_model.py first.")

# ── Feature Importance ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Top Features by Anomaly Importance")
if os.path.exists("static/feature_importance.png"):
    st.image("static/feature_importance.png", width="stretch")
    st.caption(
        "Permutation-based importance: each feature is shuffled on fraud samples; "
        "larger mean score shift = more influential for anomaly detection."
    )
else:
    st.warning("Run train_model.py first.")

# ── Threshold Analysis ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Threshold Analysis: Precision / Recall / F1")
if os.path.exists("static/threshold_analysis.png"):
    st.image("static/threshold_analysis.png", width="stretch")
    st.caption(
        "Varying the decision threshold reveals the precision-recall trade-off. "
        "In fraud detection, high recall is often prioritised to minimise missed fraud (FN)."
    )
else:
    st.warning("Run train_model.py first.")

# ── Live Prediction Tab ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Live Transaction Anomaly Prediction")
st.write(
    "Paste values for all 29 features (V1–V28 + Amount) to get a real-time "
    "anomaly score and fraud/normal prediction."
)

feature_names = meta["feature_names"]   # V1..V28, Amount

with st.expander("Enter transaction feature values", expanded=True):
    col_pairs = [st.columns(4) for _ in range((len(feature_names) + 3) // 4)]
    inputs = {}
    for i, fname in enumerate(feature_names):
        row   = i // 4
        col   = i %  4
        inputs[fname] = col_pairs[row][col].number_input(
            fname, value=0.0, format="%.6f", key=fname
        )

    predict_btn = st.button("Predict Anomaly Score")

if predict_btn:
    x_in  = np.array([[inputs[f] for f in feature_names]], dtype=np.float32)
    x_sc  = scaler.transform(x_in)
    score = float(iso.decision_function(x_sc)[0])
    pred  = iso.predict(x_sc)[0]   # 1 = normal, -1 = anomaly

    label = "FRAUD / ANOMALY" if pred == -1 else "NORMAL"
    color = "#f87171" if pred == -1 else "#34d399"

    st.markdown(
        f"<div class='result-box'>"
        f"Anomaly Score: <span style='color:{color}'>{score:.6f}</span><br>"
        f"Decision: <span style='color:{color}'>{label}</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.caption(
        f"Threshold = {meta['decision_threshold']:.6f}. "
        "Scores below threshold are flagged as anomalies."
    )

# ── Algorithm Explanation ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("How Isolation Forest Works")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Core Idea**")
    st.write("- Anomalies are few and different from normal points")
    st.write("- Randomly partition the feature space with isolation trees")
    st.write("- Anomalies are isolated in fewer splits (shorter path length)")
    st.write("- Average path length across all trees gives the anomaly score")
    st.write("- No distance or density computation — very fast")
    st.write("- Works well in high-dimensional spaces")
with col2:
    st.markdown("**Key Hyperparameters**")
    st.write(f"- **n_estimators = {meta['n_estimators']}**: number of isolation trees")
    st.write(f"- **contamination = {meta['contamination']:.4f}**: expected anomaly fraction")
    st.write("- **max_samples**: sub-sample size per tree (default = 'auto' = 256)")
    st.write("- **max_features**: features per split (1.0 = all features)")
    st.write("- **bootstrap = False**: sampling without replacement")

# ── Pipeline Used ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Pipeline Used in This Project")

st.markdown(
    f"""
    **Step 1 – Drop Time**: Remove the `Time` column (transaction order, not informative)

    **Step 2 – Scale**: StandardScaler on all 29 features (V1–V28 + Amount)

    **Step 3 – Isolation Forest**: 200 trees, contamination = {meta['contamination']:.4f}  
    (set to true fraud proportion from dataset labels)

    **Step 4 – Evaluate**: Predictions compared against ground-truth labels  
    Precision={meta['precision']:.4f} | Recall={meta['recall']:.4f} | F1={meta['f1_score']:.4f} | ROC-AUC={meta['roc_auc']:.4f}
    """
)

# ── Confusion Matrix Detail Table ─────────────────────────────────────────────
st.markdown("---")
st.subheader("Confusion Matrix Breakdown")
cm = meta["confusion_matrix"]
cm_table = pd.DataFrame({
    "Category":    ["True Negative (TN)", "False Positive (FP)", "False Negative (FN)", "True Positive (TP)"],
    "Count":       [cm["tn"], cm["fp"], cm["fn"], cm["tp"]],
    "Meaning":     [
        "Normal transactions correctly identified",
        "Normal transactions wrongly flagged as fraud",
        "Fraud transactions missed by model",
        "Fraud transactions correctly caught"
    ]
})
st.dataframe(cm_table, use_container_width=True)

# ── Top Features Table ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Feature Importance Table")
feat_rows = [
    {"Feature": k, "Importance Score": f"{v:.6f}"}
    for k, v in meta["top_features"].items()
]
st.dataframe(pd.DataFrame(feat_rows), use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888;'>"
    "Unsupervised ML - Isolation Forest - Credit Card Fraud Detection"
    "</p>",
    unsafe_allow_html=True
)
