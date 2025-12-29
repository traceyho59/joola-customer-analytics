"""
JOOLA Churn What-If Dashboard
=============================
Interactive tool for predicting customer churn probability based on behavioral features.
"""

import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(
    page_title="JOOLA Churn What-If Dashboard",
    page_icon="🏓",
    layout="centered"
)

import pandas as pd
import numpy as np
import joblib
import os

# ------------------------
# Config
# ------------------------

FEATURE_COLS = [
    "avg_spend",
    "total_spend",
    "avg_items",
    "marketing_optin",
    "n_discounts",
    "avg_discount",
    "frequency",
    "avg_gap_days",
]

DISPLAY_NAMES = {
    "avg_spend": "Avg Spend per Order ($)",
    "total_spend": "Total Spend ($)",
    "avg_items": "Avg Items per Order",
    "marketing_optin": "Marketing Opt-In (0 = No, 1 = Yes)",
    "n_discounts": "# Discounts Used",
    "avg_discount": "Avg Discount ($)",
    "frequency": "Purchase Frequency (# Orders)",
    "avg_gap_days": "Avg Days Between Orders",
}

# ------------------------
# Helpers: load model + data
# ------------------------

def get_model_path():
    """Get the model path, checking multiple locations."""
    possible_paths = [
        "../models/churn_pipe.pkl",
        "models/churn_pipe.pkl",
        os.path.join(os.path.dirname(__file__), "..", "models", "churn_pipe.pkl"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return possible_paths[0]  # Default


def get_data_path():
    """Get the data path, checking multiple locations."""
    possible_paths = [
        "../models/churn_features.csv",
        "models/churn_features.csv",
        os.path.join(os.path.dirname(__file__), "..", "models", "churn_features.csv"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return possible_paths[0]  # Default


@st.cache_resource
def load_model():
    """Load trained sklearn pipeline."""
    path = get_model_path()
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"Model file not found at: {path}")
        st.info("Please ensure churn_pipe.pkl is in the models/ directory.")
        return None


@st.cache_data
def load_data():
    """Load feature data for slider ranges."""
    path = get_data_path()
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Data file not found at: {path}")
        st.info("Please ensure churn_features.csv is in the models/ directory.")
        return None


# ------------------------
# Main App
# ------------------------

def main():
    st.title("🏓 JOOLA Churn What-If Dashboard")
    
    st.markdown("""
    Use the controls in the sidebar to simulate a customer's behavior and 
    see the predicted probability that they **churn** according to the 
    logistic regression model.
    """)
    
    # Load model and data
    pipe = load_model()
    data = load_data()
    
    if pipe is None or data is None:
        st.warning("Unable to load model or data. Please check file paths.")
        st.stop()
    
    # ------------------------
    # Precompute slider ranges (1st–99th percentiles)
    # ------------------------
    
    feature_stats = {}
    for col in FEATURE_COLS:
        s = data[col].dropna()
        q1, q99 = s.quantile([0.01, 0.99])
        median = s.median()
        feature_stats[col] = {
            "min": float(q1),
            "max": float(q99),
            "median": float(median),
        }
    
    # ------------------------
    # Sidebar: Customer Profile Inputs
    # ------------------------
    
    st.sidebar.header("📊 Customer Profile")
    st.sidebar.markdown("Adjust the sliders to simulate different customer behaviors.")
    
    inputs = {}
    for col in FEATURE_COLS:
        stats = feature_stats[col]
        label = DISPLAY_NAMES.get(col, col)
    
        span = stats["max"] - stats["min"]
        step = span / 100 if span > 0 else 0.01
        step = max(step, 0.01)
    
        if col == "marketing_optin":
            # Binary toggle instead of slider
            default_idx = int(round(stats["median"]))
            if default_idx not in [0, 1]:
                default_idx = 0
            val = st.sidebar.selectbox(label, options=[0, 1], index=default_idx)
        else:
            val = st.sidebar.slider(
                label,
                min_value=float(round(stats["min"], 2)),
                max_value=float(round(stats["max"], 2)),
                value=float(round(stats["median"], 2)),
                step=float(round(step, 2)),
            )
    
        inputs[col] = val
    
    # Convert inputs to model-ready DataFrame
    input_df = pd.DataFrame([inputs], columns=FEATURE_COLS)
    
    # ------------------------
    # Prediction Display
    # ------------------------
    
    # Predict churn probability
    prob_churn = float(pipe.predict_proba(input_df)[0, 1])
    threshold = 0.5
    is_churner = prob_churn >= threshold
    label = "Likely to CHURN" if is_churner else "Likely to STAY"
    
    st.subheader("📈 Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Churn Probability",
            f"{prob_churn:.1%}",
            delta=None
        )
    
    with col2:
        if is_churner:
            st.error(f"**{label}** ⚠️")
        else:
            st.success(f"**{label}** ✅")
    
    st.write(
        f"Using a decision threshold of **{threshold:.0%}**, "
        f"this customer is classified as: **{label}**."
    )
    
    # ------------------------
    # Input Summary
    # ------------------------
    
    with st.expander("📋 Show Input Values"):
        st.dataframe(
            input_df.T.rename(columns={0: "Value"}),
            use_container_width=True
        )
    
    # ------------------------
    # Feature Importance Guide
    # ------------------------
    
    with st.expander("💡 Feature Impact Guide"):
        st.markdown("""
        Based on our analysis, here's how each feature typically affects churn:
        
        | Feature | Effect on Churn |
        |---------|-----------------|
        | **Number of Discounts Used** | ↓ More discounts = Lower churn |
        | **Total Spend** | ↓ Higher spend = Lower churn |
        | **Average Discount Amount** | ↑ Higher avg discount = Higher churn |
        | **Avg Spend per Order** | ↑ Higher avg order = Higher churn |
        | **Purchase Frequency** | ↓ More orders = Lower churn |
        
        *Note: These are general patterns from the model's learned relationships.*
        """)
    
    # Footer
    st.markdown("---")
    st.caption(
        "Model: Logistic regression with standardized features. "
        "Part of the JOOLA Customer Analytics project by Next Gen Consulting."
    )


if __name__ == "__main__":
    main()
