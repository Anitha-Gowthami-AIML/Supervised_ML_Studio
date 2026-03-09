import streamlit as st
import pandas as pd
import numpy as np
from theme import apply_theme

# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(layout="wide")
apply_theme("ML_image.jpeg")

st.title("🤖 Model Preparation & Selection")

# =====================================================
# 1️⃣ CHECK PREPROCESSED DATA
# =====================================================
# =====================================================
# Use fully preprocessed and encoded data
# =====================================================
# =====================================================
# 1️⃣ CHECK PREPROCESSED DATA
# =====================================================
if "preprocessed_data" not in st.session_state:
    st.warning("⚠ Please complete preprocessing first.")
    st.stop()

df = st.session_state.preprocessed_data.copy()

# Remove duplicate columns safely
if df.columns.duplicated().any():
    df = df.loc[:, ~df.columns.duplicated()]
    st.session_state.preprocessed_data = df.copy()

if "target" not in st.session_state:
    st.warning("⚠ Target column not found.")
    st.stop()

target = st.session_state.target


#df = st.session_state.preprocessed_data.copy()

# Remove duplicate columns safely
if df.columns.duplicated().any():
    df = df.loc[:, ~df.columns.duplicated()]
    st.session_state.preprocessed_data = df.copy()

if "target" not in st.session_state:
    st.warning("⚠ Target column not found.")
    st.stop()

target = st.session_state.target

# =====================================================
# 2️⃣ DATASET OVERVIEW
# =====================================================
st.markdown("## 📊 Updated Model Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    metric_rows = st.empty()
    metric_rows.metric("Rows", df.shape[0])

with col2:
    metric_cols = st.empty()
    metric_cols.metric("Columns", df.shape[1])

with col3:
    metric_target = st.empty()
    metric_target.metric("Target Column", target)

st.markdown("---")

# =====================================================
# 3️⃣ DATA PREVIEW
# =====================================================
st.markdown("🔎 Dataset Preview (First 20 Rows)")
st.dataframe(df.head(20), use_container_width=True, key="data_preview")

# =====================================================
# 4️⃣ DATA TYPES & MISSING VALUES
# =====================================================
st.markdown("📌 Data Types")
dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.values
})
st.dataframe(dtype_df, use_container_width=True, hide_index=True, key="dtype_table")

#st.markdown("📌 Missing Values")
#missing_df = pd.DataFrame({
#   "Column": df.columns,
#   "Missing Count": df.isnull().sum().values,
#   "Missing %": (df.isnull().sum() / len(df) * 100).round(2)
#})
#st.dataframe(missing_df, use_container_width=True, hide_index=True, key="missing_table")
#=================================================
# ====================================================
# ❗ Missing Values
# ====================================================
st.markdown("❗ Missing Value Summary")

missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing Count": df.isnull().sum(),
    "Missing %": (df.isnull().mean() * 100).round(2)
}).sort_values("Missing %", ascending=False)

st.dataframe(missing_df, use_container_width=True,hide_index=True)


# =====================================================
# 5️⃣ DESCRIPTIVE STATISTICS
# =====================================================
st.markdown("📈 Descriptive Statistics")
st.dataframe(
    df.describe(include="all"),
    use_container_width=True,
    key="desc_stats"
)

st.markdown("---")
st.success("Dataset Ready for Modeling 🚀", icon="✅")
st.markdown("---")

# =====================================================
# 6️⃣ TRAIN TEST SPLIT SETTINGS
# =====================================================
if "X" not in st.session_state or "y" not in st.session_state:
    X = df.drop(columns=[target])
    y = df[target]
    st.session_state.X = X.copy()
    st.session_state.y = y.copy()

# =====================================================
# NAVIGATION
# =====================================================
col1, col2 = st.columns(2)

with col1:
    if st.button("⬅ Back to Preprocessing", use_container_width=True, key="nav_back"):
        st.session_state["reset_advanced_preprocessing"] = False
        st.switch_page("pages/4_Advanced_Preprocessing.py")

with col2:
    if st.button("➡ Proceed to Model Training", use_container_width=True, key="nav_next"):

        st.switch_page("pages/6_training.py")
