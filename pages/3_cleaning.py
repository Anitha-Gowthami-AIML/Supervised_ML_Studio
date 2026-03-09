import streamlit as st
import pandas as pd
import numpy as np
from theme import apply_theme

st.set_page_config(layout="wide")
apply_theme("ML_image.jpeg")

st.title("🧹 Data Cleaning & Preprocessing")

# =====================================================
# 🔷 Custom Styling (Tabs + Container)
# =====================================================

st.markdown("""
<style>

/* Tabs container */
.stTabs [data-baseweb="tab-list"] {
    background-color: rgba(255,255,255,0.08);
    padding: 10px;
    border-radius: 12px;
}

/* Tab style */
.stTabs [data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 600;
    padding: 8px 18px;
    border-radius: 8px;
}

/* Active tab */
.stTabs [aria-selected="true"] {
    background-color: #4CAF50 !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 1️⃣ CHECK DATA EXISTS
# =====================================================

if "working_data" not in st.session_state:
    st.warning("⚠ Please complete Mapping first.")
    st.stop()

df = st.session_state.working_data

# Save original mapping output for reset
if "mapping_output" not in st.session_state:
    st.session_state.mapping_output = df.copy()

# =====================================================
# 🔷 Background Container
# =====================================================

with st.container():

    st.markdown("""
    <div style="
        background-color: rgba(255,255,255,0.05);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    ">
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Preview",
        "🧩 Missing Values",
        "🗑 Duplicates",
        "♻ Reset"
    ])

    # =====================================================
    # TAB 1 — PREVIEW
    # =====================================================

    with tab1:
        st.subheader("Dataset Snapshot")
        st.dataframe(df.head(), use_container_width=True, hide_index=True)
        st.write("Shape:", df.shape)

    # =====================================================
    # TAB 2 — MISSING VALUES
    # =====================================================

    with tab2:

        st.subheader("Missing Value Treatment")

        missing_counts = df.isnull().sum()
        missing_df = pd.DataFrame({
            "Column": missing_counts.index,
            "Missing Values": missing_counts.values
        })

        st.dataframe(missing_df, use_container_width=True, hide_index=True)

        col_to_clean = st.selectbox(
            "Select Column",
            df.columns
        )

        clean_option = st.selectbox(
            "Select Method",
            ["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"]
        )

        if st.button("Apply Missing Value Treatment"):

            if clean_option == "Drop Rows":
                df = df.dropna(subset=[col_to_clean])

            elif clean_option == "Fill with Mean":
                if pd.api.types.is_numeric_dtype(df[col_to_clean]):
                    df[col_to_clean] = df[col_to_clean].fillna(df[col_to_clean].mean())
                else:
                    st.error("Mean works only for numeric columns.")
                    st.stop()

            elif clean_option == "Fill with Median":
                if pd.api.types.is_numeric_dtype(df[col_to_clean]):
                    df[col_to_clean] = df[col_to_clean].fillna(df[col_to_clean].median())
                else:
                    st.error("Median works only for numeric columns.")
                    st.stop()

            elif clean_option == "Fill with Mode":
                df[col_to_clean] = df[col_to_clean].fillna(df[col_to_clean].mode()[0])

            st.session_state.working_data = df
            st.success("Missing values handled successfully!")
            st.rerun()

    # =====================================================
    # TAB 3 — REMOVE DUPLICATES
    # =====================================================

    with tab3:

        st.subheader("Remove Duplicate Rows")

        duplicate_count = df.duplicated().sum()
        st.write(f"Duplicate Rows: {duplicate_count}")

        if st.button("Remove Duplicates"):
            df = df.drop_duplicates()
            st.session_state.working_data = df
            st.success("Duplicates removed successfully!")
            st.rerun()

    # =====================================================
    # TAB 4 — RESET
    # =====================================================

    with tab4:

        st.subheader("Reset Dataset")

        if st.button("Reset to Mapping Output"):
            st.session_state.working_data = st.session_state.mapping_output.copy()
            st.success("Dataset reset successfully!")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FINAL PREVIEW
# =====================================================

st.markdown("---")
st.subheader("📌 Cleaned Data Preview")

st.dataframe(
    st.session_state.working_data.head(),
    use_container_width=True,
    hide_index=True
)

# =====================================================
# NAVIGATION
# =====================================================

#st.markdown("---")

#col1, col2 = st.columns(2)

#with col1:
#    if st.button("⬅ Back to Mapping", use_container_width=True):
#       st.switch_page("pages/2_mapping.py")

#with col2:
#   if st.button("➡ Next: Advanced Preprocessing", use_container_width=True):
#       #final_df = pd.concat([X, y], axis=1)
#        st.session_state.preprocessed_data = st.session_state.working_data.copy()
#        st.switch_page("pages/4_Advanced_Preprocessing.py")

#=====================================================
# new navigation
#====================================================

# =====================================================
# NAVIGATION
# =====================================================

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("⬅ Back to Mapping", use_container_width=True):
        st.switch_page("pages/2_mapping.py")

with col2:
    if st.button("➡ Next: Advanced Preprocessing", use_container_width=True):

        if "target" not in st.session_state:
            st.error("Target column not defined.")
            st.stop()

        target_col = st.session_state.target

        df_cleaned = st.session_state.working_data.copy()

        if target_col not in df_cleaned.columns:
            st.error(f"Target column '{target_col}' missing from dataset.")
            st.stop()

        # Ensure target remains inside cleaned_data
        st.session_state.cleaned_data = df_cleaned.copy()
        st.session_state["reset_advanced_preprocessing"] = True
        st.switch_page("pages/4_Advanced_preprocessing.py")

        st.write("Columns in working_data:", st.session_state.working_data.columns.tolist())
        st.write("Target:", st.session_state.target)



        
