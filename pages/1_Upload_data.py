import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from theme import apply_theme

st.set_page_config(layout="wide")
apply_theme("ML_image.jpeg")

# ====================================================
st.title("📂 Upload Dataset & Overview")

# --- Upload Section ---
uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    
    # Load Data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    #st.session_state['raw_data'] = df
    #st.session_state.df = df
    st.session_state.raw_data = df
    st.session_state.working_data = df.copy()

    st.success("Dataset uploaded successfully!")

    # ====================================================
    # 📊 Dataset Overview
    # ====================================================
    st.markdown("### 📊 Dataset Overview")

    st.markdown('<div class="overview-container">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Total Missing", df.isnull().sum().sum())

    st.markdown('</div>', unsafe_allow_html=True)

    st.dataframe(df.head(), use_container_width=True,hide_index=True)

    # ====================================================
    # 🧾 Data Types
    # ====================================================
    st.subheader("🧾 Data Types")

    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes
    })

    #st.dataframe(dtype_df, use_container_width=True,hide_index=True)
    st.data_editor(
    dtype_df,
    use_container_width=True,
    hide_index=True,
    disabled=True
)

    # ====================================================
    # ❗ Missing Values
    # ====================================================
    st.subheader("❗ Missing Value Summary")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().mean() * 100).round(2)
    }).sort_values("Missing %", ascending=False)

    st.dataframe(missing_df, use_container_width=True,hide_index=True)

    # ====================================================
    # 📈 Descriptive Statistics
    # ====================================================
    st.subheader("📈 Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    # ====================================================
    # 🔥 Correlation Heatmap
    # ====================================================
    # ====================================================
    # 🔥 Correlation Heatmap
    # ====================================================

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 1:
        st.subheader("🔥 Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df[numeric_cols].corr()

        cax = ax.matshow(corr, cmap="coolwarm")

        # Axis ticks
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=90)
        ax.set_yticklabels(numeric_cols)

        # Add correlation values inside cells
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                        ha="center", va="center",
                        color="black")

        fig.colorbar(cax)
        st.pyplot(fig)
    #==================================================
    
    # ====================================================
    # 📦 Outlier Analysis (IQR Method)
    # ====================================================
    st.subheader("📦 Outlier Analysis (IQR Method)")

    if len(numeric_cols) > 0:

        outlier_summary = []

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

            outlier_summary.append({
                "Column": col,
                "Lower Bound": round(lower_bound, 2),
                "Upper Bound": round(upper_bound, 2),
                "Outlier Count": len(outliers),
                "Outlier %": round(len(outliers)/len(df)*100, 2)
            })

        outlier_df = pd.DataFrame(outlier_summary).sort_values(
            "Outlier %", ascending=False
        )

        st.markdown("### 📊 Outlier Summary Table")
        st.dataframe(outlier_df, use_container_width=True,hide_index=True)

        # ---------------------------------------------
        # 📈 Boxplot Visualization
        # ---------------------------------------------
        st.markdown("### 📈 Boxplot Visualization")

                # ---------------------------------------------
        # 📈 Professional Annotated Boxplot (IQR)
        # ---------------------------------------------

       # ================================================
        st.subheader("📦 Distribution & Outlier Analysis")

        # Select numeric columns
        df_num = df.select_dtypes(include=np.number)

        selected_column = st.selectbox("Select Numeric Feature", df_num.columns)

        if selected_column:

            box_data = df_num[selected_column].dropna()

            # -------- IQR Calculation --------
            Q1 = np.percentile(box_data, 25)
            Q2 = np.percentile(box_data, 50)
            Q3 = np.percentile(box_data, 75)

            IQR = Q3 - Q1
            LL = Q1 - 1.5 * IQR
            UL = Q3 + 1.5 * IQR

            # -------- Box Plot --------
            fig, ax = plt.subplots(figsize=(10, 3))

            ax.boxplot(
                box_data,
                vert=False,
                patch_artist=True,
                boxprops=dict(facecolor='#C7D3F3', color='white', linewidth=2),
                medianprops=dict(color='#FFD700', linewidth=3),
                whiskerprops=dict(color='white', linewidth=2),
                capprops=dict(color='white', linewidth=2)
            )

            # Vertical lines
            ax.axvline(LL, linestyle='--', linewidth=2)
            ax.axvline(UL, linestyle='--', linewidth=2)

            # Annotations
            ax.text(Q1, 1.20, f"Q1: {Q1:.2f}", ha='center', color='white')
            ax.text(Q2, 1.12, f"Median: {Q2:.2f}", ha='center', fontweight='bold', color='#FFD700')
            ax.text(Q3, 1.20, f"Q3: {Q3:.2f}", ha='center', color='white')
            ax.text(LL, 1.04, f"LL: {LL:.2f}", ha='center', color='white')
            ax.text(UL, 1.04, f"UL: {UL:.2f}", ha='center', color='white')

            ax.set_title(f"Distribution of {selected_column}", color='white')
            ax.set_xlabel(selected_column, color='white')
            ax.set_yticks([])
            ax.tick_params(colors='white')
            ax.set_facecolor('none')
            fig.patch.set_alpha(0)

            plt.tight_layout()
            st.pyplot(fig)


                # ================================================
        st.subheader("📈 Skewness Analysis")

        # Skewness categorization
        def skew_tag(value):
            if abs(value) < 0.5:
                return 'Low skewed'
            elif abs(value) < 1:
                return 'Moderately skewed'
            else:
                return 'Highly skewed'

        skewness = df_num.skew()

        skew_summary = (
            skewness
            .reset_index()
            .rename(columns={'index': 'Feature', 0: 'Skewness'})
        )

        skew_summary['Skew_Category'] = skew_summary['Skewness'].apply(skew_tag)

        skew_summary = skew_summary.sort_values(by='Skewness', ascending=False)


        # 🎨 Color Function
        def color_skew_category(val):
            if val == 'Low skewed':
                return 'background-color: #2ecc71; color: white;'  # Green
            elif val == 'Moderately skewed':
                return 'background-color: #f39c12; color: white;'  # Orange
            else:
                return 'background-color: #e74c3c; color: white;'  # Red


        styled_df = (
            skew_summary.style
            .applymap(color_skew_category, subset=['Skew_Category'])
            .format({'Skewness': '{:.3f}'})
        )

        st.dataframe(
            styled_df,
            use_container_width=True,hide_index=True
        )
    # ====================================================
    # 🚀 Navigation Section
    # ====================================================
    st.markdown("---")

    col1, col2, col3 = st.columns([2,4,2])

    with col1:
        if st.button("⬅ Back", use_container_width=True):
            st.switch_page("main.py")

    with col3:
        if st.button("🚀 Next", use_container_width=True):
            st.switch_page("pages/2_mapping.py")