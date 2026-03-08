import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, boxcox
from sklearn.preprocessing import PowerTransformer, LabelEncoder, StandardScaler, MinMaxScaler
from theme import apply_theme

st.set_page_config(layout="wide")
apply_theme("ML_image.jpeg")
st.title("⚙ Advanced Preprocessing - Modular Workflow")

# ========================
# CHECK DATA
# ========================
if "cleaned_data" not in st.session_state or "target" not in st.session_state:
    st.warning("⚠ Please complete Cleaning first and define target.")
    st.stop()

df_full = st.session_state.cleaned_data.copy()
target_col = st.session_state.target

y = df_full[target_col].copy()
X = df_full.drop(columns=[target_col]).copy()

# ========================
# SESSION STATE INIT
# ========================
reset_flag = st.session_state.get("reset_advanced_preprocessing", True)
if reset_flag or "X_working" not in st.session_state:
#if "X_working" not in st.session_state:
    st.session_state.X_working = X.copy()
    st.session_state.y_working = y.copy()
    # Reset flag after initializing
    st.session_state["reset_advanced_preprocessing"] = False

X_working = st.session_state.X_working.copy()
y_working = st.session_state.y_working.copy()

# ========================
# HELPER FUNCTIONS
# ========================

def get_numeric_cols(df):
    return df.select_dtypes(include=["int64","float64","Int64"]).columns.tolist()

def get_categorical_cols(df):
    return df.select_dtypes(include=["object","category","string","bool"]).columns.tolist()

def sync_target_after_row_filter(X_new, y_current):
    return X_new, y_current.loc[X_new.index]

# ------------------------
# OUTLIER SUMMARY
# ------------------------
def outlier_summary(df, cols):

    summary = pd.DataFrame(
        index=cols,
        columns=["Min","Q1","Median","Q3","Max","Outliers(IQR)"]
    )
    for col in cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        Outlier_Count= len(outliers)

        summary.loc[col] = [
            df[col].min(),
            Q1,
            df[col].median(),
            Q3,
            df[col].max(),
            Outlier_Count
        ]

    return summary


# ------------------------
# SKEWNESS SUMMARY
# ------------------------
def skewness_summary(df, cols):

    summary = pd.DataFrame(
        index=cols,
        columns=[
            "Mean",
            "Median",
            "Std",
            "Skewness",
            "Skew Type",
            "Suggested Transformation"
        ]
    )

    for col in cols:

        data = df[col].dropna()

        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()

        skew_val = skew(data)

        if abs(skew_val) < 0.5:
            skew_type = "Symmetric"
            suggestion = "None"

        elif abs(skew_val) < 1:
            skew_type = "Moderate"
            suggestion = "Log / Sqrt"

        else:
            skew_type = "Highly Skewed"
            suggestion = "Log / BoxCox / YeoJohnson"

        summary.loc[col] = [
            mean_val,
            median_val,
            std_val,
            skew_val,
            skew_type,
            suggestion
        ]

    return summary


numeric_cols = get_numeric_cols(st.session_state.X_working)
categorical_cols = get_categorical_cols(st.session_state.X_working)

# =====================================================
# NUMERIC VARIABLES
# =====================================================

st.header("📊 Numeric Variables")

if numeric_cols:

    # ---------------------------------------
    # OUTLIER SUMMARY TABLE
    # ---------------------------------------
    st.subheader("📈 Outlier Summary")

    outlier_table = outlier_summary(X_working, numeric_cols)

    st.dataframe(outlier_table, use_container_width=True)


    # =================================================
    # OUTLIER TREATMENT
    # =================================================
    st.markdown("### ⚙ Outlier Treatment")

    selected_numeric_out = st.multiselect(
        "Select Columns for Outlier Treatment",
        numeric_cols
    )

    outlier_method_map = {}

    for col in selected_numeric_out:

        outlier_method_map[col] = st.selectbox(
            f"{col} ➜ Outlier Method",
            [
                "IQR Capping",
                "ZScore Capping",
                "Percentile Capping",
                "Remove Outliers"
            ],
            key=f"out_{col}"
        )

    if st.button("Apply Outlier Treatment"):

        X_new = X_working.copy()
        y_new = y_working.copy()

        for col, method in outlier_method_map.items():

            if method == "IQR Capping":

                Q1 = X_new[col].quantile(0.25)
                Q3 = X_new[col].quantile(0.75)

                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                X_new[col] = np.clip(X_new[col], lower, upper)

            elif method == "ZScore Capping":

                mean = X_new[col].mean()
                std = X_new[col].std()

                z = (X_new[col] - mean) / std

                X_new.loc[z > 3, col] = mean + 3 * std
                X_new.loc[z < -3, col] = mean - 3 * std

            elif method == "Percentile Capping":

                lower = np.percentile(X_new[col], 1)
                upper = np.percentile(X_new[col], 99)

                X_new[col] = np.clip(X_new[col], lower, upper)

            elif method == "Remove Outliers":

                Q1 = X_new[col].quantile(0.25)
                Q3 = X_new[col].quantile(0.75)

                IQR = Q3 - Q1

                X_new = X_new[
                    (X_new[col] >= Q1 - 1.5 * IQR) &
                    (X_new[col] <= Q3 + 1.5 * IQR)
                ]

                X_new, y_new = sync_target_after_row_filter(X_new, y_new)

        st.session_state.X_working = X_new
        st.session_state.y_working = y_new

        st.success("Outlier treatment applied")

        st.dataframe(outlier_summary(X_new, numeric_cols))


    # ---------------------------------------
    # SKEWNESS SUMMARY TABLE
    # ---------------------------------------
    st.subheader("📉 Skewness Summary")

    skew_table = skewness_summary(X_working, numeric_cols)

    st.dataframe(skew_table, use_container_width=True)


    # =================================================
    # SKEWNESS TREATMENT
    # =================================================

    st.markdown("### ⚙ Skewness Treatment")

    skew_cols = st.multiselect(
        "Select Columns for Skewness Correction",
        numeric_cols
    )

    skew_method_map = {}

    for col in skew_cols:

        skew_method_map[col] = st.selectbox(
            f"{col} ➜ Transformation",
            [
                "None",
                "Log1p",
                "Square Root",
                "Cube Root",
                "Box-Cox",
                "Yeo-Johnson"
            ],
            key=f"sk_{col}"
        )

    if st.button("Apply Skewness Treatment"):

        X_new = st.session_state.X_working.copy()

        for col, method in skew_method_map.items():

            try:

                if method == "Log1p":
                    X_new[col] = np.log1p(X_new[col])

                elif method == "Square Root":
                    X_new[col] = np.sqrt(np.clip(X_new[col],0,None))

                elif method == "Cube Root":
                    X_new[col] = np.cbrt(X_new[col])

                elif method == "Box-Cox":

                    if (X_new[col] > 0).all():
                        X_new[col], _ = boxcox(X_new[col])

                elif method == "Yeo-Johnson":

                    X_new[col] = PowerTransformer(
                        method="yeo-johnson"
                    ).fit_transform(X_new[[col]])

            except:
                pass

        st.session_state.X_working = X_new

        st.success("Skewness treatment applied")

        st.dataframe(skewness_summary(X_new, skew_cols))


    # =================================================
    # SCALING
    # =================================================

    st.markdown("### ⚙ Scaling")

    scaling_option = st.radio(
        "Select Scaling Method",
        [
            "None",
            "Standard Scaling",
            "MinMax Scaling"
        ]
    )

    if st.button("Apply Scaling"):

        X_new = st.session_state.X_working.copy()

        if scaling_option == "Standard Scaling":

            X_new[numeric_cols] = StandardScaler().fit_transform(
                X_new[numeric_cols]
            )

        elif scaling_option == "MinMax Scaling":

            X_new[numeric_cols] = MinMaxScaler().fit_transform(
                X_new[numeric_cols]
            )

        st.session_state.X_working = X_new

        st.success("Scaling applied")

        st.dataframe(X_new[numeric_cols].describe())


# =====================================================
# CATEGORICAL VARIABLES
# =====================================================

st.header("🔤 Categorical Variables")

if categorical_cols:

    cat_summary = pd.DataFrame({
        "Column": categorical_cols,
        "Unique Values":[X_working[col].nunique() for col in categorical_cols]
    })

    st.dataframe(cat_summary)

    st.markdown("### ⚙ Encoding")

    selected_cat = st.multiselect(
        "Select Columns for Encoding",
        categorical_cols
    )

    cat_method_map = {}

    for col in selected_cat:

        cat_method_map[col] = st.selectbox(
            f"{col} ➜ Encoding Method",
            [
                "Label Encoding",
                "One Hot Encoding",
                "Frequency Encoding"
            ]
        )

    if st.button("Apply Encoding"):

        X_new = st.session_state.X_working.copy()

        for col, method in cat_method_map.items():

            if method == "Label Encoding":

                X_new[col] = LabelEncoder().fit_transform(
                    X_new[col].astype(str)
                )

            elif method == "One Hot Encoding":

                dummies = pd.get_dummies(
                    X_new[col],
                    prefix=col
                )

                X_new = pd.concat(
                    [X_new.drop(columns=[col]), dummies],
                    axis=1
                )

            elif method == "Frequency Encoding":

                X_new[col] = X_new[col].map(
                    X_new[col].value_counts(normalize=True)
                )

        X_new = X_new.loc[:, ~X_new.columns.duplicated()]

        st.session_state.X_working = X_new

        st.success("Encoding applied")

        st.dataframe(X_new.head())


# =====================================================
# SAVE & NAVIGATION
# =====================================================

st.markdown("---")

col1,col2 = st.columns(2)

with col1:

    if st.button("⬅ Back to Cleaning"):

        st.switch_page("pages/3_cleaning.py")

with col2:

    if st.button("➡ Next: Model Selection"):

        final_df = pd.concat(
            [st.session_state.X_working,
             st.session_state.y_working],
            axis=1
        )

        # WITH EDA treatments
        st.session_state.preprocessed_data = final_df.copy()

        # WITHOUT EDA treatments
        st.session_state.raw_model_data = df_full.copy()

        st.switch_page("pages/5_model_selection.py")