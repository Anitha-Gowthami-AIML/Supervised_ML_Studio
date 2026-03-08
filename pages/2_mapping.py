import streamlit as st
import pandas as pd
import numpy as np
from theme import apply_theme

st.set_page_config(layout="wide")
apply_theme("ML_image.jpeg")

st.title("🗺 Variable Mapping & Role Assignment")

# =====================================================
# LOAD DATA SAFELY
# =====================================================

if "raw_data" not in st.session_state:
    st.warning("⚠ Please upload dataset first.")
    st.stop()

if "working_data" not in st.session_state:
    st.session_state.working_data = st.session_state.raw_data.copy()

df = st.session_state.working_data.copy()

# =====================================================
# TABS (LOGICAL ORDER)
# =====================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Dataset Preview",
    "⚙ Data Type Override",
    "🎯 Feature & Target Selection"
])

# =====================================================
# TAB 1 — PREVIEW
# =====================================================

with tab1:
    st.subheader("Dataset Snapshot")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    st.write("Shape:", df.shape)

    st.markdown("### Current Data Types")
    st.dataframe(
        pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.values
        }),
        use_container_width=True,
        hide_index=True
    )

# =====================================================
# TAB 2 — DATA TYPE OVERRIDE
# =====================================================

with tab2:

    st.subheader("⚙ Manual Data Type Override")

    column_to_convert = st.selectbox(
        "Select Column",
        df.columns
    )

    new_type = st.selectbox(
        "Convert To",
        ["int", "float", "bool", "category", "string", "datetime"]
    )

    if st.button("🔄 Apply Conversion"):

        try:
            if new_type == "int":
                df[column_to_convert] = pd.to_numeric(
                    df[column_to_convert], errors="coerce"
                ).astype("Int64")

            elif new_type == "float":
                df[column_to_convert] = pd.to_numeric(
                    df[column_to_convert], errors="coerce"
                )

            elif new_type == "bool":
                df[column_to_convert] = df[column_to_convert].astype("bool")

            elif new_type == "category":
                df[column_to_convert] = df[column_to_convert].astype("category")

            elif new_type == "string":
                df[column_to_convert] = df[column_to_convert].astype("string")

            elif new_type == "datetime":
                df[column_to_convert] = pd.to_datetime(
                    df[column_to_convert], errors="coerce"
                )

            st.session_state.working_data = df.copy()
            st.success(f"✅ {column_to_convert} converted to {new_type}")
            st.rerun()

        except Exception as e:
            st.error(f"Conversion Failed: {e}")

# =====================================================
# TAB 3 — FEATURE & TARGET SELECTION
# =====================================================

with tab3:

    st.subheader("🎯 Select Target Variable")

    # Preserve previous selection safely
    if "target" in st.session_state and st.session_state["target"] in df.columns:
        default_index = list(df.columns).index(st.session_state["target"])
    else:
        default_index = 0

    # IMPORTANT: selectbox must be OUTSIDE the if/else
    target = st.selectbox(
        "Choose Target Column",
        df.columns,
        index=default_index
    )

    # Save target immediately
    st.session_state["target"] = target

    st.subheader("📊 Select Feature Columns")

    feature_columns = st.multiselect(
        "Select Features",
        [col for col in df.columns if col != target],
        default=[
            col for col in df.columns
            if col != target
        ]
    )

    st.session_state["features"] = feature_columns

    if len(feature_columns) == 0:
        st.info("Please select at least one feature to continue.")
    else:
        temp_df = df[feature_columns + [target]]

        numeric_cols = temp_df[feature_columns].select_dtypes(
            include=["int64", "float64", "Int64"]
        ).columns.tolist()

        categorical_cols = temp_df[feature_columns].select_dtypes(
            include=["object", "category", "string", "bool"]
        ).columns.tolist()
        st.session_state["numeric_cols"] = numeric_cols
        st.session_state["categorical_cols"] = categorical_cols
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔢 Numeric Features")
            if numeric_cols:
                st.dataframe(
                    pd.DataFrame({"Numeric Columns": numeric_cols}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No Numeric Columns")

        with col2:
            st.markdown("### 🏷 Categorical Features")
            if categorical_cols:
                st.dataframe(
                    pd.DataFrame({"Categorical Columns": categorical_cols}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No Categorical Columns")
# =====================================================
# SAVE & CONTINUE
# =====================================================

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("⬅ Back to Upload"):
        st.switch_page("pages/1_Upload_data.py")

with col2:
    if st.button("💾 Save Mapping & Continue"):

        if "target" not in st.session_state or "numeric_cols" not in st.session_state or "categorical_cols" not in st.session_state:
            st.warning("⚠ Please select target and features first.")
        else:
            #final_df = df[st.session_state.features + [st.session_state.target]]
            selected_cols = list(dict.fromkeys(st.session_state.numeric_cols+st.session_state.categorical_cols + [st.session_state.target]))
            final_df = df[selected_cols]

            st.session_state.working_data = final_df.copy()
            st.session_state.cleaned_data = final_df.copy()

            st.success("Mapping Saved Successfully ✅")
            st.switch_page("pages/3_cleaning.py")

            