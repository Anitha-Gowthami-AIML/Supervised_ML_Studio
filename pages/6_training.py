import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import *
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR, SVC
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor
)

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.model_selection import GridSearchCV

from theme import apply_theme



# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(layout="wide")
apply_theme("ML_image.jpeg")

nav1, nav2, nav3 = st.columns([1,6,1])

with nav1:
    if st.button("🏠 Home", key="home_btn"):
        st.switch_page("Welcome_Page.py")

with nav3:
    if st.button("⬅ Back", key="back_btn"):
        st.switch_page("pages/5_model_selection.py")

st.title("🚀 ML Model Studio — Training, Diagnostics & Explainability")

# ==========================================
# LOAD DATA
# ==========================================

if "preprocessed_data" not in st.session_state:
    st.warning("⚠ Please finish preprocessing first.")
    st.stop()

if "target" not in st.session_state:
    st.warning("⚠ Target column not defined.")
    st.stop()

df = st.session_state.preprocessed_data.copy()
target = st.session_state.target

df = df.dropna().reset_index(drop=True)

X = df.drop(columns=[target])
y = df[target]

# ==========================================
# DETECT PROBLEM TYPE
# ==========================================

if y.dtype == "object" or y.nunique() <= 10:
    problem_type = "Classification"
else:
    problem_type = "Regression"

st.success(f"Detected Problem Type: **{problem_type}**")

# ==========================================
# MAIN TABS
# ==========================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
"⚙ Train Test Setup",
"🧠 Feature Selection",
"🤖 Model Training",
"📊 Diagnostics & Plots",
"🔍 SHAP Explainability"
])

# =================================================
# TAB 1 — TRAIN TEST SETUP
# =================================================

with tab1:

    st.subheader("Train Test Split Configuration")

    col1, col2, col3 = st.columns(3)

    # Test size
    with col1:
        test_size = st.slider(
            "Test Size (Split Ratio)",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05
        )

    # Random state
    with col2:
        random_state = st.number_input(
            "Random State",
            min_value=0,
            value=42
        )

    # Stratified sampling option
    with col3:

        stratify_option = False

        if problem_type == "Classification":

            stratify_choice = st.radio(
                "Sampling Method",
                ["Random Sampling", "Stratified Sampling"]
            )

            if stratify_choice == "Stratified Sampling":
                stratify_option = True

        else:
            st.info("Stratified sampling only applies to classification problems")

    st.divider()

    # Perform split
    if problem_type == "Classification" and stratify_option:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

    else:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

    # Display shapes
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Train Rows", X_train.shape[0])

    with col2:
        st.metric("Test Rows", X_test.shape[0])

    st.write("Train Shape:", X_train.shape)
    st.write("Test Shape:", X_test.shape)

    st.divider()

    # =================================================
    # Classification Distribution Check
    # =================================================

    if problem_type == "Classification":

        st.subheader("Target Distribution")

        dist_train = y_train.value_counts().rename("Train")
        dist_test = y_test.value_counts().rename("Test")

        dist_df = pd.concat([dist_train, dist_test], axis=1)

        st.dataframe(dist_df)

    st.divider()

    # =================================================
    # Show Sample Data
    # =================================================

    st.subheader("Train Data Sample")

    train_sample = pd.concat([X_train, y_train], axis=1)
    st.dataframe(train_sample.sample(min(10, len(train_sample))),
                 use_container_width=True)

    st.subheader("Test Data Sample")

    test_sample = pd.concat([X_test, y_test], axis=1)
    st.dataframe(test_sample.sample(min(10, len(test_sample))),
                 use_container_width=True)

    st.divider()

    # Save to session state
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test

    st.success("Train-Test split applied successfully")
# =================================================
# TAB 2 — FEATURE SELECTION
# =================================================

with tab2:

    st.subheader("Feature Engineering & Feature Selection")

    # -------------------------------------------------
    # Check Train-Test Split
    # -------------------------------------------------

    if "X_train" not in st.session_state:
        st.warning("Please run Train-Test Split first")
        st.stop()

    # -------------------------------------------------
    # Initialize Working Dataset
    # -------------------------------------------------

    if "X_train_work" not in st.session_state:
        st.session_state["X_train_work"] = st.session_state["X_train"].copy()

    X_current = st.session_state["X_train_work"]
    y_current = st.session_state["y_train"]

    st.write("Current Feature Count:", X_current.shape[1])

    # =================================================
    # 1️⃣ VIF MULTICOLLINEARITY REMOVAL
    # =================================================

    st.markdown("### 1️⃣ VIF Multicollinearity Removal (Iterative)")

    vif_threshold = st.slider(
        "Select VIF Threshold",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5
    )

    X_num = X_current.select_dtypes(include=np.number)

    # -------- VIF FUNCTIONS --------

    def calculate_vif(X):

        vif = pd.DataFrame()
        vif["Feature"] = X.columns
        vif["VIF"] = [
            variance_inflation_factor(X.values, i)
            for i in range(X.shape[1])
        ]

        return vif


    def remove_high_vif(X, threshold=10):

        X = X.copy()

        iteration_log = []
        iteration = 1

        while True:

            vif = calculate_vif(X)

            max_vif = vif["VIF"].max()

            if max_vif > threshold:

                feature_to_drop = (
                    vif.sort_values("VIF", ascending=False)
                    ["Feature"]
                    .iloc[0]
                )

                iteration_log.append({
                    "Iteration": iteration,
                    "Removed_Feature": feature_to_drop,
                    "VIF_at_Removal": round(max_vif, 3)
                })

                X = X.drop(columns=[feature_to_drop])

                iteration += 1

            else:
                break

        final_vif = calculate_vif(X)

        return X, final_vif, pd.DataFrame(iteration_log)

    # -------- Show Initial VIF --------

    if X_num.shape[1] > 1:

        st.subheader("Initial VIF Table")

        st.dataframe(calculate_vif(X_num))

        if st.button("Apply Iterative VIF Removal"):

            X_clean, final_vif, iteration_log = remove_high_vif(
                X_num,
                threshold=vif_threshold
            )

            removed_cols = iteration_log["Removed_Feature"].tolist() if not iteration_log.empty else []

            X_current = X_current.drop(columns=removed_cols, errors="ignore")

            st.session_state["X_train_work"] = X_current

            st.subheader("Features Removed")

            if not iteration_log.empty:
                st.dataframe(iteration_log)
            else:
                st.success("No features exceeded the VIF threshold")

            st.subheader("Final VIF")

            st.dataframe(final_vif)

            st.success(f"{len(removed_cols)} features removed")

    # =================================================
    # 2️⃣ MANUAL FEATURE DROP
    # =================================================

    st.markdown("### 2️⃣ Manual Feature Removal")

    drop_cols = st.multiselect(
        "Select Features to Drop",
        X_current.columns
    )

    if st.button("Apply Manual Drop"):

        X_current = X_current.drop(columns=drop_cols)

        st.session_state["X_train_work"] = X_current

        st.success(f"{len(drop_cols)} features removed")

    # =================================================
    # 3️⃣ VARIANCE THRESHOLD
    # =================================================

    st.markdown("### 3️⃣ Variance Threshold")

    threshold = st.slider(
        "Variance Threshold",
        0.0,
        1.0,
        0.0,
        0.01
    )

    if st.button("Apply Variance Filter"):

        X_num = X_current.select_dtypes(include=np.number)

        selector = VarianceThreshold(threshold)

        selector.fit(X_num)

        keep_cols = X_num.columns[selector.get_support()]

        removed_cols = list(set(X_num.columns) - set(keep_cols))

        X_current = X_current[keep_cols]

        st.session_state["X_train_work"] = X_current

        st.success(f"{len(removed_cols)} low variance features removed")

        st.dataframe(pd.DataFrame({
            "Removed Features": removed_cols
        }))

    # =================================================
    # 4️⃣ WALD TEST
    # =================================================

    st.markdown("### 4️⃣ Wald Test (Feature Significance)")

    if st.button("Run Wald Test"):

        X_sm = sm.add_constant(X_current)

        try:

            if problem_type == "Regression":

                model = sm.OLS(y_current, X_sm).fit()

            else:

                model = sm.Logit(y_current, X_sm).fit()

            wald_df = pd.DataFrame({
                "Feature": model.params.index,
                "Coefficient": model.params.values,
                "p_value": model.pvalues.values
            })

            st.dataframe(wald_df)

            st.markdown("### Significant Features")

            significant = wald_df[wald_df["p_value"] < 0.05]["Feature"].tolist()

            if "const" in significant:
                significant.remove("const")

            st.write(significant)

        except:

            st.error("Model could not be fit. Check feature types.")

    # =================================================
    # FINAL FEATURE SET
    # =================================================

    st.markdown("### Final Feature Set")

    X_current = st.session_state["X_train_work"]

    st.write("Total Selected Features:", X_current.shape[1])

    st.dataframe(pd.DataFrame({
        "Selected Features": X_current.columns
    }))

    # -------------------------------------------------
    # Save Selected Features
    # -------------------------------------------------

    st.session_state["X_train_selected"] = X_current

    selected_cols = X_current.columns

    st.session_state["X_test_selected"] = st.session_state["X_test"][selected_cols]

    st.success("Feature selection applied to both Train and Test data")
# =================================================
# TAB 3 — MODEL TRAINING PIPELINE
# =================================================

with tab3:

    st.header("Model Training Studio")

    X_train = st.session_state["X_train_selected"]
    X_test = st.session_state["X_test_selected"]
    y_train = st.session_state["y_train"]
    y_test = st.session_state["y_test"]

    # store feature list used for training
    st.session_state["training_columns"] = X_train.columns

    # align test data
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    if "trained_models" not in st.session_state:
        st.session_state["trained_models"] = {}

    if "leaderboard" not in st.session_state:
        st.session_state["leaderboard"] = pd.DataFrame()

    # =================================================
    # MODEL LIBRARY
    # =================================================

    if problem_type == "Regression":

        models = {

        "Linear Regression":LinearRegression(),
        "Ridge":Ridge(),
        "Lasso":Lasso(),
        "ElasticNet":ElasticNet(),

        "Decision Tree":DecisionTreeRegressor(),
        "Random Forest":RandomForestRegressor(),
        "Gradient Boosting":GradientBoostingRegressor(),

        "AdaBoost":AdaBoostRegressor(),
        #"Extra Trees":ExtraTreesRegressor(),

        "SVR":SVR()

        }

    else:

        models = {

        "Logistic Regression":LogisticRegression(max_iter=1000),

        "Decision Tree":DecisionTreeClassifier(),
        "Random Forest":RandomForestClassifier(),
        "Gradient Boosting":GradientBoostingClassifier(),

        "AdaBoost":AdaBoostClassifier(),
        #"Extra Trees":ExtraTreesClassifier(),

        "SVC":SVC(probability=True)

        }

    # =================================================
    # TAB STRUCTURE
    # =================================================

    train_tab,compare_tab,tune_tab,export_tab = st.tabs(

        ["Model Training","Model Comparison","Hyperparameter Tuning","Export"]

    )

# =================================================
# 1️⃣ MODEL TRAINING TAB
# =================================================

    with train_tab:

        st.subheader("Train Selected Model")

        selected_model = st.selectbox(
        "Select Algorithm",
        list(models.keys())
        )

        if st.button("Train Model"):

            model = models[selected_model]

            model.fit(X_train,y_train)

            st.session_state["trained_models"][selected_model] = model

            train_pred = model.predict(X_train)
            #test_pred = model.predict(X_test)
            X_test_aligned = X_test.reindex(columns=X_train.columns, fill_value=0)
            test_pred = model.predict(X_test_aligned)

            if problem_type == "Regression":

                r2_train = r2_score(y_train,train_pred)
                r2_test = r2_score(y_test,test_pred)

                n = X_train.shape[0]
                p = X_train.shape[1]

                adj_r2 = 1 - (1-r2_test)*(n-1)/(n-p-1)

                mae = mean_absolute_error(y_test,test_pred)
                mse = mean_squared_error(y_test,test_pred)
                rmse = np.sqrt(mse)

                metrics = pd.DataFrame({

                "Metric":["R2 Train","R2 Test","Adj R2","MAE","MSE","RMSE"],

                "Value":[
                r2_train,
                r2_test,
                adj_r2,
                mae,
                mse,
                rmse
                ]

                })

                st.dataframe(metrics)

                new_row = pd.DataFrame([{

                "Model":selected_model,
                "R2_Train":r2_train,
                "R2_Test":r2_test,
                "Adj_R2":adj_r2,
                "MAE":mae,
                "MSE":mse,
                "RMSE":rmse

                }])

                st.session_state["leaderboard"] = pd.concat(
                [st.session_state["leaderboard"],new_row],
                ignore_index=True
                )

            else:
 
                                # Predictions
                train_pred = model.predict(X_train)
                X_test_aligned = X_test.reindex(columns=X_train.columns, fill_value=0)
                test_pred = model.predict(X_test_aligned)

                # Metrics for Train
                train_acc = accuracy_score(y_train, train_pred)
                train_precision = precision_score(y_train, train_pred, average="weighted")
                train_recall = recall_score(y_train, train_pred, average="weighted")
                train_f1 = f1_score(y_train, train_pred, average="weighted")

                # Metrics for Test
                test_acc = accuracy_score(y_test, test_pred)
                test_precision = precision_score(y_test, test_pred, average="weighted")
                test_recall = recall_score(y_test, test_pred, average="weighted")
                test_f1 = f1_score(y_test, test_pred, average="weighted")

                st.subheader("Model Performance")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Train Metrics")
                    st.metric("Accuracy", round(train_acc,3))
                    st.metric("Precision", round(train_precision,3))
                    st.metric("Recall", round(train_recall,3))
                    st.metric("F1 Score", round(train_f1,3))

                with col2:
                    st.markdown("### Test Metrics")
                    st.metric("Accuracy", round(test_acc,3))
                    st.metric("Precision", round(test_precision,3))
                    st.metric("Recall", round(test_recall,3))
                    st.metric("F1 Score", round(test_f1,3))

with compare_tab:

    st.subheader("Compare All Models")

    if st.button("Run Model Comparison"):

        results = []

        for name, model in models.items():

            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)

            X_test_aligned = X_test.reindex(columns=X_train.columns, fill_value=0)
            test_pred = model.predict(X_test_aligned)

            if problem_type == "Regression":

                r2_train = r2_score(y_train, train_pred)
                r2_test = r2_score(y_test, test_pred)

                mse = mean_squared_error(y_test, test_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, test_pred)

                n = X_train.shape[0]
                p = X_train.shape[1]

                adj_r2 = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)

                results.append({

                    "Model": name,
                    "R2 Train": r2_train,
                    "R2 Test": r2_test,
                    "Adj R2": adj_r2,
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse

                })

            else:

                acc_train = accuracy_score(y_train, train_pred)
                acc_test = accuracy_score(y_test, test_pred)

                precision = precision_score(y_test, test_pred, average="weighted")
                recall = recall_score(y_test, test_pred, average="weighted")
                f1 = f1_score(y_test, test_pred, average="weighted")

                results.append({

                    "Model": name,
                    "Accuracy Train": acc_train,
                    "Accuracy Test": acc_test,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1

                })

        # =========================
        # AFTER LOOP
        # =========================

        df_results = pd.DataFrame(results)

        st.session_state["leaderboard"] = df_results

        if problem_type == "Regression":

            st.dataframe(df_results.sort_values("R2 Test", ascending=False))

            fig, ax = plt.subplots(figsize=(14,7))

            df_results.plot(
                x="Model",
                y=["R2 Train","R2 Test"],
                kind="bar",
                ax=ax,
                colormap="Set3"
            )

        else:

            st.dataframe(df_results.sort_values("Accuracy Test", ascending=False))

            fig, ax = plt.subplots(figsize=(14,7))

            df_results.plot(
                x="Model",
                y=["Accuracy Train","Accuracy Test"],
                kind="bar",
                ax=ax,
                colormap="Set2"
            )

        plt.xticks(rotation=45)
        st.pyplot(fig)         
# =================================================
# 3️⃣ HYPERPARAMETER TUNING
# =================================================

    
# =====================================================
# TAB — HYPERPARAMETER TUNING
# =====================================================

    with tune_tab:

        st.subheader("Hyperparameter Tuning (GridSearchCV)")

        X_train = st.session_state["X_train_selected"]
        X_test = st.session_state["X_test_selected"]
        y_train = st.session_state["y_train"]
        y_test = st.session_state["y_test"]

        random_state = 42

        if "tuned_models" not in st.session_state:
            st.session_state["tuned_models"] = {}

        # =====================================================
        # MODEL SELECTION
        # =====================================================

        if problem_type == "Regression":

            model_options = [
                "Linear Regression",
                "Ridge",
                "Lasso",
                "ElasticNet",
                "Decision Tree Regressor",
                "Random Forest Regressor",
                "Gradient Boosting Regressor",
                "SVR"
            ]

        else:

            model_options = [
                "Logistic Regression",
                "Decision Tree Classifier",
                "Random Forest Classifier",
                "Gradient Boosting Classifier",
                "SVC"
            ]

        selected_models = st.multiselect(
            "Select Models to Tune",
            model_options,
            default=model_options[:2]
        )

        # =====================================================
        # PARAMETER GRIDS
        # =====================================================

        param_grids = {

            "Ridge":{
                "alpha":[0.01,0.1,1,10]
            },

            "Lasso":{
                "alpha":[0.001,0.01,0.1,1]
            },

            "ElasticNet":{
                "alpha":[0.01,0.1,1],
                "l1_ratio":[0.2,0.5,0.8]
            },

            "Decision Tree Regressor":{
                "max_depth":[None,5,10],
                "min_samples_split":[2,5,10]
            },

            "Random Forest Regressor":{
                "n_estimators":[100,200],
                "max_depth":[None,5,10]
            },

            "Gradient Boosting Regressor":{
                "n_estimators":[100,200],
                "learning_rate":[0.01,0.1]
            },

            "SVR":{
                "C":[0.1,1,10],
                "kernel":["rbf","linear"]
            },

            "Logistic Regression":{
                "C":[0.01,0.1,1,10]
            },

            "Decision Tree Classifier":{
                "max_depth":[None,5,10],
                "min_samples_split":[2,5,10]
            },

            "Random Forest Classifier":{
                "n_estimators":[100,200],
                "max_depth":[None,5,10]
            },

            "Gradient Boosting Classifier":{
                "n_estimators":[100,200],
                "learning_rate":[0.01,0.1]
            },

            "SVC":{
                "C":[0.1,1,10],
                "kernel":["rbf","linear"]
            }

        }

        # =====================================================
        # RUN GRID SEARCH
        # =====================================================

        if st.button("🚀 Run Hyperparameter Tuning", key="gridsearch_run"):

            results = []

            progress = st.progress(0)

            for i, model_name in enumerate(selected_models):

                st.write(f"### Running GridSearch for {model_name}")

                # -------------------------------------------------
                # MODEL INITIALIZATION
                # -------------------------------------------------

                if model_name == "Linear Regression":
                    model = LinearRegression()
                    scoring = "r2"

                elif model_name == "Ridge":
                    model = Ridge()
                    scoring = "r2"

                elif model_name == "Lasso":
                    model = Lasso()
                    scoring = "r2"

                elif model_name == "ElasticNet":
                    model = ElasticNet()
                    scoring = "r2"

                elif model_name == "Decision Tree Regressor":
                    model = DecisionTreeRegressor(random_state=random_state)
                    scoring = "r2"

                elif model_name == "Random Forest Regressor":
                    model = RandomForestRegressor(random_state=random_state)
                    scoring = "r2"

                elif model_name == "Gradient Boosting Regressor":
                    model = GradientBoostingRegressor(random_state=random_state)
                    scoring = "r2"

                elif model_name == "SVR":
                    model = SVR()
                    scoring = "r2"

                elif model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000, random_state=random_state)
                    scoring = "accuracy"

                elif model_name == "Decision Tree Classifier":
                    model = DecisionTreeClassifier(random_state=random_state)
                    scoring = "accuracy"

                elif model_name == "Random Forest Classifier":
                    model = RandomForestClassifier(random_state=random_state)
                    scoring = "accuracy"

                elif model_name == "Gradient Boosting Classifier":
                    model = GradientBoostingClassifier(random_state=random_state)
                    scoring = "accuracy"

                elif model_name == "SVC":
                    model = SVC(probability=True, random_state=random_state)
                    scoring = "accuracy"

                # -------------------------------------------------
                # GRID SEARCH
                # -------------------------------------------------

                grid = GridSearchCV(
                    model,
                    param_grids.get(model_name,{}),
                    cv=5,
                    scoring=scoring,
                    n_jobs=-1
                )

                grid.fit(X_train,y_train)

                best_model = grid.best_estimator_

                st.session_state["tuned_models"][model_name] = best_model

                #y_pred = best_model.predict(X_test)
                X_test_aligned = X_test.reindex(columns=X_train.columns, fill_value=0)
                y_pred = best_model.predict(X_test_aligned)

                # -------------------------------------------------
                # METRICS
                # -------------------------------------------------

                if problem_type == "Regression":

                    r2 = r2_score(y_test,y_pred)

                    adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_train.shape[1] - 1)

                    results.append({
                        "Model":model_name,
                        "Best Parameters":str(grid.best_params_),
                        "R2":round(r2,4),
                        "Adjusted R2":round(adj_r2,4),
                        "MAE":round(mean_absolute_error(y_test,y_pred),4),
                        "MSE":round(mean_squared_error(y_test,y_pred),4),
                        "RMSE":round(np.sqrt(mean_squared_error(y_test,y_pred)),4)
                    })

                else:

                    results.append({
                        "Model":model_name,
                        "Best Parameters":str(grid.best_params_),
                        "Accuracy":round(accuracy_score(y_test,y_pred),4),
                        "Precision":round(precision_score(y_test,y_pred,average="weighted"),4),
                        "Recall":round(recall_score(y_test,y_pred,average="weighted"),4),
                        "F1 Score":round(f1_score(y_test,y_pred,average="weighted"),4)
                    })

                progress.progress((i+1)/len(selected_models))

            # =====================================================
            # RESULTS TABLE
            # =====================================================

            results_df = pd.DataFrame(results)

            st.markdown("## Tuned Model Performance Leaderboard")

            st.dataframe(
                results_df.sort_values(results_df.columns[2],ascending=False),
                use_container_width=True
            )

            # =====================================================
            # DOWNLOAD RESULTS
            # =====================================================

            csv = results_df.to_csv(index=False)

            st.download_button(
                "Download GridSearch Results CSV",
                csv,
                file_name="gridsearch_results.csv",
                key="download_gridsearch"
            )
# =================================================
# 4️⃣ EXPORT TAB
# =================================================

    with export_tab:

        st.subheader("Export Artifacts")

        # ==========================
        # SAVE MODEL
        # ==========================

        model_names = list(st.session_state["trained_models"].keys())

        if len(model_names)>0:

            selected_export_model = st.selectbox(
            "Select Model to Save",
            model_names
            )

            if st.button("Save Model", key="export_model_btn"):

                model = st.session_state["trained_models"][selected_export_model]

                joblib.dump(model,"trained_model.pkl")

                with open("trained_model.pkl","rb") as f:

                    st.download_button(

                    "Download Model",
                    f,
                    file_name="trained_model.pkl"

                    )

        # ==========================
        # SAVE DATASET
        # ==========================

        if st.button("Download Train Dataset"):

            csv = X_train.to_csv(index=False).encode()

            st.download_button(

            "Download CSV",
            csv,
            "train_dataset.csv"

            )

        # ==========================
        # SAVE LEADERBOARD
        # ==========================

        if not st.session_state["leaderboard"].empty:

            csv = st.session_state["leaderboard"].to_csv(index=False).encode()

            st.download_button(

            "Download Model Results",
            csv,
            "model_results.csv"

            )
    # =================================================
# TAB 4 — DIAGNOSTICS
# =================================================

with tab4:

    st.subheader("Model Diagnostics")
    if "trained_models" not in st.session_state:
        st.session_state["trained_models"] = {}

    if ("trained_models" not in st.session_state
    or len(st.session_state["trained_models"]) == 0
):
        st.info("Train at least one model in Model Training tab.")
        st.stop()

        st.info("Train models first in Model Training tab")

    else:

        model_name = st.selectbox(
            "Select Model",
            list(st.session_state["trained_models"].keys())
        )

        model = st.session_state["trained_models"][model_name]

        X_test = st.session_state["X_test_selected"]
        y_test = st.session_state["y_test"]

        y_pred = model.predict(X_test)

        # =================================================
        # CLASSIFICATION DIAGNOSTICS
        # =================================================

        if problem_type == "Classification":

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_test,y_pred)

            fig,ax = plt.subplots()
            sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            st.info("""
Interpretation:

• Diagonal values → Correct predictions  
• Off diagonal → Misclassification  
• Higher diagonal values indicate better model performance
            """)

            # ------------------------------------------------
            # Classification Report
            # ------------------------------------------------

            st.subheader("Classification Report")

            report = classification_report(y_test,y_pred,output_dict=True)

            st.dataframe(pd.DataFrame(report).transpose())

            st.info("""
Precision → How many predicted positives were correct  
Recall → How many actual positives were captured  
F1 Score → Harmonic mean of Precision and Recall
                        """)
            # ------------------------------------------------
            # FEATURE IMPORTANCE
            # ------------------------------------------------

            st.subheader("Feature Importance")

            try:

                if hasattr(model,"feature_importances_"):

                    importance = model.feature_importances_

                elif hasattr(model,"coef_"):

                    importance = np.abs(model.coef_)

                else:
                    raise Exception()

                fi = pd.DataFrame({
                "Feature":X_test.columns,
                "Importance":importance
                }).sort_values("Importance",ascending=False)

                fig,ax = plt.subplots()

                sns.barplot(
                data=fi.head(15),
                x="Importance",
                y="Feature",
                ax=ax
                )

                st.pyplot(fig)

                st.info("Shows which variables most influence predictions.")

            except:

                st.warning("Feature importance not available for this model.")
            # ------------------------------------------------
            # ROC Curve
            # ------------------------------------------------

            st.subheader("ROC Curve")

            try:

                y_prob = model.predict_proba(X_test)[:,1]

                fpr,tpr,_ = roc_curve(y_test,y_prob)

                auc_score = roc_auc_score(y_test,y_prob)

                fig,ax = plt.subplots()

                ax.plot(fpr,tpr,label=f"AUC = {auc_score:.3f}")
                ax.plot([0,1],[0,1],"--")

                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend()

                st.pyplot(fig)

                st.info("""
ROC Interpretation:

AUC = 0.5 → Random model  
AUC > 0.7 → Good  
AUC > 0.8 → Strong model
                """)

            except:

                st.warning("ROC not available for this model")

            # ------------------------------------------------
            # Precision Recall Curve
            # ------------------------------------------------

            try:

                st.subheader("Precision Recall Curve")

                precision,recall,_ = precision_recall_curve(y_test,y_prob)

                fig,ax = plt.subplots()

                ax.plot(recall,precision)

                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")

                st.pyplot(fig)

                st.info("""
Precision Recall Curve:

Useful when dataset is imbalanced.  
Higher area under curve indicates better performance.
                """)

            except:
                pass

            # ------------------------------------------------
            # Probability Distribution
            # ------------------------------------------------

            try:

                st.subheader("Prediction Probability Distribution")

                fig,ax = plt.subplots()

                sns.histplot(y_prob,kde=True,ax=ax)

                st.pyplot(fig)

                st.info("""
If probabilities cluster near 0 or 1 → strong model confidence  
If probabilities cluster around 0.5 → uncertain predictions
                """)

            except:
                pass

        # =================================================
        # REGRESSION DIAGNOSTICS
        # =================================================

        else:

            residuals = y_test - y_pred

            # ------------------------------------------------
            # Error Metrics
            # ------------------------------------------------

            st.subheader("Error Metrics")

            metrics = pd.DataFrame({

                "Metric":[
                "R2",
                "MSE",
                "MAE",
                "RMSE"
                ],

                "Value":[
                r2_score(y_test,y_pred),
                mean_squared_error(y_test,y_pred),
                mean_absolute_error(y_test,y_pred),
                np.sqrt(mean_squared_error(y_test,y_pred))
                ]

            })

            st.dataframe(metrics)

            # ------------------------------------------------
            # Actual vs Predicted
            # ------------------------------------------------

            st.subheader("Actual vs Predicted")

            fig,ax = plt.subplots()

            sns.scatterplot(x=y_test,y=y_pred,ax=ax)

            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")

            st.pyplot(fig)

            st.info("""
Points closer to diagonal line indicate better prediction accuracy.
            """)

            # ------------------------------------------------
            # Residual Plot
            # ------------------------------------------------

            st.subheader("Residual Plot")

            fig,ax = plt.subplots()

            sns.scatterplot(x=y_pred,y=residuals,ax=ax)

            ax.axhline(0,color="red")

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residual")

            st.pyplot(fig)

            st.info("""
Residuals should be randomly scattered around zero.  
Patterns indicate model bias.
            """)

            # ------------------------------------------------
            # Residual Distribution
            # ------------------------------------------------

            st.subheader("Residual Distribution")

            fig,ax = plt.subplots()

            sns.histplot(residuals,kde=True,ax=ax)

            st.pyplot(fig)

            st.info("""
Residuals should approximately follow normal distribution.
            """)

            # ------------------------------------------------
            # QQ Plot
            # ------------------------------------------------

            st.subheader("QQ Plot")

            fig = plt.figure()

            stats.probplot(residuals,dist="norm",plot=plt)

            st.pyplot(fig)

            st.info("""
If points lie on straight line → residuals normally distributed.
            """)

            # ------------------------------------------------
            # Cook's Distance
            # ------------------------------------------------

            try:

                st.subheader("Cook's Distance")

                X = st.session_state["X_train_selected"]

                X_const = sm.add_constant(X)

                model_sm = sm.OLS(
                    st.session_state["y_train"],
                    X_const
                ).fit()

                influence = model_sm.get_influence()

                cooks = influence.cooks_distance[0]

                fig,ax = plt.subplots()

                ax.stem(cooks)

                st.pyplot(fig)

                st.info("""
Large Cook's Distance values indicate influential observations
that heavily affect the regression model.
                """)

            except:
                pass
        # ------------------------------------------------
        # FEATURE IMPORTANCE
        # ------------------------------------------------

        st.subheader("Feature Importance")

        try:

            if hasattr(model,"feature_importances_"):

                importance = model.feature_importances_

            elif hasattr(model,"coef_"):

                importance = np.abs(model.coef_)

            else:
                raise Exception()

            fi = pd.DataFrame({
            "Feature":X_test.columns,
            "Importance":importance
            }).sort_values("Importance",ascending=False)

            fig,ax = plt.subplots()

            sns.barplot(
            data=fi.head(15),
            x="Importance",
            y="Feature",
            ax=ax
            )

            st.pyplot(fig)

            st.info("Shows which variables most influence predictions.")

        except:

            st.warning("Feature importance not available for this model.")
# =================================================
# TAB 5 — SHAP EXPLAINABILITY
# =================================================

with tab5:

    st.subheader("Model Explainability (SHAP)")

    if "trained_models" not in st.session_state:

        st.info("Train models first")

    else:

        model_name = st.selectbox(
            "Select Model for SHAP",
            list(st.session_state["trained_models"].keys())
        )

        model = st.session_state["trained_models"][model_name]

        # Use feature-selected data
        X_train = st.session_state.get("X_train_selected", st.session_state["X_train"])
        X_test = st.session_state.get("X_test_selected", st.session_state["X_test"])

        try:

            # =================================================
            # CREATE SHAP EXPLAINER
            # =================================================

            explainer = shap.Explainer(model, X_train)

            shap_values = explainer(X_test)

            st.success("SHAP values computed successfully")

            # =================================================
            # GLOBAL FEATURE IMPORTANCE
            # =================================================

            st.subheader("Global Feature Importance")

            fig1, ax1 = plt.subplots()

            shap.plots.bar(shap_values, show=False)

            st.pyplot(fig1)

            st.info("""
Interpretation:

• Higher SHAP value → Feature contributes more to predictions  
• Features at the top have the strongest global impact  
• Useful for understanding **which variables drive the model**
            """)

            # =================================================
            # SHAP SUMMARY PLOT
            # =================================================

            st.subheader("SHAP Summary Plot")

            fig2, ax2 = plt.subplots()

            shap.summary_plot(
                shap_values,
                X_test,
                show=False
            )

            st.pyplot(fig2)

            st.info("""
Interpretation:

• Each dot represents one observation  
• Red → High feature value  
• Blue → Low feature value  

If red dots appear on the right side:
→ Higher feature values increase predictions.

If red dots appear on the left:
→ Higher feature values decrease predictions.
            """)

            # =================================================
            # LOCAL EXPLANATION
            # =================================================

            st.subheader("Local Prediction Explanation")

            row_index = st.slider(
                "Select observation to explain",
                0,
                len(X_test) - 1,
                0
            )

            st.write("Selected Observation")

            st.dataframe(X_test.iloc[[row_index]])

            fig3, ax3 = plt.subplots()

            shap.plots.waterfall(
                shap_values[row_index],
                show=False
            )

            st.pyplot(fig3)

            st.info("""
Interpretation:

• Base value → Average prediction of the model  
• Red bars → Features increasing prediction  
• Blue bars → Features decreasing prediction  

This explains **why the model predicted this value for this observation**.
            """)

        except Exception as e:

            st.error(f"SHAP failed for this model: {e}")

            st.info("""
Some models (especially certain pipelines or unsupported estimators) 
may not work with SHAP's default explainer.

Try using:

• Tree-based models (RandomForest, XGBoost, LightGBM)
• Linear models (LinearRegression, LogisticRegression)
            """)
        # =================================================
        # MODEL SAVE
        # =================================================

        st.subheader("Save Trained Model")

        if st.button("Save Model", key="save_model_btn"):

            joblib.dump(model, "model.pkl")

            with open("model.pkl", "rb") as f:

                st.download_button(
                    "Download Model",
                    f,
                    file_name="model.pkl"
                )

        st.info("""
Model Download:

You can download the trained model as **model.pkl**  
This file can be reused for:

• Production deployment  
• API serving  
• Batch predictions
        """)