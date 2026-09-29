"""
Supplier Risk Scoring Analytics - Streamlit Dashboard
=====================================================
AI-powered supplier risk assessment and explainable decision support.
Designed for Round 3 Hackathon Demonstration.
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Ensure local imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.preprocessing import SupplierDataPreprocessor, FEATURE_ORDER, COUNT_FEATURES, load_preprocessor
from utils.explainability import get_feature_importance_table, generate_analytical_insights, generate_supplier_recommendation
from utils.visualization import (
    plot_risk_probability_donut,
    plot_feature_importance_bar,
    plot_supplier_vs_benchmark,
    plot_confusion_matrix_heatmap
)

# Page Configuration
st.set_page_config(
    page_title="Supplier Risk Scoring Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Visual Excellence
st.markdown("""
    <style>
    /* Main Theme Overrides */
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Header Styling */
    .title-text {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .subtitle-text {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 25px;
    }
    
    /* Risk Cards */
    .risk-card-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.3) 100%);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.2);
    }
    .risk-card-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(4, 120, 87, 0.3) 100%);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.2);
    }
    .risk-title {
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    
    /* Metric Badge */
    .metric-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        font-weight: 600;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E293B;
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_pipeline():
    """Load Random Forest Model & Preprocessing Pipeline with caching."""
    model_paths = [
        "models/supplier_risk_random_forest.pkl",
        "supplier_risk_random_forest.pkl"
    ]
    model = None
    for path in model_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            break
            
    if model is None:
        st.error("❌ Model file 'supplier_risk_random_forest.pkl' not found! Please check models directory.")
        st.stop()

    pipeline_paths = [
        "models/preprocessing_pipeline.pkl",
        "preprocessing_pipeline.pkl"
    ]
    pipeline_path = next((p for p in pipeline_paths if os.path.exists(p)), None)
    preprocessor = SupplierDataPreprocessor(pipeline_path=pipeline_path)

    return model, preprocessor


def main():
    # Load Model and Preprocessor
    model, preprocessor = load_model_and_pipeline()
    benchmark_medians = preprocessor.medians

    # App Header
    st.markdown('<p class="title-text">🛡️ Supplier Risk Scoring Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">AI-powered supplier risk assessment and explainable decision support</p>', unsafe_allow_html=True)

    # Main Tabs
    tab_single, tab_batch, tab_about = st.tabs([
        "🎯 Single Supplier Prediction",
        "🧪 Unseen Jury Data Batch Evaluation",
        "ℹ️ System & Preprocessing Specs"
    ])

    # ----------------------------------------------------
    # TAB 1: Single Supplier Assessment
    # ----------------------------------------------------
    with tab_single:
        # Sidebar Controls
        st.sidebar.header("📋 Supplier Input Parameters")
        st.sidebar.markdown("Enter unseen supplier parameters for evaluation:")

        # Quick Preset Buttons
        st.sidebar.markdown("##### ⚡ Quick Preset Test Cases")
        col_p1, col_p2 = st.sidebar.columns(2)
        
        default_vals = {
            "Financial_Stability_Score": 65.0,
            "On_Time_Delivery_Rate": 85.0,
            "Defect_Rate": 2.5,
            "Geopolitical_Risk_Index": 25,
            "Lead_Time_Days": 20,
            "Alternative_Suppliers_Available": 3,
            "Contract_Length_Months": 24,
            "Environmental_Compliance": 90.0,
            "Previous_Disruptions": 0
        }

        if col_p1.button("🟢 Low Risk Preset"):
            st.session_state["inputs"] = {
                "Financial_Stability_Score": 82.5,
                "On_Time_Delivery_Rate": 94.0,
                "Defect_Rate": 0.8,
                "Geopolitical_Risk_Index": 15,
                "Lead_Time_Days": 12,
                "Alternative_Suppliers_Available": 4,
                "Contract_Length_Months": 36,
                "Environmental_Compliance": 96.0,
                "Previous_Disruptions": 0
            }
        if col_p2.button("🔴 High Risk Preset"):
            st.session_state["inputs"] = {
                "Financial_Stability_Score": 30.0,
                "On_Time_Delivery_Rate": 60.0,
                "Defect_Rate": 9.5,
                "Geopolitical_Risk_Index": 72,
                "Lead_Time_Days": 50,
                "Alternative_Suppliers_Available": 0,
                "Contract_Length_Months": 6,
                "Environmental_Compliance": 40.0,
                "Previous_Disruptions": 3
            }

        curr = st.session_state.get("inputs", default_vals)

        # Form Inputs
        fin_stab = st.sidebar.number_input("Financial Stability Score (0-100)", min_value=0.0, max_value=100.0, value=float(curr["Financial_Stability_Score"]), step=1.0)
        on_time = st.sidebar.number_input("On-Time Delivery Rate (%)", min_value=0.0, max_value=100.0, value=float(curr["On_Time_Delivery_Rate"]), step=1.0)
        defect_rate = st.sidebar.number_input("Defect Rate (%)", min_value=0.0, max_value=50.0, value=float(curr["Defect_Rate"]), step=0.1)
        geo_risk = st.sidebar.slider("Geopolitical Risk Index (0-100)", min_value=0, max_value=100, value=int(curr["Geopolitical_Risk_Index"]))
        lead_time = st.sidebar.number_input("Lead Time (Days)", min_value=0, max_value=180, value=int(curr["Lead_Time_Days"]), step=1)
        alt_supp = st.sidebar.number_input("Alternative Suppliers Available", min_value=0, max_value=20, value=int(curr["Alternative_Suppliers_Available"]), step=1)
        contract_len = st.sidebar.number_input("Contract Length (Months)", min_value=1, max_value=120, value=int(curr["Contract_Length_Months"]), step=1)
        env_comp = st.sidebar.number_input("Environmental Compliance Score", min_value=0.0, max_value=150.0, value=float(curr["Environmental_Compliance"]), step=1.0, help="Values > 100 will be automatically capped at 100 during preprocessing.")
        prev_disr = st.sidebar.number_input("Previous Disruptions (Count)", min_value=0, max_value=20, value=int(curr["Previous_Disruptions"]), step=1)

        raw_input_dict = {
            "Financial_Stability_Score": fin_stab,
            "On_Time_Delivery_Rate": on_time,
            "Defect_Rate": defect_rate,
            "Geopolitical_Risk_Index": geo_risk,
            "Lead_Time_Days": lead_time,
            "Alternative_Suppliers_Available": alt_supp,
            "Contract_Length_Months": contract_len,
            "Environmental_Compliance": env_comp,
            "Previous_Disruptions": prev_disr
        }

        # Input Validation Warning if out of range
        validation_msgs = []
        if env_comp > 100:
            validation_msgs.append("⚠️ Environmental Compliance > 100 entered: Preprocessing pipeline will cap this to 100.")
        if fin_stab < 0 or fin_stab > 100:
            validation_msgs.append("❌ Invalid Financial Stability Score range.")

        for msg in validation_msgs:
            st.sidebar.warning(msg)

        predict_btn = st.sidebar.button("⚡ Predict Supplier Risk", type="primary", use_container_width=True)

        # Execution Block
        # 1. Preprocess raw input
        df_raw = pd.DataFrame([raw_input_dict])
        df_processed = preprocessor.transform(df_raw)

        # 2. Inference
        pred_class = int(model.predict(df_processed)[0])
        probabilities = model.predict_proba(df_processed)[0]
        proba_low = float(probabilities[0])
        proba_high = float(probabilities[1])

        # 3. Feature Importances for this prediction
        df_imp = get_feature_importance_table(model, FEATURE_ORDER, raw_input_dict, benchmark_medians)
        top_driver = df_imp.iloc[0]["Feature"]

        # Layout Columns: Top KPI Summary
        col_res, col_gauge = st.columns([1.2, 1.0])

        with col_res:
            if pred_class == 1:
                st.markdown(f"""
                    <div class="risk-card-high">
                        <div style="color: #FCA5A5; font-weight: 700;">PREDICTED SUPPLIER RISK CLASS</div>
                        <div class="risk-title" style="color: #EF4444;">🚨 HIGH RISK</div>
                        <div style="color: #F8FAFC; font-size: 1.1rem;">Risk Probability: <b>{proba_high*100:.1f}%</b></div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="risk-card-low">
                        <div style="color: #6EE7B7; font-weight: 700;">PREDICTED SUPPLIER RISK CLASS</div>
                        <div class="risk-title" style="color: #10B981;">✅ LOW RISK</div>
                        <div style="color: #F8FAFC; font-size: 1.1rem;">Low Risk Confidence: <b>{proba_low*100:.1f}%</b></div>
                    </div>
                """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Low Risk Prob</div>
                    <div class="metric-val" style="color: #10B981;">{proba_low*100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            m2.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">High Risk Prob</div>
                    <div class="metric-val" style="color: #EF4444;">{proba_high*100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            m3.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Top Influential Driver</div>
                    <div class="metric-val" style="font-size: 1.0rem; color: #38BDF8;">{top_driver}</div>
                </div>
            """, unsafe_allow_html=True)

        with col_gauge:
            fig_donut = plot_risk_probability_donut(proba_low, proba_high)
            st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("---")

        # Explainability & Insights Section
        st.subheader("🔍 Model Explainability & Feature Importances")
        st.caption("Based on trained Random Forest feature_importances_ and input attribute values.")

        exp_col1, exp_col2 = st.columns([1.1, 1.0])

        with exp_col1:
            st.markdown("##### 📊 Feature Importance Breakdown")
            st.dataframe(
                df_imp[["Rank", "Feature", "Importance_Pct", "Supplier Value", "Benchmark Median"]],
                use_container_width=True,
                hide_index=True
            )
            st.info("ℹ️ **Methodology**: Features are ranked by relative importance in the Random Forest model. Values are compared against median dataset benchmarks.")

        with exp_col2:
            fig_bar = plot_feature_importance_bar(df_imp)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # Analytical Insights & Executive Recommendation Section
        col_ins, col_rec = st.columns(2)

        with col_ins:
            st.subheader("💡 Analytical Data-Driven Insights")
            insights_text = generate_analytical_insights(raw_input_dict, pred_class, proba_high, df_imp)
            st.markdown(insights_text)

        with col_rec:
            st.subheader("📋 Decision-Support Recommendation")
            rec_text = generate_supplier_recommendation(pred_class, proba_high, df_imp)
            st.markdown(rec_text)

        st.markdown("---")

        # Benchmark Comparison Section
        st.subheader("📈 Supplier Input vs Dataset Benchmark Medians")
        fig_bench = plot_supplier_vs_benchmark(raw_input_dict, benchmark_medians)
        st.plotly_chart(fig_bench, use_container_width=True)

    # ----------------------------------------------------
    # TAB 2: Unseen Jury Data Batch Evaluation
    # ----------------------------------------------------
    with tab_batch:
        st.subheader("🧪 Unseen Jury Data Automated Batch Testing")
        st.markdown("Test the trained Random Forest model and preprocessing pipeline against unseen jury records without manual intervention.")

        col_b1, col_b2 = st.columns([2, 1])

        with col_b1:
            uploaded_file = st.file_uploader("Upload Unseen Jury Dataset (CSV)", type=["csv"])
            use_preloaded = st.checkbox("Or use pre-loaded unseen jury dataset ('data/unseen_jury_data.csv')", value=True)

        df_jury_raw = None
        if uploaded_file is not None:
            df_jury_raw = pd.read_csv(uploaded_file)
            st.success(f"✓ Uploaded CSV loaded: {len(df_jury_raw)} records.")
        elif use_preloaded:
            jury_path = "data/unseen_jury_data.csv"
            if os.path.exists(jury_path):
                df_jury_raw = pd.read_csv(jury_path)
                st.info(f"📁 Loaded pre-loaded unseen jury data: {len(df_jury_raw)} records.")
            else:
                st.warning("Pre-loaded unseen jury data file not found.")

        if df_jury_raw is not None:
            st.markdown("##### 1. Raw Input Unseen Records")
            st.dataframe(df_jury_raw.head(10), use_container_width=True)

            if st.button("🚀 Run Automated Jury Batch Evaluation", type="primary"):
                with st.spinner("Processing through automated preprocessing pipeline & model inference..."):
                    # Automated Preprocessing (No manual data alteration)
                    X_jury_proc = preprocessor.transform(df_jury_raw)
                    
                    predictions = model.predict(X_jury_proc)
                    probabilities = model.predict_proba(X_jury_proc)

                    supplier_ids = df_jury_raw['Supplier_ID'] if 'Supplier_ID' in df_jury_raw.columns else [f"SUP_{i+1:03d}" for i in range(len(df_jury_raw))]
                    has_labels = 'Risk_Level' in df_jury_raw.columns

                    results = []
                    for i in range(len(X_jury_proc)):
                        p_class = int(predictions[i])
                        p_low = float(probabilities[i][0])
                        p_high = float(probabilities[i][1])
                        
                        df_imp_single = get_feature_importance_table(model, FEATURE_ORDER, X_jury_proc.iloc[i].to_dict(), benchmark_medians)
                        
                        res = {
                            "Supplier_ID": supplier_ids.iloc[i] if hasattr(supplier_ids, 'iloc') else supplier_ids[i],
                            "Predicted Class": p_class,
                            "Risk Label": "HIGH RISK" if p_class == 1 else "LOW RISK",
                            "Low Risk Prob": f"{p_low*100:.1f}%",
                            "High Risk Prob": f"{p_high*100:.1f}%",
                            "Top Driver": df_imp_single.iloc[0]["Feature"]
                        }
                        if has_labels:
                            actual = int(df_jury_raw['Risk_Level'].iloc[i])
                            res["Actual Label"] = "HIGH RISK" if actual == 1 else "LOW RISK"
                            res["Match"] = "✅ Correct" if p_class == actual else "❌ Misclassified"
                        
                        results.append(res)

                    df_batch_res = pd.DataFrame(results)

                    st.markdown("##### 2. Model Prediction Output Summary")
                    st.dataframe(df_batch_res, use_container_width=True)

                    if has_labels:
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                        y_true = df_jury_raw['Risk_Level'].astype(int)
                        y_pred = predictions

                        acc = accuracy_score(y_true, y_pred)
                        prec = precision_score(y_true, y_pred, zero_division=0)
                        rec = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        cm = confusion_matrix(y_true, y_pred)

                        st.markdown("##### 3. Performance Metrics on Unseen Labeled Jury Data")
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        m_col1.metric("Accuracy", f"{acc*100:.2f}%")
                        m_col2.metric("Precision", f"{prec:.4f}")
                        m_col3.metric("Recall", f"{rec:.4f}")
                        m_col4.metric("F1-Score", f"{f1:.4f}")

                        fig_cm = plot_confusion_matrix_heatmap(cm)
                        st.plotly_chart(fig_cm, use_container_width=True)

                    # Export Results CSV
                    csv_data = df_batch_res.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Batch Evaluation Results CSV",
                        data=csv_data,
                        file_name="unseen_jury_predictions.csv",
                        mime="text/csv"
                    )

    # ----------------------------------------------------
    # TAB 3: System & Preprocessing Specifications
    # ----------------------------------------------------
    with tab_about:
        st.subheader("⚙️ System Architecture & Round 1 Preprocessing Reproducibility")
        
        st.markdown("""
        ### Round 1 Preprocessing Rules Implemented:
        1. **Median Imputation**: Missing numeric feature values are imputed using pre-fitted dataset medians.
        2. **Deduplication**: Exact duplicate records and repeated `Supplier_ID` instances are automatically resolved.
        3. **Environmental Compliance Capping**: `Environmental_Compliance` values greater than 100 are strictly capped at 100.
        4. **Count Field Rounding**: `Lead_Time_Days`, `Alternative_Suppliers_Available`, and `Previous_Disruptions` are rounded to whole numbers (integers).
        5. **Outlier Winsorization**: Outliers are detected using the $1.5 \\times IQR$ rule ($Q1 - 1.5 \\times IQR, Q3 + 1.5 \\times IQR$) and capped/winsorized using pre-fitted training boundaries.
        6. **Model Feature Exclusion**: `Supplier_ID` is excluded from feature matrices.
        7. **Strict Feature Order Enforced**:
        ```python
        [
            "Financial_Stability_Score",
            "On_Time_Delivery_Rate",
            "Defect_Rate",
            "Geopolitical_Risk_Index",
            "Lead_Time_Days",
            "Alternative_Suppliers_Available",
            "Contract_Length_Months",
            "Environmental_Compliance",
            "Previous_Disruptions"
        ]
        ```

        ### Model Information:
        - **Algorithm**: Random Forest Classifier (`RandomForestClassifier`, 300 estimators)
        - **Loaded Model File**: `models/supplier_risk_random_forest.pkl`
        - **Target Variable**: `Risk_Level` ($0 = \\text{Low Risk}$, $1 = \\text{High Risk}$)
        """)


if __name__ == "__main__":
    main()
