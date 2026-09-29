# 🛡️ Supplier Risk Scoring Analytics

### AI-Powered Supplier Risk Assessment & Explainable Decision Support

---

## 📌 Problem Statement & Objective
**Objective**: Build a data-driven system that uses supplier information (delivery reliability, quality failures, financial stability, lead time, geopolitical risk, and disruption history) to predict **Supplier Risk Class** (`0 = Low Risk`, `1 = High Risk`) and provide explainable decision-support output.

---

## 🏗️ Project Architecture & File Structure

```
supplier-risk-project/
│
├── app.py                            # Interactive Streamlit Web Dashboard
├── requirements.txt                   # Dependency manifest
├── README.md                          # Project documentation & run guide
│
├── models/
│   ├── supplier_risk_random_forest.pkl# Trained Random Forest Classifier
│   └── preprocessing_pipeline.pkl     # Pre-fitted preprocessing artifacts
│
├── preprocessing/
│   └── preprocessing.py               # Round 1 Preprocessing pipeline module
│
├── utils/
│   ├── explainability.py              # Feature importances & analytical insights
│   └── visualization.py              # Plotly interactive charts & benchmarks
│
├── data/
│   ├── sample_data.csv                # Sample exploration dataset
│   └── unseen_jury_data.csv           # Unseen jury evaluation dataset
│
└── test_unseen_data.py                # Standalone test verification script
```

---

## ⚙️ Round 1 Preprocessing Rules Implemented

The preprocessing pipeline reproduces the training preprocessing steps for any new or unseen supplier input:

1. **Median Imputation**: Missing numerical values are imputed using pre-calculated dataset medians.
2. **Deduplication**: Exact duplicate rows and repeated `Supplier_ID` instances are handled.
3. **Environmental Compliance Capping**: `Environmental_Compliance` values $> 100$ are automatically capped at $100$.
4. **Count Field Conversion**: `Lead_Time_Days`, `Alternative_Suppliers_Available`, and `Previous_Disruptions` are converted to whole numbers (integers).
5. **IQR Winsorization (Outlier Capping)**: Outliers are detected via $1.5 \times IQR$ ($Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR$) and capped/winsorized using the trained dataset boundaries instead of dropping records.
6. **Feature Exclusion**: `Supplier_ID` is strictly excluded from model features.
7. **Strict Feature Order Enforced**:
   - `Financial_Stability_Score`
   - `On_Time_Delivery_Rate`
   - `Defect_Rate`
   - `Geopolitical_Risk_Index`
   - `Lead_Time_Days`
   - `Alternative_Suppliers_Available`
   - `Contract_Length_Months`
   - `Environmental_Compliance`
   - `Previous_Disruptions`

---

## 🚀 Getting Started & Execution

### 1. Prerequisites & Installation
Ensure Python 3.10+ is installed. Install required packages using:

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
Run the Streamlit application using:

```bash
streamlit run app.py
```

### 3. Run Automated Validation on Unseen Jury Data
To verify the complete pipeline on unseen jury records without manual intervention, execute:

```bash
python test_unseen_data.py
```

Options:
```bash
python test_unseen_data.py --data_path data/unseen_jury_data.csv
```

---

## 💡 Key Features of the Prototype

- **Interactive UI**: Enter individual supplier parameters via intuitive widgets or load presets (Low Risk, High Risk, Jury Sample).
- **Instant Risk Classification**: Displays predicted Risk Class (**LOW RISK** / **HIGH RISK**) along with confidence probabilities ($P(\text{Low Risk})$, $P(\text{High Risk})$).
- **Model Explainability**: Formatted feature importance breakdown based on the trained Random Forest model's `feature_importances_`.
- **Data-Driven Analytical Insights**: Non-causal, association-based narratives explaining the key driving factors.
- **Decision-Support Recommendations**: Context-aware executive recommendations for procurement and risk mitigation.
- **Unseen Jury Data Batch Testing**: Upload custom CSVs or run batch testing with automatic performance evaluation (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).

---

## ⚠️ Non-Causality Disclaimer
The model outputs and feature importances reflect statistical associations and relative feature weights learned during model training on historical data. They do not prove causal relationships or guarantee future supplier behavior. Recommendations are provided strictly as decision support to assist human domain experts.
