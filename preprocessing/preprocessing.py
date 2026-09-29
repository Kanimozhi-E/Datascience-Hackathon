"""
Supplier Data Preprocessing Pipeline Module
===========================================
Reproduces the Round 1 Preprocessing logic exactly:
1. Handles missing numeric values using median imputation fit on training data.
2. Deduplicates exact duplicate rows and duplicate Supplier_ID records.
3. Caps Environmental_Compliance values greater than 100 at 100.
4. Converts count-type fields (Lead_Time_Days, Alternative_Suppliers_Available, Previous_Disruptions) to whole numbers (integers).
5. Detects outliers using IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR) fit on training data.
6. Caps/winsorizes outliers using training boundaries (without deleting complete supplier records).
7. Excludes Supplier_ID from predictive features.
8. Enforces exact feature ordering expected by the model.
"""

import os
import joblib
import pandas as pd
import numpy as np

# Model feature order strictly required
FEATURE_ORDER = [
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

COUNT_FEATURES = [
    "Lead_Time_Days",
    "Alternative_Suppliers_Available",
    "Previous_Disruptions"
]


class SupplierDataPreprocessor:
    """
    Robust Preprocessor for Supplier Risk Data.
    Applies fitted medians and IQR winsorization bounds from model training.
    """

    def __init__(self, pipeline_path=None):
        self.feature_names = FEATURE_ORDER
        self.count_features = COUNT_FEATURES
        self.medians = {}
        self.iqr_bounds = {}

        if pipeline_path and os.path.exists(pipeline_path):
            self.load_pipeline(pipeline_path)

    def fit(self, df: pd.DataFrame):
        """Fit preprocessor medians and IQR bounds on training dataset."""
        df_clean = df.copy()
        
        # Exclude Supplier_ID and target if present
        feature_df = df_clean[[col for col in FEATURE_ORDER if col in df_clean.columns]]
        
        for col in FEATURE_ORDER:
            if col in feature_df.columns:
                self.medians[col] = float(feature_df[col].median())
                q1 = float(feature_df[col].quantile(0.25))
                q3 = float(feature_df[col].quantile(0.75))
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                self.iqr_bounds[col] = (lower_bound, upper_bound)
            else:
                self.medians[col] = 0.0
                self.iqr_bounds[col] = (-np.inf, np.inf)

    def load_pipeline(self, pipeline_path: str):
        """Load pre-fitted pipeline dictionary containing medians and bounds."""
        artifact = joblib.load(pipeline_path)
        self.feature_names = artifact.get('feature_names', FEATURE_ORDER)
        self.count_features = artifact.get('count_features', COUNT_FEATURES)
        self.medians = artifact.get('medians', {})
        self.iqr_bounds = artifact.get('iqr_bounds', {})

    def transform(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Apply identical preprocessing rules to input DataFrame.
        """
        data = df_input.copy()

        # Step 1: Remove exact duplicates & Supplier_ID repeated records if raw dataset format
        if 'Supplier_ID' in data.columns and len(data) > 1:
            data = data.drop_duplicates(subset=['Supplier_ID'], keep='first')
        data = data.drop_duplicates()

        # Step 2: Ensure all required features exist, filling missing numeric values with medians
        for col in self.feature_names:
            if col not in data.columns:
                data[col] = self.medians.get(col, 0.0)
            else:
                med_val = self.medians.get(col, 0.0)
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(med_val)

        # Step 3: Cap Environmental_Compliance values > 100 at 100
        if 'Environmental_Compliance' in data.columns:
            data['Environmental_Compliance'] = data['Environmental_Compliance'].clip(upper=100.0)

        # Step 4: Convert count-type fields to whole numbers (integers)
        for col in self.count_features:
            if col in data.columns:
                data[col] = data[col].round().astype(int)

        # Step 5: Treat outliers using IQR winsorization (capping)
        for col in self.feature_names:
            if col in self.iqr_bounds:
                lb, ub = self.iqr_bounds[col]
                data[col] = data[col].clip(lower=lb, upper=ub)

        # Step 6: Exclude Supplier_ID and return EXACT feature order
        return data[self.feature_names]

    def transform_single_dict(self, input_dict: dict) -> pd.DataFrame:
        """
        Transform a single raw dictionary input (e.g. from Streamlit UI) into a 1-row DataFrame.
        """
        df_single = pd.DataFrame([input_dict])
        return self.transform(df_single)


def load_preprocessor(pipeline_path="models/preprocessing_pipeline.pkl"):
    """Helper function to load fitted preprocessor object."""
    if os.path.exists(pipeline_path):
        return SupplierDataPreprocessor(pipeline_path=pipeline_path)
    elif os.path.exists("../models/preprocessing_pipeline.pkl"):
        return SupplierDataPreprocessor(pipeline_path="../models/preprocessing_pipeline.pkl")
    else:
        # Fallback default initialization
        prep = SupplierDataPreprocessor()
        return prep
