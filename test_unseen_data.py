"""
Standalone Validation Script for Unseen Jury Data
=================================================
Executes end-to-end automated testing on unseen supplier records.
Applies exact preprocessing pipeline, generates Random Forest predictions,
extracts feature importances, and reports evaluation metrics (if labels exist).

Usage:
    python test_unseen_data.py [--data_path data/unseen_jury_data.csv]
"""

import sys
import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.preprocessing import SupplierDataPreprocessor, FEATURE_ORDER
from utils.explainability import get_feature_importance_table


def run_unseen_test(data_path="data/unseen_jury_data.csv", model_path="models/supplier_risk_random_forest.pkl", pipeline_path="models/preprocessing_pipeline.pkl"):
    print("=" * 70)
    print("SUPPLIER RISK ANALYTICS - UNSEEN JURY DATA VALIDATION TEST")
    print("=" * 70)

    # 1. Load trained Random Forest model
    if not os.path.exists(model_path):
        if os.path.exists("supplier_risk_random_forest.pkl"):
            model_path = "supplier_risk_random_forest.pkl"
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
    print(f"[1/5] Loading trained Random Forest model from: {model_path}")
    model = joblib.load(model_path)

    # 2. Load preprocessor
    print(f"[2/5] Loading preprocessing pipeline from: {pipeline_path}")
    preprocessor = SupplierDataPreprocessor(pipeline_path=pipeline_path)

    # 3. Load unseen jury dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Unseen jury data file not found at {data_path}")

    print(f"[3/5] Reading raw unseen jury dataset: {data_path}")
    raw_df = pd.read_csv(data_path)
    print(f"      Total records loaded: {len(raw_df)}")
    print("      Raw sample head:")
    print(raw_df.head(3))

    # Preserve Supplier_ID and Risk_Level ground truth if present
    supplier_ids = raw_df['Supplier_ID'] if 'Supplier_ID' in raw_df.columns else [f"SUP_{i+1:03d}" for i in range(len(raw_df))]
    ground_truth = raw_df['Risk_Level'] if 'Risk_Level' in raw_df.columns else None

    # 4. Automated Preprocessing (Without manual modification)
    print("\n[4/5] Executing automated preprocessing pipeline...")
    X_processed = preprocessor.transform(raw_df)
    print("      Features preprocessed and formatted successfully.")
    print("      Features order:", X_processed.columns.tolist())

    # 5. Model Inference & Probabilities
    print("\n[5/5] Generating Random Forest predictions & probabilities...")
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)

    results = []
    benchmark_medians = preprocessor.medians

    for i in range(len(X_processed)):
        row_dict = X_processed.iloc[i].to_dict()
        pred_class = int(predictions[i])
        proba_low = float(probabilities[i][0])
        proba_high = float(probabilities[i][1])
        
        # Calculate feature importances for this record
        df_imp = get_feature_importance_table(model, FEATURE_ORDER, row_dict, benchmark_medians)
        top_driver = df_imp.iloc[0]['Feature']
        
        res = {
            "Supplier_ID": supplier_ids.iloc[i] if hasattr(supplier_ids, 'iloc') else supplier_ids[i],
            "Predicted_Risk_Class": pred_class,
            "Risk_Label": "HIGH RISK" if pred_class == 1 else "LOW RISK",
            "Low_Risk_Probability": round(proba_low, 4),
            "High_Risk_Probability": round(proba_high, 4),
            "Top_Risk_Factor": top_driver
        }
        
        if ground_truth is not None:
            res["Actual_Risk_Level"] = int(ground_truth.iloc[i])
            res["Correct_Prediction"] = (pred_class == int(ground_truth.iloc[i]))
            
        results.append(res)

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("PREDICTION RESULTS SUMMARY FOR UNSEEN JURY DATA")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # If ground truth labels are present, calculate evaluation metrics
    if ground_truth is not None:
        y_true = ground_truth.astype(int)
        y_pred = predictions
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        print("\n" + "=" * 70)
        print("PERFORMANCE METRICS ON LABELED UNSEEN JURY DATA")
        print("=" * 70)
        print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("\nConfusion Matrix:")
        print(pd.DataFrame(cm, index=["Actual 0 (Low)", "Actual 1 (High)"], columns=["Pred 0 (Low)", "Pred 1 (High)"]))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=["Low Risk (0)", "High Risk (1)"]))

    # Save output report
    output_csv = "unseen_predictions_output.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\n[SUCCESS] Validation complete! Full results saved to '{output_csv}'.\n")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Supplier Risk RF Model on Unseen Data")
    parser.add_argument("--data_path", type=str, default="data/unseen_jury_data.csv", help="Path to unseen jury CSV file")
    parser.add_argument("--model_path", type=str, default="models/supplier_risk_random_forest.pkl", help="Path to RF model pkl")
    parser.add_argument("--pipeline_path", type=str, default="models/preprocessing_pipeline.pkl", help="Path to preprocessing pipeline pkl")
    args = parser.parse_args()

    run_unseen_test(args.data_path, args.model_path, args.pipeline_path)
