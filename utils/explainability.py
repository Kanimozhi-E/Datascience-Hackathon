"""
Supplier Risk Explainability & Decision Support Module
======================================================
Provides feature importance analysis, data-driven analytical insights,
and decision-support recommendations.

IMPORTANT:
Does not claim causal relationships. Uses model-supported feature association
and decision-support phrasing.
"""

import pandas as pd
import numpy as np


FEATURE_LABELS = {
    "Financial_Stability_Score": "Financial Stability Score",
    "On_Time_Delivery_Rate": "On-Time Delivery Rate (%)",
    "Defect_Rate": "Defect Rate (%)",
    "Geopolitical_Risk_Index": "Geopolitical Risk Index",
    "Lead_Time_Days": "Lead Time (Days)",
    "Alternative_Suppliers_Available": "Alternative Suppliers Available",
    "Contract_Length_Months": "Contract Length (Months)",
    "Environmental_Compliance": "Environmental Compliance Score",
    "Previous_Disruptions": "Previous Disruptions (Count)"
}


def get_feature_importance_table(model, feature_names, input_values_dict=None, benchmark_medians=None):
    """
    Extract feature importances from trained Random Forest model and map to supplier values.

    Returns:
        pd.DataFrame: Sorted table with columns [Feature, Importance_Pct, Supplier_Value, Benchmark_Median, Rank]
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Provided model does not have feature_importances_ attribute.")

    importances = model.feature_importances_
    
    records = []
    for idx, name in enumerate(feature_names):
        imp_val = float(importances[idx])
        supp_val = input_values_dict.get(name, np.nan) if input_values_dict else np.nan
        bench_val = benchmark_medians.get(name, np.nan) if benchmark_medians else np.nan
        
        records.append({
            "Feature_Code": name,
            "Feature": FEATURE_LABELS.get(name, name),
            "Importance": imp_val,
            "Importance_Pct": f"{imp_val * 100:.2f}%",
            "Supplier Value": f"{supp_val:.2f}" if isinstance(supp_val, (float, int)) else str(supp_val),
            "Raw_Supplier_Value": supp_val,
            "Benchmark Median": f"{bench_val:.2f}" if isinstance(bench_val, (float, int)) else str(bench_val),
            "Raw_Benchmark_Value": bench_val
        })
        
    df_imp = pd.DataFrame(records)
    df_imp = df_imp.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    df_imp["Rank"] = range(1, len(df_imp) + 1)
    
    return df_imp


def generate_analytical_insights(input_dict, risk_class, risk_proba_high, df_imp):
    """
    Generate data-driven analytical insights based on supplier input values and model importances.
    STRICTLY avoids claiming causal relationships.
    """
    insights = []
    
    # 1. Overall Risk Overview
    proba_pct = risk_proba_high * 100
    if risk_class == 1:
        insights.append(
            f"**High Risk Profile**: The model evaluated this supplier with a **{proba_pct:.1f}% probability of High Risk**."
        )
    else:
        insights.append(
            f"**Low Risk Profile**: The model evaluated this supplier with a **{(100 - proba_pct):.1f}% probability of Low Risk** (High Risk probability: {proba_pct:.1f}%)."
        )

    # 2. Key Influential Feature Drivers (Top 3)
    top_3 = df_imp.head(3)
    insights.append("### Key Influential Features (Based on Random Forest Feature Importances)")
    
    for _, row in top_3.iterrows():
        feat_code = row["Feature_Code"]
        feat_name = row["Feature"]
        supp_val = row["Raw_Supplier_Value"]
        imp_pct = row["Importance_Pct"]
        
        if feat_code == "Financial_Stability_Score":
            if supp_val < 50:
                insights.append(
                    f"- **Financial Stability Score ({supp_val:.1f})**: High model weight ({imp_pct}). Lower financial stability score is associated with higher predicted risk in the trained model."
                )
            else:
                insights.append(
                    f"- **Financial Stability Score ({supp_val:.1f})**: High model weight ({imp_pct}). Higher financial stability is associated with lower predicted risk in the trained model."
                )
                
        elif feat_code == "Geopolitical_Risk_Index":
            if supp_val > 45:
                insights.append(
                    f"- **Geopolitical Risk Index ({supp_val:.0f})**: High model weight ({imp_pct}). Elevated geopolitical risk index contributes strongly to the model's risk assessment."
                )
            else:
                insights.append(
                    f"- **Geopolitical Risk Index ({supp_val:.0f})**: High model weight ({imp_pct}). Moderate/low geopolitical risk supports a favorable risk classification."
                )
                
        elif feat_code == "Alternative_Suppliers_Available":
            if supp_val <= 1:
                insights.append(
                    f"- **Alternative Suppliers Available ({supp_val:.0f})**: Influential factor ({imp_pct}). Limited alternative suppliers may increase exposure to supply chain disruption."
                )
            else:
                insights.append(
                    f"- **Alternative Suppliers Available ({supp_val:.0f})**: Influential factor ({imp_pct}). Availability of alternative suppliers correlates with reduced operational risk."
                )
                
        elif feat_code == "Defect_Rate":
            if supp_val > 5.0:
                insights.append(
                    f"- **Defect Rate ({supp_val:.2f}%)**: Influential quality metric ({imp_pct}). Higher defect rates align with increased quality risk in the model."
                )
            else:
                insights.append(
                    f"- **Defect Rate ({supp_val:.2f}%)**: Influential quality metric ({imp_pct}). Low defect rate supports strong quality reliability."
                )
                
        elif feat_code == "On_Time_Delivery_Rate":
            if supp_val < 75.0:
                insights.append(
                    f"- **On-Time Delivery Rate ({supp_val:.1f}%)**: Delivery reliability metric ({imp_pct}). Below-average delivery timeliness increases risk exposure."
                )
            else:
                insights.append(
                    f"- **On-Time Delivery Rate ({supp_val:.1f}%)**: Delivery reliability metric ({imp_pct}). High on-time delivery rate reinforces operational stability."
                )
                
        elif feat_code == "Previous_Disruptions":
            if supp_val >= 2:
                insights.append(
                    f"- **Previous Disruptions ({supp_val:.0f})**: Historical disruption count ({imp_pct}). Multiple past disruptions positively correlate with higher risk scoring."
                )
            else:
                insights.append(
                    f"- **Previous Disruptions ({supp_val:.0f})**: Minimal disruption history ({imp_pct}) aligns with stable supply behavior."
                )
                
        else:
            insights.append(
                f"- **{feat_name} ({supp_val})**: Influential feature ({imp_pct}) contributing to the final model prediction."
            )
            
    insights.append(
        "\n*Note: These insights reflect feature importances and statistical associations learned by the trained model, not proven causal relationships.*"
    )

    return "\n\n".join(insights)


def generate_supplier_recommendation(risk_class, risk_proba_high, df_imp):
    """
    Generate decision-support recommendations based on prediction result and key factors.
    """
    if risk_class == 1:
        rec = (
            "### ⚠️ Executive Decision-Support Recommendation\n"
            "**Supplier Risk Status: HIGH RISK**\n\n"
            "The supplier is classified as **High Risk** by the trained model. "
            "It is recommended to review the major contributing factors and consider appropriate supplier-risk mitigation measures:\n"
            "- **Enhanced Monitoring**: Implement real-time tracking of financial stability and delivery KPIs.\n"
            "- **Contingency & Secondary Sourcing**: Develop backup supplier agreements to mitigate disruption risk.\n"
            "- **Contractual Safeguards**: Shorten review cycles or request performance guarantee clauses.\n\n"
            "*Note: This recommendation is provided as automated decision support based on model predictions. Business decisions should involve human domain expertise.*"
        )
    else:
        rec = (
            "### ✅ Executive Decision-Support Recommendation\n"
            "**Supplier Risk Status: LOW RISK**\n\n"
            "The supplier is classified as **Low Risk** by the trained model.\n"
            "- **Standard Procurement**: Proceed with standard vendor onboarding and periodic review.\n"
            "- **Continuous Indicator Tracking**: Continue monitoring key risk indicators (delivery rate, financial metrics).\n"
            "- **Relationship Building**: Consider long-term strategic partnership opportunities.\n\n"
            "*Note: This recommendation is provided as automated decision support based on model predictions.*"
        )
    return rec
