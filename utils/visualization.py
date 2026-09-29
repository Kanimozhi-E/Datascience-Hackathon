"""
Visualization Module for Supplier Risk Scoring Analytics
=========================================================
Generates interactive Plotly charts for prediction probabilities,
feature importances, benchmark comparisons, and model evaluations.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_risk_probability_donut(risk_proba_low, risk_proba_high):
    """
    Create a clean donut chart showing Low Risk vs High Risk probabilities.
    """
    labels = ['Low Risk', 'High Risk']
    values = [risk_proba_low, risk_proba_high]
    colors = ['#10B981', '#EF4444']  # Emerald green vs Vibrant red

    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker=dict(colors=colors),
            textinfo='label+percent',
            hoverinfo='label+percent',
            textfont=dict(size=14, family="sans-serif", color="white")
        )
    ])

    fig.update_layout(
        title={
            'text': "<b>Risk Probability Distribution</b>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        margin=dict(t=50, b=30, l=20, r=20),
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#E2E8F0")
    )
    
    return fig


def plot_feature_importance_bar(df_imp, top_n=9):
    """
    Create a sleek horizontal bar chart of feature importances.
    """
    df_plot = df_imp.head(top_n).sort_values(by="Importance", ascending=True)
    
    fig = px.bar(
        df_plot,
        x="Importance",
        y="Feature",
        orientation='h',
        text=df_plot["Importance_Pct"],
        color="Importance",
        color_continuous_scale="Viridis",
        title="<b>Random Forest Feature Importances</b>"
    )
    
    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    )
    
    fig.update_layout(
        xaxis_title="Relative Feature Weight",
        yaxis_title="",
        coloraxis_showscale=False,
        height=380,
        margin=dict(t=50, b=30, l=10, r=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#E2E8F0")
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)')
    fig.update_yaxes(showgrid=False)
    
    return fig


def plot_supplier_vs_benchmark(input_dict, benchmark_medians):
    """
    Create a comparison bar chart between Supplier Input values and Benchmark Medians.
    Values are normalized to 0-100 scale for intuitive side-by-side comparison.
    """
    # Scale ranges for normalization to 0-100%
    ranges = {
        "Financial_Stability_Score": (0, 100),
        "On_Time_Delivery_Rate": (0, 100),
        "Defect_Rate": (0, 15),
        "Geopolitical_Risk_Index": (0, 100),
        "Lead_Time_Days": (0, 70),
        "Alternative_Suppliers_Available": (0, 10),
        "Contract_Length_Months": (0, 60),
        "Environmental_Compliance": (0, 100),
        "Previous_Disruptions": (0, 5)
    }

    labels = {
        "Financial_Stability_Score": "Financial Stability",
        "On_Time_Delivery_Rate": "On-Time Delivery",
        "Defect_Rate": "Defect Rate",
        "Geopolitical_Risk_Index": "Geopolitical Risk",
        "Lead_Time_Days": "Lead Time",
        "Alternative_Suppliers_Available": "Alt Suppliers",
        "Contract_Length_Months": "Contract Length",
        "Environmental_Compliance": "Env Compliance",
        "Previous_Disruptions": "Disruptions"
    }

    categories = []
    supplier_norm = []
    benchmark_norm = []
    supplier_raw = []
    benchmark_raw = []

    for k, label in labels.items():
        if k in input_dict and k in benchmark_medians:
            val_s = float(input_dict[k])
            val_b = float(benchmark_medians[k])
            r_min, r_max = ranges[k]
            
            s_n = max(0.0, min(100.0, ((val_s - r_min) / (r_max - r_min)) * 100))
            b_n = max(0.0, min(100.0, ((val_b - r_min) / (r_max - r_min)) * 100))
            
            categories.append(label)
            supplier_norm.append(s_n)
            benchmark_norm.append(b_n)
            supplier_raw.append(val_s)
            benchmark_raw.append(val_b)

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Supplier Value (Normalized %)',
        x=categories,
        y=supplier_norm,
        marker_color='#3B82F6', # Vivid Blue
        customdata=supplier_raw,
        hovertemplate='<b>%{x}</b><br>Supplier Value: %{customdata:.2f}<br>Normalized: %{y:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Dataset Benchmark Median (Normalized %)',
        x=categories,
        y=benchmark_norm,
        marker_color='#94A3B8', # Slate gray
        customdata=benchmark_raw,
        hovertemplate='<b>%{x}</b><br>Benchmark Median: %{customdata:.2f}<br>Normalized: %{y:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title="<b>Supplier Inputs vs Dataset Benchmark Medians (Normalized 0-100%)</b>",
        barmode='group',
        height=380,
        margin=dict(t=50, b=40, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#E2E8F0")
    )
    fig.update_yaxes(range=[0, 105], showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)')

    return fig


def plot_confusion_matrix_heatmap(cm):
    """
    Create a heatmap of the confusion matrix for batch evaluation.
    """
    labels = ['Low Risk (0)', 'High Risk (1)']
    
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        title="<b>Model Evaluation: Confusion Matrix</b>"
    )
    
    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        height=340,
        margin=dict(t=50, b=40, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#E2E8F0")
    )
    
    return fig
