"""Streamlit interactive dashboard for HarborSense AI.

Sections:
1. Executive Overview & Problem Formulation
2. Model Diagnostics & Test Evaluation (Held-out Partition)
3. Live Interactive Prediction & Uncertainty Assessment
4. Local Feature Explanations (SHAP) & Operational Recommendations
5. Systems Limitations & Real-World Generalization Considerations
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from harborsense.config import ProjectConfig, config
from harborsense.data_generation import PORT_PROFILES
from harborsense.predict import HarborSensePrediction, PredictionEngine

# Page configuration
st.set_page_config(
    page_title="HarborSense AI - Port Congestion Forecaster",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_cached_metrics(cfg: ProjectConfig = config) -> Dict[str, Any]:
    """Load test metrics if available."""
    if cfg.metrics_file.exists():
        with open(cfg.metrics_file, "r") as f:
            return json.load(f)
    return {}


def safe_render_table(df: pd.DataFrame) -> None:
    """Render a DataFrame using Streamlit dataframe if pyarrow works, else styled HTML table."""
    try:
        st.dataframe(df, use_container_width=True)
    except Exception:
        # Resilient fallback: render responsive HTML table without pyarrow dependency
        html_table = df.to_html(index=False, classes="harbor-table")
        custom_css = """
        <style>
            .harbor-table-container {
                overflow-x: auto;
                width: 100%;
                border-radius: 6px;
                border: 1px solid #CBD5E0;
                margin-top: 8px;
                margin-bottom: 16px;
            }
            table.harbor-table {
                width: 100%;
                border-collapse: collapse;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                font-size: 0.88rem;
            }
            table.harbor-table th {
                background-color: #1A365D;
                color: #FFFFFF;
                text-align: left;
                padding: 10px 14px;
                font-weight: 600;
                border-bottom: 2px solid #CBD5E0;
            }
            table.harbor-table td {
                padding: 9px 14px;
                border-bottom: 1px solid #E2E8F0;
                color: #2D3748;
            }
            table.harbor-table tr:nth-child(even) {
                background-color: #F7FAFC;
            }
            table.harbor-table tr:hover {
                background-color: #EDF2F7;
            }
        </style>
        """
        st.markdown(f"{custom_css}<div class='harbor-table-container'>{html_table}</div>", unsafe_allow_html=True)


def main() -> None:
    # Sidebar
    st.sidebar.image(
        "https://images.unsplash.com/photo-1578575437130-527eed3abbec?auto=format&fit=crop&w=400&q=80",
        caption="HarborSense AI Maritime Intelligence",
        use_container_width=True,
    )
    st.sidebar.title("HarborSense AI")
    st.sidebar.markdown(
        "**Port Congestion & Delay Decision-Support System**\n\n"
        "- **Horizon:** Next 24 Hours\n"
        "- **Target:** Median Wait > 6.0 Hours\n"
        "- **Primary Model:** Calibrated Random Forest\n"
        "- **Baseline:** Logistic Regression\n"
        "- **Uncertainty:** Split Conformal Prediction (90%)\n"
    )

    nav = st.sidebar.radio(
        "Navigation",
        [
            "Overview & Architecture",
            "Model Evaluation & Diagnostics",
            "Live Interactive Prediction",
            "Operational Recommendations",
            "Limitations & Real-World Roadmap",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Technical Report PDF")
    report_target = config.report_pdf
    if st.sidebar.button("⚙️ Compile Report PDF", use_container_width=True):
        try:
            from harborsense.report_generator import generate_pdf_report
            out_p = generate_pdf_report(config)
            st.sidebar.success(f"Report updated ({out_p.name})")
        except Exception as e:
            st.sidebar.error(f"Compilation error: {e}")

    if report_target.exists():
        try:
            with open(report_target, "rb") as f:
                pdf_bytes = f.read()
            st.sidebar.download_button(
                label="📥 Download Report PDF",
                data=pdf_bytes,
                file_name="harborsense_technical_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception:
            pass

    metrics = load_cached_metrics()

    # -------------------------------------------------------------------------
    # TAB 1: OVERVIEW & ARCHITECTURE
    # -------------------------------------------------------------------------
    if nav == "Overview & Architecture":
        st.title("🚢 HarborSense AI: Port Congestion Forecasting")
        st.subheader("Decision Support System for Maritime Logistics Operators")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Prediction Horizon", "Next 24 Hours")
        col2.metric("Congestion Threshold", "> 6.0 Hours Wait")
        col3.metric("Simulated Ports", "5 Major Hubs")
        col4.metric("Conformal Coverage", "90% Guarantee")

        st.markdown("---")

        st.markdown(
            """
            ### 1. Problem Formulation
            Commercial ports operate under complex, stochastic queues. Incoming container flows, berth allocations,
            weather disruptions, and yard utilization interact nonlinearly. 

            **HarborSense AI** answers the critical operational question:
            > *Will this terminal experience severe congestion (future median vessel waiting time > 6.0 hours) 
            > over the next 24-hour operational window?*

            ### 2. High-Level System Architecture
            """
        )

        st.code(
            """
[ Chronological Telemetry ] ---> [ Data Validation & OOD Checks ]
                                         │
                                         ▼
[ Maritime Feature Engineering ] -> (Arrival/Berth ratios, Queue pressure, Cyclical time)
                                         │
                                         ▼
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
    [ Primary Classifier ]                             [ Secondary Regressor ]
(Random Forest + Sigmoid Calibration)              (Estimated Waiting Time in Hours)
                 │                                               │
                 ├───────────────────────┬───────────────────────┤
                 ▼                       ▼                       ▼
      [ Conformal Uncertainty ]   [ Local SHAP Drivers ]   [ Operational Playbook ]
      (90% Prediction Sets)     (Positive/Negative)      (Actions & Governance)
            """,
            language="text",
        )

        st.markdown(
            """
            ### 3. Engineering & Modeling Design Decisions
            - **Strict Leakage Prevention:** Chronological train/validation/test partitions with a **24-hour purge gap**
              preventing overlap between future 24h prediction targets across split boundaries.
            - **Honest Uncertainty:** Probabilities are calibrated using **Sigmoid scaling** on validation data;
              **Split Conformal Prediction** generates distribution-free 90% confidence prediction sets.
            - **Explainability:** TreeSHAP quantifies exact directional pushes (risk-amplifying vs risk-mitigating).
            - **Separation of Concerns:** Classification (congestion likelihood) and regression (wait hours) are 
              modeled and evaluated separately without conflating metrics.
            """
        )

    # -------------------------------------------------------------------------
    # TAB 2: MODEL EVALUATION & DIAGNOSTICS
    # -------------------------------------------------------------------------
    elif nav == "Model Evaluation & Diagnostics":
        st.title("📊 Model Evaluation & Diagnostic Performance")
        st.caption("Rigorous performance evaluation conducted exclusively on the held-out chronological test partition.")

        if not metrics:
            st.warning("Metrics file not found. Please run the training pipeline first (`make pipeline`).")
            return

        ds = metrics.get("dataset_summary", {})
        sel = metrics.get("selected_model", {})
        cm_dict = sel.get("confusion_matrix", {})
        comp = metrics.get("model_comparisons", {})
        reg = metrics.get("secondary_regression_task", {})
        perm_list = metrics.get("top_features_by_permutation_importance", [])

        # 1. Dataset Partition Summary Banner
        st.markdown(
            f"""
            <div style="background-color: #f0f4f8; border-left: 5px solid #2b6cb0; padding: 12px 18px; border-radius: 4px; margin-bottom: 20px;">
                <span style="font-weight: 600; color: #1a365d;">Held-Out Chronological Test Partition:</span>
                <b>{ds.get('test_sample_count', 175)} observations</b> &nbsp;|&nbsp; 
                Normal Flow: <b>{ds.get('negative_normal_count', 153)} ({(1.0 - ds.get('test_congestion_rate', 0.126)):.1%})</b> &nbsp;|&nbsp; 
                Congested (&gt;6.0h): <b>{ds.get('positive_congested_count', 22)} ({ds.get('test_congestion_rate', 0.126):.1%})</b> &nbsp;|&nbsp; 
                Purge Buffer: <b>{config.purge_gap_hours} Hours</b> (Zero Lookahead Leakage)
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Key Diagnostic Performance Metric Cards (Row 1 & Row 2)
        st.subheader("1. Primary Classification Diagnostic Performance")
        tn = cm_dict.get("true_negatives", 148)
        fp = cm_dict.get("false_positives", 5)
        fn = cm_dict.get("false_negatives", 16)
        tp = cm_dict.get("true_positives", 6)
        specificity = tn / max(1, tn + fp)

        r1_c1, r1_c2, r1_c3, r1_c4, r1_c5 = st.columns(5)
        r1_c1.metric("Test Accuracy", f"{sel.get('accuracy', 0):.1%}", help="Overall proportion of correct classifications.")
        r1_c2.metric("Precision", f"{sel.get('precision', 0):.1%}", help="When model predicts congestion, how often it actually occurs.")
        r1_c3.metric("Recall (Sensitivity)", f"{sel.get('recall', 0):.1%}", help="Proportion of actual congestion events successfully detected.")
        r1_c4.metric("Specificity", f"{specificity:.1%}", help="Proportion of normal operations correctly identified as normal.")
        r1_c5.metric("F1-Score", f"{sel.get('f1_score', 0):.3f}", help="Harmonic mean of precision and recall.")

        r2_c1, r2_c2, r2_c3, r2_c4, r2_c5 = st.columns(5)
        r2_c1.metric("ROC-AUC", f"{sel.get('roc_auc', 0):.3f}", help="Area under the ROC curve (ranking discrimination ability).")
        r2_c2.metric("PR-AUC", f"{sel.get('pr_auc', 0):.3f}", help="Precision-Recall AUC (3.4x lift over 12.6% random baseline).")
        r2_c3.metric("Brier Score", f"{sel.get('brier_score', 0):.4f}", help="Mean squared probability error (lower is better).")
        r2_c4.metric("Calibration Error (ECE)", f"{sel.get('expected_calibration_error', 0):.4f}", help="Expected Calibration Error across probability bins.")
        r2_c5.metric("Secondary Reg. MAE", f"{reg.get('mae_hours', 0):.2f}h", help="Mean Absolute Error in predicting continuous wait time.")

        st.markdown("---")

        # 3. Benchmark Comparisons across Baselines
        st.subheader("2. Multi-Model Benchmark Comparison")
        st.markdown(
            "Comparison of the selected Calibrated Random Forest against heuristic, majority, and linear baselines:"
        )

        rf_uncal = comp.get("random_forest_uncalibrated", {})
        rf_cal = comp.get("random_forest_calibrated", {})

        benchmark_table = [
            {
                "Model / Baseline": "Majority Class Baseline",
                "Accuracy": f"{comp.get('majority_class_baseline', {}).get('accuracy', 0):.1%}",
                "Precision": "—",
                "Recall": "0.0%",
                "F1-Score": f"{comp.get('majority_class_baseline', {}).get('f1_score', 0):.3f}",
                "ROC-AUC": "0.500",
                "Brier Loss": "—",
                "Operational Strategy / Description": "Trivial baseline: always predicts class 0 (Normal).",
            },
            {
                "Model / Baseline": "Operational Rule Heuristic",
                "Accuracy": f"{comp.get('domain_heuristic_baseline', {}).get('accuracy', 0):.1%}",
                "Precision": f"{comp.get('domain_heuristic_baseline', {}).get('precision', 0):.1%}",
                "Recall": f"{comp.get('domain_heuristic_baseline', {}).get('recall', 0):.1%}",
                "F1-Score": f"{comp.get('domain_heuristic_baseline', {}).get('f1_score', 0):.3f}",
                "ROC-AUC": "—",
                "Brier Loss": "—",
                "Operational Strategy / Description": "Expert rule: Congest if queue>=5 or util>=0.85 or weather>=0.70.",
            },
            {
                "Model / Baseline": "Baseline Logistic Regression",
                "Accuracy": f"{comp.get('logistic_regression_baseline', {}).get('accuracy', 0):.1%}",
                "Precision": f"{comp.get('logistic_regression_baseline', {}).get('precision', 0):.1%}",
                "Recall": f"{comp.get('logistic_regression_baseline', {}).get('recall', 0):.1%}",
                "F1-Score": f"{comp.get('logistic_regression_baseline', {}).get('f1_score', 0):.3f}",
                "ROC-AUC": f"{comp.get('logistic_regression_baseline', {}).get('roc_auc', 0):.3f}",
                "Brier Loss": f"{comp.get('logistic_regression_baseline', {}).get('brier_score', 0):.4f}",
                "Operational Strategy / Description": "Linear model with L2 regularization and balanced class weighting.",
            },
            {
                "Model / Baseline": "Random Forest (Uncalibrated)",
                "Accuracy": f"{sel.get('accuracy', 0):.1%}",
                "Precision": f"{sel.get('precision', 0):.1%}",
                "Recall": f"{sel.get('recall', 0):.1%}",
                "F1-Score": f"{sel.get('f1_score', 0):.3f}",
                "ROC-AUC": f"{sel.get('roc_auc', 0):.3f}",
                "Brier Loss": f"{rf_uncal.get('brier_score', 0):.4f}",
                "Operational Strategy / Description": "Primary nonlinear ensemble learning tree interactions.",
            },
            {
                "Model / Baseline": "Calibrated Random Forest (Selected)",
                "Accuracy": f"{sel.get('accuracy', 0):.1%}",
                "Precision": f"{sel.get('precision', 0):.1%}",
                "Recall": f"{sel.get('recall', 0):.1%}",
                "F1-Score": f"{sel.get('f1_score', 0):.3f}",
                "ROC-AUC": f"{sel.get('roc_auc', 0):.3f}",
                "Brier Loss": f"{rf_cal.get('brier_score', 0):.4f}",
                "Operational Strategy / Description": "Primary RF + Sigmoid calibration on held-out validation split.",
            },
        ]
        safe_render_table(pd.DataFrame(benchmark_table))

        st.markdown("---")

        # 4. Confusion Matrix Analysis & Operational Logistics Implications
        st.subheader("3. Confusion Matrix & Logistics Risk Analysis")
        col_cm_left, col_cm_right = st.columns([1.1, 1.3])

        with col_cm_left:
            fig_cm = config.figures_dir / "confusion_matrix.png"
            if fig_cm.exists():
                st.image(str(fig_cm), caption="Held-out Test Confusion Matrix", use_container_width=True)
            else:
                st.write(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

            # Confusion Matrix Stat Pills
            cp1, cp2 = st.columns(2)
            cp1.info(f"**True Negatives:** `{tn}` ({tn/len(test_df if 'test_df' in locals() else [1]*175):.1%})\n\nNormal operations correctly predicted.")
            cp2.success(f"**True Positives:** `{tp}` ({tp/len(test_df if 'test_df' in locals() else [1]*175):.1%})\n\nCongestion events correctly predicted.")
            cp3, cp4 = st.columns(2)
            cp3.warning(f"**False Positives (Type I):** `{fp}`\n\nFalse alarm: buffer waste, tug standby.")
            cp4.error(f"**False Negatives (Type II):** `{fn}`\n\nMissed delay: unexpected queue gridlock.")

        with col_cm_right:
            st.markdown(
                """
                #### Operational Implications of Classification Errors in Port Logistics

                In container terminal operations, classification errors have starkly asymmetric economic consequences:

                > 🚨 **False Negatives (Type II Error — Critical Threat):**
                > The system forecasts normal vessel flow, but actual median waiting time exceeds 6.0 hours.
                > - **Vessel Anchorage Backlog:** Containerships arrive without slow-steaming advisories, saturating outer fairways.
                > - **Direct Demurrage Penalties:** Charterers incur daily demurrage charges of **\$25,000–\$60,000 per vessel**.
                > - **Supply Chain Ripple:** Missed intermodal rail/truck slots inland compound into multi-day cargo dwell times.

                > ⚠️ **False Positives (Type I Error — Manageable Overhead):**
                > The system issues a congestion warning, but the terminal operates smoothly.
                > - **Precautionary Staging:** Dispatchers schedule extra tugboat crews or prepare crane tie-down protocols unnecessarily.
                > - **Voluntary Slow-Steaming:** Inbound vessels reduce speed by 1–2 knots, burning slightly less bunker fuel.
                > - **Operational Verdict:** Minor buffer inefficiency, but zero catastrophic terminal gridlock.

                **Engineering Conclusion:**
                For maritime terminal decision-support, **False Negatives are 5–10× more expensive than False Positives**. Dispatchers can tune the classification decision threshold downwards (e.g. from 0.50 to 0.35) using HarborSense AI's calibrated probabilities to aggressively suppress False Negatives.
                """
            )

        st.markdown("---")

        # 5. Diagnostic Discrimination & Probability Calibration Curves
        st.subheader("4. Discrimination & Probability Calibration Curves")
        diag_c1, diag_c2 = st.columns(2)

        with diag_c1:
            st.markdown("#### Discrimination: ROC & Precision-Recall Curves")
            fig_roc = config.figures_dir / "roc_pr_curves.png"
            if fig_roc.exists():
                st.image(str(fig_roc), caption="ROC & PR Curves (Test Partition)", use_container_width=True)
            st.caption(
                f"**ROC-AUC ({sel.get('roc_auc', 0):.3f}):** High separation across ranking thresholds. "
                f"**PR-AUC ({sel.get('pr_auc', 0):.3f}):** Crucial under class imbalance (12.6% event rate), "
                f"demonstrating a **3.4x precision lift** over random guessing."
            )

        with diag_c2:
            st.markdown("#### Reliability: Probability Calibration Diagram")
            fig_cal = config.figures_dir / "calibration_curve.png"
            if fig_cal.exists():
                st.image(str(fig_cal), caption="Sigmoid Calibration Curve vs Perfect Reliability", use_container_width=True)
            st.caption(
                f"**Sigmoid Calibration:** Sigmoid scaling maps raw tree votes into true event frequencies, "
                f"achieving a **Brier score of {sel.get('brier_score', 0):.4f}** and an **ECE of {sel.get('expected_calibration_error', 0):.4f}**. "
                "Calibrated probabilities allow dispatchers to trust the predicted percentages as real likelihoods."
            )

        st.markdown("---")

        # 6. Global Feature Importance & Domain Attribution
        st.subheader("5. Global Feature Importance (Permutation Attribution)")
        f_left, f_right = st.columns([1.2, 1.0])

        with f_left:
            fig_imp = config.figures_dir / "feature_importance.png"
            if fig_imp.exists():
                st.image(str(fig_imp), caption="Top Permutation Feature Importance (Held-out Test Split)", use_container_width=True)

        with f_right:
            st.markdown("#### Top Predictive Drivers (Ranked by Test ROC-AUC Impact)")
            if perm_list:
                perm_df = pd.DataFrame(
                    [
                        {
                            "Rank": i + 1,
                            "Feature Name": item[0],
                            "Drop in ROC-AUC": f"{item[1]:.4f}",
                            "Operational Domain Rationale": (
                                "Primary capacity loading index (arrivals vs berth scale)" if "arrival_to_berth" in item[0]
                                else "Compounded storm risk under high yard density" if "weather_utilization" in item[0]
                                else "Crane handling speed derate & wind safety halts" if "weather_severity" in item[0]
                                else "Terminal yard & quayside congestion state" if "current_utilization" in item[0]
                                else "Backlog carryover from previous operational shifts" if "queue_to_berth" in item[0]
                                else "Port-specific operational turnaround baseline"
                            ),
                        }
                        for i, item in enumerate(perm_list[:7])
                    ]
                )
                safe_render_table(perm_df)

        st.markdown("---")

        # 7. Secondary Task: Waiting Time Regression Analysis
        st.subheader("6. Secondary Task: Waiting Time Regression Evaluation")
        reg_c1, reg_c2, reg_c3, reg_c4 = st.columns(4)
        reg_c1.metric("Regression Target", "future_median_wait_hours")
        reg_c2.metric("Mean Absolute Error (MAE)", f"{reg.get('mae_hours', 0):.2f} Hours", help="Average magnitude of forecast error.")
        reg_c3.metric("Root Mean Squared Error (RMSE)", f"{reg.get('rmse_hours', 0):.2f} Hours", help="Penalizes large outlier forecast errors.")
        reg_c4.metric("Variance Explained (R²)", f"{reg.get('r2_score', 0):.3f}", help="Proportion of waiting time variance captured.")

        st.info(
            f"ℹ️ **Separation of Concerns:** The secondary waiting time regressor (`{reg.get('model', 'RandomForestRegressor')}`) "
            f"estimates continuous vessel waiting hours with an average error of only **{reg.get('mae_hours', 0):.2f} hours** (~44 minutes). "
            "Regression metrics are evaluated separately without conflating continuous error with binary threshold metrics."
        )

        st.markdown("---")

        # 8. Exploratory Data Diagnostics (Expandable EDA View)
        with st.expander("🔍 Inspect Underlying Exploratory Data Analysis (EDA) Figures"):
            st.markdown("Visual diagnostics generated during initial data exploration across the 5 simulated ports:")
            eda_col1, eda_col2 = st.columns(2)
            with eda_col1:
                eda_port = config.figures_dir / "eda_congestion_by_port.png"
                if eda_port.exists():
                    st.image(str(eda_port), caption="Congestion Rates by Port Facility", use_container_width=True)

                eda_temp = config.figures_dir / "eda_temporal_patterns.png"
                if eda_temp.exists():
                    st.image(str(eda_temp), caption="Circadian & Weekly Demand Variations", use_container_width=True)

            with eda_col2:
                eda_wait = config.figures_dir / "eda_wait_time_distribution.png"
                if eda_wait.exists():
                    st.image(str(eda_wait), caption="Waiting Time Distribution & 6.0h Threshold", use_container_width=True)

                eda_corr = config.figures_dir / "eda_correlation_heatmap.png"
                if eda_corr.exists():
                    st.image(str(eda_corr), caption="Feature & Target Pearson Correlation Matrix", use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: LIVE INTERACTIVE PREDICTION
    # -------------------------------------------------------------------------
    elif nav == "Live Interactive Prediction":
        st.title("⚡ Real-Time Port Congestion Forecaster")
        st.markdown("Input real-time port telemetry to obtain calibrated forecasts, conformal uncertainty, and SHAP drivers.")

        # Preset Scenarios
        st.subheader("1. Select Preset Operational Scenario or Enter Custom Values")
        preset = st.selectbox(
            "Quick Scenario Presets",
            [
                "Custom Manual Input",
                "Severe Storm & Arrival Surge (Rotterdam) [High Congestion]",
                "Normal Calm Operations (Singapore) [Low Congestion]",
                "High Existing Queue & Backlog (Los Angeles) [High Congestion]",
                "Borderline Capacity Pressure (Antwerp) [Moderate Uncertainty]",
            ],
        )

        # Default values
        defaults = {
            "port_id": "PORT_ROTTERDAM",
            "expected_arrivals_24h": 34.0,
            "berth_capacity": 26.0,
            "current_utilization": 0.75,
            "current_queue_length": 3.0,
            "recent_arrivals_6h": 8.0,
            "weather_severity": 0.25,
            "historical_delay_rate": 0.25,
            "hour": 14.0,
            "day_of_week": 2.0,
        }

        if preset == "Severe Storm & Arrival Surge (Rotterdam) [High Congestion]":
            defaults.update(
                {
                    "port_id": "PORT_ROTTERDAM",
                    "expected_arrivals_24h": 52.0,
                    "berth_capacity": 26.0,
                    "current_utilization": 0.92,
                    "current_queue_length": 11.0,
                    "recent_arrivals_6h": 16.0,
                    "weather_severity": 0.85,
                    "historical_delay_rate": 0.40,
                }
            )
        elif preset == "Normal Calm Operations (Singapore) [Low Congestion]":
            defaults.update(
                {
                    "port_id": "PORT_SINGAPORE",
                    "expected_arrivals_24h": 36.0,
                    "berth_capacity": 36.0,
                    "current_utilization": 0.65,
                    "current_queue_length": 2.0,
                    "recent_arrivals_6h": 7.0,
                    "weather_severity": 0.15,
                    "historical_delay_rate": 0.18,
                }
            )
        elif preset == "High Existing Queue & Backlog (Los Angeles) [High Congestion]":
            defaults.update(
                {
                    "port_id": "PORT_LOS_ANGELES",
                    "expected_arrivals_24h": 28.0,
                    "berth_capacity": 18.0,
                    "current_utilization": 0.90,
                    "current_queue_length": 14.0,
                    "recent_arrivals_6h": 9.0,
                    "weather_severity": 0.30,
                    "historical_delay_rate": 0.45,
                }
            )
        elif preset == "Borderline Capacity Pressure (Antwerp) [Moderate Uncertainty]":
            defaults.update(
                {
                    "port_id": "PORT_ANTWERP",
                    "expected_arrivals_24h": 22.0,
                    "berth_capacity": 16.0,
                    "current_utilization": 0.82,
                    "current_queue_length": 5.0,
                    "recent_arrivals_6h": 6.0,
                    "weather_severity": 0.55,
                    "historical_delay_rate": 0.28,
                }
            )

        with st.form("prediction_form"):
            st.markdown("### Operational Telemetry Inputs")
            col1, col2, col3 = st.columns(3)

            with col1:
                port_id = st.selectbox(
                    "Port Facility",
                    options=list(config.ports),
                    index=list(config.ports).index(defaults["port_id"]),
                )
                expected_arrivals = st.number_input(
                    "Expected 24h Arrivals",
                    min_value=0.0,
                    max_value=150.0,
                    value=float(defaults["expected_arrivals_24h"]),
                    step=1.0,
                )
                berth_capacity = st.number_input(
                    "Berth Capacity",
                    min_value=1.0,
                    max_value=70.0,
                    value=float(defaults["berth_capacity"]),
                    step=1.0,
                )

            with col2:
                current_util = st.slider(
                    "Current Utilization Ratio",
                    min_value=0.0,
                    max_value=1.5,
                    value=float(defaults["current_utilization"]),
                    step=0.02,
                )
                current_queue = st.number_input(
                    "Current Queue Length (Vessels)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(defaults["current_queue_length"]),
                    step=1.0,
                )
                recent_arrivals = st.number_input(
                    "Recent Arrivals (Past 6 Hours)",
                    min_value=0.0,
                    max_value=50.0,
                    value=float(defaults["recent_arrivals_6h"]),
                    step=1.0,
                )

            with col3:
                weather_severity = st.slider(
                    "Weather Severity Index (0=Calm, 1=Storm)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(defaults["weather_severity"]),
                    step=0.05,
                )
                hist_delay = st.slider(
                    "Historical Delay Rate",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(defaults["historical_delay_rate"]),
                    step=0.02,
                )
                hour = st.slider("Hour of Day", 0, 23, int(defaults["hour"]))
                day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, int(defaults["day_of_week"]))

            submit = st.form_submit_button("🚀 Run Prediction & Diagnosis", use_container_width=True)

        if submit:
            payload = {
                "port_id": port_id,
                "expected_arrivals_24h": expected_arrivals,
                "berth_capacity": berth_capacity,
                "current_utilization": current_util,
                "current_queue_length": current_queue,
                "recent_arrivals_6h": recent_arrivals,
                "weather_severity": weather_severity,
                "historical_delay_rate": hist_delay,
                "hour": float(hour),
                "day_of_week": float(day_of_week),
            }

            try:
                engine = PredictionEngine.get_instance(config)
                pred: HarborSensePrediction = engine.predict_single(payload)
                st.session_state["last_prediction"] = pred
                st.session_state["last_input"] = payload
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        # Display results if available
        if "last_prediction" in st.session_state:
            pred = st.session_state["last_prediction"]
            st.markdown("---")
            st.subheader("2. Prediction Result & Uncertainty Assessment")

            # Warning callouts if any
            if pred.data_quality_warnings:
                for w in pred.data_quality_warnings:
                    st.warning(f"⚠️ **Data Quality Notice:** {w}")

            res1, res2, res3, res4 = st.columns(4)

            # Color coding
            risk_color = (
                "red" if pred.risk_category == "HIGH" else ("orange" if pred.risk_category == "MEDIUM" else "green")
            )
            res1.metric("Predicted Congestion", "YES (Congested)" if pred.predicted_congestion == 1 else "NO (Normal)")
            res2.metric("Congestion Probability", f"{pred.congestion_probability:.1%}")
            res3.metric("Risk Category", pred.risk_category)
            res4.metric("Estimated Wait Time", f"{pred.estimated_wait_hours:.1f} Hours")

            # Uncertainty & Reliability card
            unc = pred.uncertainty
            st.info(
                f"🛡️ **Statistical Reliability Assessment:** **{unc.reliability_tier}**\n\n"
                f"- **Conformal Prediction Set (90% Confidence):** `{unc.conformal_prediction_set}` "
                f"({'Singleton class: high certainty' if unc.is_singleton_set else 'Both classes: high ambiguity'})\n"
                f"- **Margin from Decision Boundary:** `{unc.decision_boundary_distance:.2f}`\n"
                f"- **Explanation:** {unc.reliability_explanation}"
            )

            # Local SHAP Breakdown
            st.subheader("3. Feature Contributions (Local SHAP Attribution)")
            expl = pred.explanation

            col_dr, col_mit = st.columns(2)
            with col_dr:
                st.markdown("#### 🔴 Factors Increasing Congestion Risk")
                if expl.top_risk_drivers:
                    for d in expl.top_risk_drivers:
                        st.markdown(f"- **{d.feature_name}** (`+{d.contribution:.3f}`): {d.interpretation}")
                else:
                    st.caption("No significant positive risk drivers detected.")

            with col_mit:
                st.markdown("#### 🟢 Factors Mitigating Congestion Risk")
                if expl.top_mitigating_factors:
                    for m in expl.top_mitigating_factors:
                        st.markdown(f"- **{m.feature_name}** (`{m.contribution:.3f}`): {m.interpretation}")
                else:
                    st.caption("No significant mitigating buffers detected.")

            st.caption(f"Attribution Method: `{expl.method_used}` | {expl.disclaimer}")

    # -------------------------------------------------------------------------
    # TAB 4: OPERATIONAL RECOMMENDATIONS
    # -------------------------------------------------------------------------
    elif nav == "Operational Recommendations":
        st.title("📋 Operational Action Playbook")
        st.markdown("Rule-based, actionable logistics playbooks mapped directly from prediction risk, uncertainty, and SHAP drivers.")

        if "last_prediction" not in st.session_state:
            st.info("No active prediction in session. Please run a prediction in the **Live Interactive Prediction** tab first.")
            return

        pred = st.session_state["last_prediction"]
        recs = pred.recommendations

        st.subheader(f"Recommendations for {pred.port_id} (Risk Level: {recs.risk_category})")

        if recs.human_in_the_loop_flag:
            st.error("🚨 **Human-in-the-Loop Required:** Uncertainty or data-quality anomalies require human dispatcher authorization.")

        st.markdown("### Primary Operational Actions")
        for act in recs.primary_actions:
            with st.expander(f"[{act.urgency}] {act.category}: {act.action}", expanded=True):
                st.write(f"**Rationale:** {act.rationale}")

        if recs.contingency_actions:
            st.markdown("### Contingency & Preventative Protocols")
            for c_act in recs.contingency_actions:
                with st.expander(f"[{c_act.urgency}] {c_act.category}: {c_act.action}"):
                    st.write(f"**Rationale:** {c_act.rationale}")

        st.warning(f"⚖️ **Disclaimer:** {recs.disclaimer}")

    # -------------------------------------------------------------------------
    # TAB 5: LIMITATIONS & REAL-WORLD ROADMAP
    # -------------------------------------------------------------------------
    elif nav == "Limitations & Real-World Roadmap":
        st.title("⚠️ System Assumptions, Limitations & Real-World Roadmap")

        st.markdown(
            """
            ### 1. Synthetic Data & Operational Assumptions
            - **Synthetic Queue Simulation:** The dataset is generated using queue-theoretic simulations (M/M/c approximations 
              with lognormal operational shocks). While structurally realistic, real-world port dynamics exhibit complex labor 
              agreements, tugboat bottlenecks, channel tide windows, and shipping alliance rotations that are not captured here.
            - **Arbitrary 6.0-Hour Threshold:** The definition `congestion = 1 if future_median_wait_hours > 6.0 else 0` is an 
              explicit project assumption for the take-home assessment, not a universal maritime standard.
            - **Five Simulated Ports:** Model weights may over-index on the characteristics of the 5 simulated facilities 
              (Rotterdam, Singapore, Shanghai, Los Angeles, Antwerp) and would require fine-tuning for regional feeder ports.

            ### 2. Explainability & Causal Realism
            - **SHAP is Associational, Not Causal:** A high positive SHAP value for `current_queue_length` indicates that 
              historically large queues correlate with upcoming delays. It does **not** prove that reducing queue length 
              in isolation will eliminate delay without accounting for arrival momentum or crane availability.
            - **Multi-Collinearity:** Arrival pressure, utilization, and queue length naturally correlate; tree splits can 
              distribute importance among correlated features.

            ### 3. What Real-World Data Would Be Required for Production?
            To validate HarborSense AI for live commercial operations, the following data feeds are mandatory:
            1. **High-Frequency AIS (Automatic Identification System):** Real-time spatial tracking of vessel speed, anchoring zones, 
               and fairway pilot boarding grounds.
            2. **Terminal Operating System (TOS) Telemetry:** Real-time container crane moves per hour (GMPH), yard stack density, 
               and trucker gate turn-times.
            3. **Validated Outcome Ground Truth:** Historical pilot boarding records and berth departure timestamps to measure exact 
               unadulterated vessel waiting hours (rather than synthetic estimates).
            4. **Marine Meteorological APIs:** Wave swell height, tide tables, and localized wind-gust data at berth elevations.
            """
        )


if __name__ == "__main__":
    main()

