"""Evaluation and diagnostic metrics generator for HarborSense AI.

Evaluates on held-out chronological test split:
1. Baseline Comparisons:
   - Majority class baseline
   - Simple operational heuristic baseline
   - Logistic Regression baseline
   - Random Forest (uncalibrated)
   - Random Forest (calibrated with Sigmoid)
2. Classification Metrics:
   - Accuracy, Precision, Recall, F1, PR-AUC, ROC-AUC, Brier score, Confusion matrix, ECE.
3. Regression Metrics (Secondary Task):
   - MAE, RMSE, R-squared for future_median_wait_hours.
4. Visualization Generation:
   - Confusion matrix heatmap
   - ROC & Precision-Recall curves
   - Probability calibration reliability diagram
   - Global permutation feature importance
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)

from harborsense.config import ProjectConfig, config
from harborsense.preprocessing import HarborSensePreprocessor

logger = logging.getLogger(__name__)


def compute_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_mask = (y_prob > bin_limits[i]) & (y_prob <= bin_limits[i + 1])
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_all(
    cfg: ProjectConfig = config,
) -> Dict[str, Any]:
    """Run full evaluation suite on the held-out test split."""
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load splits and artifacts
    splits_dir = cfg.data_dir / "splits"
    if not (splits_dir / "test.csv").exists():
        raise FileNotFoundError("Test split not found. Run training pipeline first.")

    train_df = pd.read_csv(splits_dir / "train.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    preprocessor = HarborSensePreprocessor.load(cfg.model_dir / "preprocessor.joblib")
    baseline_lr = joblib.load(cfg.model_dir / "baseline_lr.joblib")
    primary_rf_uncal = joblib.load(cfg.model_dir / "primary_rf_uncalibrated.joblib")
    calibrated_rf = joblib.load(cfg.model_dir / "classifier.joblib")
    regressor = joblib.load(cfg.model_dir / "regressor.joblib")

    with open(cfg.model_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    X_test_trans = preprocessor.transform(test_df)
    y_test_cls = test_df[cfg.target_classification].values.astype(int)
    y_test_reg = test_df[cfg.target_regression].values.astype(float)

    n_test = len(y_test_cls)
    pos_count = int(np.sum(y_test_cls))
    neg_count = n_test - pos_count
    pos_rate = float(pos_count / n_test)

    # 2. Baseline 1: Majority Class
    majority_class = int(train_df[cfg.target_classification].mode()[0])
    y_pred_majority = np.full(n_test, majority_class)
    majority_acc = float(accuracy_score(y_test_cls, y_pred_majority))
    majority_f1 = float(f1_score(y_test_cls, y_pred_majority, zero_division=0))

    # Baseline 2: Operational Heuristic
    # Rule: Predict congestion if (queue >= 5 or utilization >= 0.85 or weather >= 0.70)
    heuristic_pred = (
        (test_df["current_queue_length"] >= 5)
        | (test_df["current_utilization"] >= 0.85)
        | (test_df["weather_severity"] >= 0.70)
    ).astype(int).values
    heuristic_acc = float(accuracy_score(y_test_cls, heuristic_pred))
    heuristic_f1 = float(f1_score(y_test_cls, heuristic_pred, zero_division=0))
    heuristic_prec = float(precision_score(y_test_cls, heuristic_pred, zero_division=0))
    heuristic_rec = float(recall_score(y_test_cls, heuristic_pred, zero_division=0))

    # Baseline 3: Logistic Regression
    lr_probs = baseline_lr.predict_proba(X_test_trans)[:, 1]
    lr_preds = (lr_probs >= 0.5).astype(int)
    lr_acc = float(accuracy_score(y_test_cls, lr_preds))
    lr_prec = float(precision_score(y_test_cls, lr_preds, zero_division=0))
    lr_rec = float(recall_score(y_test_cls, lr_preds, zero_division=0))
    lr_f1 = float(f1_score(y_test_cls, lr_preds, zero_division=0))
    try:
        lr_roc_auc = float(roc_auc_score(y_test_cls, lr_probs))
    except ValueError:
        lr_roc_auc = 0.5
    lr_brier = float(brier_score_loss(y_test_cls, lr_probs))

    # Primary Model: Uncalibrated RF
    rf_uncal_probs = primary_rf_uncal.predict_proba(X_test_trans)[:, 1]
    rf_uncal_preds = (rf_uncal_probs >= 0.5).astype(int)
    rf_uncal_brier = float(brier_score_loss(y_test_cls, rf_uncal_probs))
    rf_uncal_ece = compute_expected_calibration_error(y_test_cls, rf_uncal_probs)

    # Primary Model: Calibrated RF (Selected Model)
    cal_probs = calibrated_rf.predict_proba(X_test_trans)[:, 1]
    cal_preds = (cal_probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_test_cls, cal_preds))
    prec = float(precision_score(y_test_cls, cal_preds, zero_division=0))
    rec = float(recall_score(y_test_cls, cal_preds, zero_division=0))
    f1 = float(f1_score(y_test_cls, cal_preds, zero_division=0))
    try:
        roc_auc = float(roc_auc_score(y_test_cls, cal_probs))
    except ValueError:
        roc_auc = 0.5

    # Precision-Recall AUC
    try:
        p_curve, r_curve, _ = precision_recall_curve(y_test_cls, cal_probs)
        pr_auc = float(auc(r_curve, p_curve))
    except Exception:
        pr_auc = float(np.mean(y_test_cls))

    brier = float(brier_score_loss(y_test_cls, cal_probs))
    ece = compute_expected_calibration_error(y_test_cls, cal_probs)

    cm = confusion_matrix(y_test_cls, cal_preds, labels=[0, 1]).tolist()
    tn, fp, fn, tp = confusion_matrix(y_test_cls, cal_preds, labels=[0, 1]).ravel()

    # Secondary Task: Regression
    reg_preds = regressor.predict(X_test_trans)
    reg_mae = float(mean_absolute_error(y_test_reg, reg_preds))
    reg_rmse = float(np.sqrt(mean_squared_error(y_test_reg, reg_preds)))
    reg_r2 = float(r2_score(y_test_reg, reg_preds))

    # 3. Permutation Feature Importance
    logger.info("Computing test permutation importance...")
    perm_result = permutation_importance(
        calibrated_rf,
        X_test_trans,
        y_test_cls,
        n_repeats=10,
        random_state=cfg.random_seed,
        scoring="roc_auc",
    )
    perm_importance_dict = {
        name: float(imp)
        for name, imp in zip(feature_names, perm_result.importances_mean)
    }
    sorted_importances = sorted(perm_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # 4. Generate Figures
    _plot_confusion_matrix(cm, cfg.figures_dir / "confusion_matrix.png")
    _plot_roc_and_pr_curves(
        y_test_cls,
        cal_probs,
        roc_auc,
        pr_auc,
        cfg.figures_dir / "roc_pr_curves.png",
    )
    _plot_calibration_curve(
        y_test_cls,
        rf_uncal_probs,
        cal_probs,
        cfg.figures_dir / "calibration_curve.png",
    )
    _plot_feature_importance(
        sorted_importances[:12],
        cfg.figures_dir / "feature_importance.png",
    )

    # 5. Assemble Comprehensive Metrics Payload
    metrics_payload = {
        "dataset_summary": {
            "test_sample_count": n_test,
            "positive_congested_count": pos_count,
            "negative_normal_count": neg_count,
            "test_congestion_rate": float(np.round(pos_rate, 4)),
        },
        "selected_model": {
            "model_name": "Calibrated Random Forest (Sigmoid)",
            "accuracy": float(np.round(acc, 4)),
            "precision": float(np.round(prec, 4)),
            "recall": float(np.round(rec, 4)),
            "f1_score": float(np.round(f1, 4)),
            "roc_auc": float(np.round(roc_auc, 4)),
            "pr_auc": float(np.round(pr_auc, 4)),
            "brier_score": float(np.round(brier, 4)),
            "expected_calibration_error": float(np.round(ece, 4)),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
        },
        "model_comparisons": {
            "majority_class_baseline": {
                "strategy": f"Always predict {majority_class}",
                "accuracy": float(np.round(majority_acc, 4)),
                "f1_score": float(np.round(majority_f1, 4)),
            },
            "domain_heuristic_baseline": {
                "strategy": "Congestion if queue >= 5 or util >= 0.85 or weather >= 0.70",
                "accuracy": float(np.round(heuristic_acc, 4)),
                "precision": float(np.round(heuristic_prec, 4)),
                "recall": float(np.round(heuristic_rec, 4)),
                "f1_score": float(np.round(heuristic_f1, 4)),
            },
            "logistic_regression_baseline": {
                "accuracy": float(np.round(lr_acc, 4)),
                "precision": float(np.round(lr_prec, 4)),
                "recall": float(np.round(lr_rec, 4)),
                "f1_score": float(np.round(lr_f1, 4)),
                "roc_auc": float(np.round(lr_roc_auc, 4)),
                "brier_score": float(np.round(lr_brier, 4)),
            },
            "random_forest_uncalibrated": {
                "brier_score": float(np.round(rf_uncal_brier, 4)),
                "expected_calibration_error": float(np.round(rf_uncal_ece, 4)),
            },
            "random_forest_calibrated": {
                "brier_score": float(np.round(brier, 4)),
                "expected_calibration_error": float(np.round(ece, 4)),
            },
        },
        "secondary_regression_task": {
            "target": "future_median_wait_hours",
            "model": "RandomForestRegressor(n_estimators=100)",
            "mae_hours": float(np.round(reg_mae, 3)),
            "rmse_hours": float(np.round(reg_rmse, 3)),
            "r2_score": float(np.round(reg_r2, 4)),
        },
        "top_features_by_permutation_importance": sorted_importances[:10],
    }

    with open(cfg.metrics_file, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info("Evaluation complete. Metrics saved to %s", cfg.metrics_file)
    return metrics_payload


def _plot_confusion_matrix(cm: List[List[int]], output_path: Path) -> None:
    """Generate clean confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    cm_arr = np.array(cm)
    cax = ax.matshow(cm_arr, cmap="Blues", alpha=0.85)

    for i in range(2):
        for j in range(2):
            val = cm_arr[i, j]
            color = "white" if val > cm_arr.max() / 2 else "black"
            ax.text(j, i, f"{val}\n({val / cm_arr.sum():.1%})", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal (0)", "Congested (1)"], fontsize=10)
    ax.set_yticklabels(["Normal (0)", "Congested (1)"], fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Ground Truth", fontsize=11, labelpad=8)
    ax.set_title("Test Confusion Matrix", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_roc_and_pr_curves(
    y_true: np.ndarray, y_prob: np.ndarray, roc_auc: float, pr_auc: float, output_path: Path
) -> None:
    """Plot joint ROC and PR curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax1.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax1.set_xlabel("False Positive Rate", fontsize=10)
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=10)
    ax1.set_title("Receiver Operating Characteristic", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    base_rate = np.mean(y_true)
    ax2.plot(rec, prec, color="#ff7f0e", lw=2, label=f"PR (AUC = {pr_auc:.3f})")
    ax2.axhline(base_rate, color="gray", linestyle="--", lw=1, label=f"Baseline ({base_rate:.1%})")
    ax2.set_xlabel("Recall", fontsize=10)
    ax2.set_ylabel("Precision", fontsize=10)
    ax2.set_title("Precision-Recall Curve", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_calibration_curve(
    y_true: np.ndarray, uncal_prob: np.ndarray, cal_prob: np.ndarray, output_path: Path
) -> None:
    """Plot calibration reliability curves comparing uncalibrated and calibrated probabilities."""
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)

    prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, uncal_prob, n_bins=8)
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, cal_prob, n_bins=8)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", lw=1.2)
    ax.plot(prob_pred_uncal, prob_true_uncal, "s-", color="#d62728", lw=1.5, label="RF Uncalibrated")
    ax.plot(prob_pred_cal, prob_true_cal, "o-", color="#2ca02c", lw=2, label="RF Calibrated (Sigmoid)")

    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Empirical Event Fraction", fontsize=10)
    ax.set_title("Reliability Diagram (Probability Calibration)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_feature_importance(
    importances: List[Tuple[str, float]], output_path: Path
) -> None:
    """Plot top permutation feature importances."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    names = [x[0] for x in reversed(importances)]
    values = [x[1] for x in reversed(importances)]

    bars = ax.barh(names, values, color="#1f77b4", alpha=0.85)
    ax.set_xlabel("Mean Drop in ROC-AUC upon Permutation", fontsize=10)
    ax.set_title("Top Feature Importances (Permutation on Held-out Test Set)", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    metrics = evaluate_all()
    print(json.dumps(metrics, indent=2))

