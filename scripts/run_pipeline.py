"""End-to-end pipeline execution script for HarborSense AI.

Orchestrates:
1. Synthetic dataset generation and validation
2. Chronological splitting with purge gap
3. Preprocessing, model training, and probability calibration
4. Held-out test evaluation, metric logging, and figure generation
5. Technical report generation (Markdown & PDF)
6. Sample prediction verification
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add src to pythonpath
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

from harborsense.config import config
from harborsense.data_generation import generate_dataset
from harborsense.eda import run_eda
from harborsense.evaluate import evaluate_all
from harborsense.predict import PredictionEngine
from harborsense.report_generator import generate_pdf_report, generate_markdown_report
from harborsense.train import train_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_pipeline")


def run_full_pipeline() -> None:
    """Execute complete HarborSense AI pipeline."""
    logger.info("=== STEP 1: DATA GENERATION & EDA ===")
    df = generate_dataset(n_steps_per_port=300, seed=config.random_seed, output_path=config.data_file)
    logger.info("Dataset ready with %d rows across %d ports.", len(df), df["port_id"].nunique())
    eda_stats = run_eda(df=df, cfg=config)
    logger.info("EDA completed: Overall Congestion Rate = %.1f%%", eda_stats["overall_congestion_rate"] * 100)

    logger.info("=== STEP 2: MODEL TRAINING & CALIBRATION ===")
    train_meta = train_models(df=df, cfg=config)
    logger.info("Models trained: %s", train_meta["model_type"])

    logger.info("=== STEP 3: EVALUATION & FIGURE GENERATION ===")
    metrics = evaluate_all(cfg=config)
    logger.info(
        "Evaluation finished. Test Accuracy: %.1f%%, F1: %.3f, ROC-AUC: %.3f",
        metrics["selected_model"]["accuracy"] * 100,
        metrics["selected_model"]["f1_score"],
        metrics["selected_model"]["roc_auc"],
    )

    logger.info("=== STEP 4: VERIFY PREDICTION ENGINE ===")
    engine = PredictionEngine(config)
    sample_input = {
        "port_id": "PORT_ROTTERDAM",
        "expected_arrivals_24h": 42.0,
        "berth_capacity": 26.0,
        "current_utilization": 0.88,
        "current_queue_length": 8.0,
        "recent_arrivals_6h": 12.0,
        "weather_severity": 0.65,
        "historical_delay_rate": 0.35,
        "hour": 14.0,
        "day_of_week": 2.0,
    }
    pred = engine.predict_single(sample_input)
    logger.info(
        "Sample Prediction: Congestion=%d (Prob=%.1f%%, Risk=%s, Wait=%.1fh, Reliability=%s)",
        pred.predicted_congestion,
        pred.congestion_probability * 100,
        pred.risk_category,
        pred.estimated_wait_hours,
        pred.uncertainty.reliability_tier,
    )

    logger.info("=== STEP 5: TECHNICAL REPORT GENERATION ===")
    generate_markdown_report(cfg=config, sample_pred=pred)
    generate_pdf_report(cfg=config, sample_pred=pred)
    logger.info("Technical reports generated at %s and %s", config.report_md, config.report_pdf)

    logger.info("=== HARBORSENSE AI PIPELINE COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_full_pipeline()
