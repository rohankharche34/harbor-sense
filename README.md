# HarborSense AI: Port Congestion and Shipment Delay Prediction System

![HarborSense Banner](https://images.unsplash.com/photo-1578575437130-527eed3abbec?auto=format&fit=crop&w=1200&q=80)

HarborSense AI is a machine learning decision-support platform designed to forecast port congestion and vessel delays 24 hours in advance. Developed for maritime logistics operators and port dispatchers, HarborSense AI provides calibrated probabilities, distribution-free uncertainty intervals (conformal prediction), directional feature attributions (TreeSHAP), and actionable operational recommendations.

---

## 1. Project Overview & Problem Formulation

### 1.1 The Operational Challenge
Maritime container terminals and ports are stochastic queuing systems subject to vessel arrival surges, yard capacity bottlenecks, pilotage availability, and weather disruptions. Port delays cost ocean carriers and shippers billions of dollars each year in demurrage, fuel consumption, and disrupted supply chains.

### 1.2 Target Definition & Prediction Horizon
- **Observation Unit:** A specific port facility at a designated prediction timestamp $t$.
- **Prediction Horizon:** Next 24 hours ($t$ to $t + 24\text{h}$).
- **Primary Classification Target:**
  $$\text{congestion} = 1 \iff \text{future\_median\_wait\_hours} > 6.0$$
  Otherwise, $\text{congestion} = 0$.
  *(The 6.0-hour waiting threshold is a designated project assumption for the assessment).*
- **Secondary Regression Task:** Estimation of continuous `future_median_wait_hours` in hours, evaluated independently using MAE and RMSE without conflating classification metrics.
- **Strict Leakage Prevention:** Predictions strictly utilize information available at or before timestamp $t$. Future waiting times, outcomes, and derived metrics never enter the feature matrix.

---

## 2. System Architecture & Data Flow

```
                      ┌───────────────────────────────────────┐
                      │    Port Telemetry & AIS Feeds (t)    │
                      └──────────────────┬────────────────────┘
                                         │
                                         ▼
                      ┌───────────────────────────────────────┐
                      │ Input Validation & Domain Bounds Check│
                      │   (Missing, Range & Outlier Guards)   │
                      └──────────────────┬────────────────────┘
                                         │
                                         ▼
                      ┌───────────────────────────────────────┐
                      │   Maritime Feature Engineering Pipeline │
                      │ (Arrival/Berth ratios, Queue pressure,│
                      │  Weather interaction, Cyclical time)  │
                      └──────────────────┬────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
  ┌─────────────────────────────┐                 ┌─────────────────────────────┐
  │ Primary Classifier Pipeline │                 │ Secondary Regressor Pipeline│
  │ (Calibrated Random Forest)  │                 │   (Random Forest Regressor) │
  └──────────────┬──────────────┘                 └──────────────┬──────────────┘
                 │                                               │
                 ├───────────────────────┬───────────────────────┤
                 ▼                       ▼                       ▼
  ┌─────────────────────────────┐ ┌──────────────┐ ┌─────────────────────────────┐
  │ Split Conformal Uncertainty │ │ Local TreeSHAP│ │   Operational Action Plan   │
  │   (90% Prediction Sets)     │ │ Attributions │ │  (Traffic, Berth, Weather)  │
  └──────────────┬──────────────┘ └──────┬───────┘ └─────────────┬───────────────┘
                 │                       │                       │
                 └───────────────────────┼───────────────────────┘
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │ Unified Prediction Engine: FastAPI & Dashboard│
                 └───────────────────────────────────────────────┘
```

---

## 3. Technology Stack

- **Python 3.11** runtime
- **Data & Preprocessing:** `NumPy`, `pandas`, `scikit-learn` (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer`)
- **Machine Learning:** `scikit-learn` (`RandomForestClassifier`, `LogisticRegression`, `CalibratedClassifierCV`, `RandomForestRegressor`)
- **Explainability:** `SHAP` (`TreeExplainer` with perturbation-based fallback)
- **Uncertainty Quantification:** Split Conformal Prediction (Inductive Conformal Prediction) with 90% marginal coverage guarantee
- **Backend API:** `FastAPI`, `Uvicorn`, `Pydantic v2`
- **Frontend Dashboard:** `Streamlit`
- **Testing:** `pytest`, `pytest-cov`, `httpx`
- **PDF & Document Generation:** `ReportLab`
- **Environment & Automation:** Python Virtual Environment (`venv`), `Makefile`

*(No Docker, Kubernetes, or cloud infrastructure used; fully local and reproducible).*

---

## 4. Repository Structure

```
harborsense-ai/
├── README.md                      # Complete system documentation
├── DEMO.md                        # 5-minute evaluation demonstration script
├── requirements.txt               # Pinned Python dependencies
├── pyproject.toml                 # Packaging & pytest configuration
├── .gitignore                     # Git ignore patterns
├── .env.example                   # Environment variable template
├── Makefile                       # Developer automation commands
├── data/
│   ├── README.md                  # Data dictionary and simulation assumptions
│   ├── port_congestion.csv        # Synthetic dataset (1,500 observations)
│   └── splits/                    # Chronological train/val/test partitions
├── src/
│   └── harborsense/
│       ├── __init__.py
│       ├── config.py              # Centralized configuration & thresholds
│       ├── data_generation.py     # Queue-theoretic synthetic data generator
│       ├── data_validation.py     # Schema, domain bounds, and leakage guards
│       ├── features.py            # Maritime feature engineering
│       ├── preprocessing.py       # ColumnTransformer & purged split builder
│       ├── eda.py                 # Exploratory data analysis & figure generator
│       ├── train.py               # Baseline & primary model training & calibration
│       ├── evaluate.py            # Held-out test evaluation & diagnostic plots
│       ├── predict.py             # Unified inference engine
│       ├── uncertainty.py         # Split conformal prediction & OOD checks
│       ├── explain.py             # TreeSHAP & perturbation explainability
│       ├── recommendations.py     # Prescriptive operational playbooks
│       └── report_generator.py    # ReportLab 2-3 page PDF & Markdown generator
├── api/
│   └── main.py                    # FastAPI application & Pydantic models
├── dashboard/
│   └── app.py                     # Streamlit interactive analyst dashboard
├── notebooks/
│   └── exploratory_analysis.ipynb # Interactive EDA Jupyter Notebook
├── models/                        # Serialized fitted artifacts & metadata
├── tests/
│   ├── test_data.py               # Data generation, schema & leakage tests
│   ├── test_pipeline.py           # Preprocessing, training & conformal tests
│   ├── test_explanations.py       # SHAP attributions & recommendation tests
│   └── test_api.py                # FastAPI endpoint integration tests
├── scripts/
│   └── run_pipeline.py            # Master end-to-end execution script
└── reports/
    ├── technical_report.md        # Comprehensive technical report in Markdown
    ├── harborsense_technical_report.pdf # Polished 2-3 page PDF technical report
    ├── technical_report.pdf       # Mirrored copy of PDF technical report
    ├── metrics.json               # Actual held-out test evaluation metrics
    └── figures/                   # Diagnostic & EDA visualizations
```

---

## 5. Installation & Setup

### 5.1 Prerequisites
- Linux OS (or macOS/WSL2)
- Python 3.11 installed (`python3.11 --version`)

### 5.2 Virtual Environment Setup
Clone the repository and set up a clean Python 3.11 virtual environment:

```bash
# 1. Create virtual environment
python3.11 -m venv .venv

# 2. Activate environment
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Alternatively, use the Makefile:
```bash
make install
```

---

## 6. End-to-End Pipeline Execution

Run the complete pipeline (data generation, EDA, training, calibration, evaluation, report compilation):

```bash
source .venv/bin/activate
python scripts/run_pipeline.py
```
Or via Makefile:
```bash
make pipeline
```

### Execution Steps Breakdown
1. **Data Generation & EDA:** Generates 1,500 rows across 5 ports and saves `data/port_congestion.csv`. Produces EDA figures in `reports/figures/`.
2. **Model Training:** Splits data chronologically (70/15/15) with a 24-hour purge gap. Trains Logistic Regression baseline, Random Forest primary classifier, fits Sigmoid calibration on validation data, trains secondary wait-time regressor, and fits Conformal Calibrator.
3. **Evaluation:** Evaluates all models on held-out test data. Generates confusion matrix, ROC/PR curves, and calibration curves. Writes `reports/metrics.json`.
4. **Deliverables:** Compiles `reports/technical_report.md` and the polished 2-3 page PDF report at `reports/harborsense_technical_report.pdf` (mirrored to `reports/technical_report.pdf`).

### 6.1 Compiling the Technical Report
To compile or regenerate the PDF technical report directly:
```bash
source .venv/bin/activate
python -m harborsense.report_generator
# Or: make report
```
- **Output Report:** `reports/harborsense_technical_report.pdf` (2–3 page PDF document)
- **Output Summary:** `reports/technical_report.md` (Markdown version)

---

## 7. Running the Applications

### 7.1 FastAPI Prediction Backend
Launch the high-performance prediction API:

```bash
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Or: make api
```

- **Interactive OpenAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative Redoc Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

#### Example API Request (cURL)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "port_id": "PORT_ROTTERDAM",
       "expected_arrivals_24h": 42.0,
       "berth_capacity": 26.0,
       "current_utilization": 0.88,
       "current_queue_length": 8.0,
       "recent_arrivals_6h": 12.0,
       "weather_severity": 0.65,
       "historical_delay_rate": 0.35,
       "hour": 14.0,
       "day_of_week": 2.0
     }'
```

### 7.2 Streamlit Logistics Analyst Dashboard
Launch the visual web dashboard:

```bash
source .venv/bin/activate
streamlit run dashboard/app.py --server.port 8501
# Or: make dashboard
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 8. Automated Testing

Execute the comprehensive automated test suite with coverage:

```bash
source .venv/bin/activate
pytest tests/ -v
# Or: make test
```

The test suite covers:
- Synthetic dataset generation reproducibility and schema compliance.
- Absence of nulls, duplicates, and plausible feature bounds.
- Congestion threshold consistency (`future_median_wait_hours > 6.0`).
- Target leakage prevention and input feature sanitization.
- Chronological splitting and temporal purge gap enforcement.
- Preprocessing pipeline fitting and feature name tracking.
- Baseline and primary model training and serialization.
- Probability calibration and Brier score loss assessment.
- Split conformal prediction set validity ({0}, {1}, or {0, 1}).
- Local explanation generation (TreeSHAP and perturbation fallback).
- Prescriptive operational recommendations logic.
- FastAPI endpoints (`/health`, `/metadata`, `/predict`, `/predict/batch`, `/metrics`).

---

## 9. Evaluation Results (Held-Out Test Partition)

*Evaluated on 175 strictly chronological held-out test observations with a 24-hour purge gap:*

| Model / Strategy | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Brier Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class Baseline** | 87.4% | — | 0.0% | 0.000 | 0.500 | — |
| **Operational Heuristic** | 80.6% | 32.4% | 50.0% | 0.393 | — | — |
| **Baseline Logistic Regression** | 86.9% | 47.6% | 45.5% | 0.465 | 0.897 | 0.0825 |
| **Calibrated Random Forest (Selected)** | **88.0%** | **54.5%** | **27.3%** | **0.364** | **0.874** | **0.0895** |

- **Held-Out Test Confusion Matrix:** True Negatives = 148 | False Positives = 5 | False Negatives = 16 | True Positives = 6
- **Calibration Assessment:** Expected Calibration Error (ECE) = 0.0586 | Brier Score = 0.0895 (Sigmoid calibrated)
- **Precision-Recall Performance:** PR-AUC = 0.434 (a 3.4x lift over the 12.6% test event prevalence)

### Secondary Regression Task: Waiting Time
- **Model:** `RandomForestRegressor(n_estimators=100)`
- **Mean Absolute Error (MAE):** `0.73 hours` (~44 minutes)
- **Root Mean Squared Error (RMSE):** `0.91 hours`
- **R² Variance Explained:** `0.540`

---

## 10. Operational Recommendations Engine

The system maps predictions into clear, prioritized decision-support playbooks:
- **High Risk + Arrival Surge:** Issue virtual queuing / vessel slow-steaming advisories to smooth incoming wave.
- **High Risk + Severe Weather ($\ge 0.6$):** Mobilize heavy weather crane tie-down protocol; shift crane labor to yard stacking.
- **High Risk + Backlog Queue ($\ge 8$ vessels):** Prioritize bottleneck berths and reallocate auxiliary tugs to expedite vessel turnarounds.
- **Low Reliability / Anomalous Telemetry:** Escalate to senior human dispatcher for telemetry verification.
- **Low Risk:** Maintain standard operational cadence; schedule preventative crane maintenance.

---

## 11. Known Assumptions, Limitations & Future Roadmap

1. **Synthetic Data Nature:** The dataset is generated via queue-theoretic simulation (M/M/c with lognormal shocks). While structurally realistic, live maritime logistics involves labor collective bargaining, pilotage bottlenecks, and shipping alliance vessel sharing.
2. **Designated 6.0h Threshold:** The threshold is a take-home assessment parameter, not an industry universal.
3. **SHAP is Associational:** High feature importance indicates correlation, not proven causal intervention guarantees.
4. **Commercial Deployment Requirements:**
   - Streaming terrestrial and satellite AIS vessel tracking feeds.
   - Terminal Operating System (TOS) telemetry (crane moves per hour, yard stack density).
   - Validated outcome ground-truth records from certified pilot boarding logs.

---

## 12. 5-Minute Evaluation Demo Walkthrough

A structured 5-minute walkthrough script with talking points, preset scenarios, and reproducible API commands is provided in [DEMO.md](DEMO.md).

---

## 13. License
This project is licensed under the MIT License - see the `LICENSE` file for details.

