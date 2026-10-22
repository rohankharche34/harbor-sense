# HarborSense AI: 5-Minute Demonstration Walkthrough

This script guides an evaluator or stakeholder through an interactive 5-minute demonstration of HarborSense AI.

---

## ⏱️ Timeline Summary

| Segment | Duration | Objective & Key Talking Points |
| :--- | :--- | :--- |
| **1. Problem & Motivation** | 30s | The challenge of stochastic port queues, vessel waiting costs, and supply chain ripple effects. |
| **2. Problem Formulation & Data** | 45s | 24-hour horizon, `median_wait > 6.0h` definition, 5 simulated global hubs, zero target leakage. |
| **3. Model Performance & Diagnostics** | 60s | Held-out test metrics, baseline comparisons, sigmoid calibration, and conformal uncertainty. |
| **4. Live Interactive Prediction** | 90s | Running a live scenario in the Streamlit dashboard, viewing SHAP drivers and operational actions. |
| **5. API Architecture & Testing** | 45s | FastAPI OpenAPI schema, input validation, test coverage (data, pipeline, explainability, API). |
| **6. Conclusion & Roadmap** | 30s | Engineering takeaways, absence of real AIS/TOS validation, path to commercial deployment. |

---

## 1. Problem & Motivation (0:00 - 0:30)
> *"Global supply chains lose billions annually to unpredictable port congestion. When container ships arrive at saturated terminals or during severe weather windows, vessel queues spike nonlinearly. Dispatchers need more than a generic alert—they need a reliable 24-hour forward look, honest confidence intervals, explainable feature attributions, and actionable operational playbooks. HarborSense AI delivers this decision-support capability."*

---

## 2. Dataset & Congestion Definition (0:30 - 1:15)
> *"HarborSense AI formulates the problem as 24-hour forward congestion forecasting:  
> **congestion = 1 if future_median_wait_hours > 6.0 else 0**  
> We simulated 1,500 chronological observations across five major global ports (Rotterdam, Singapore, Shanghai, Los Angeles, Antwerp) over 50 operating days. Delays are modeled using queue-theoretic principles where weather derates crane handling capacity and queues escalate superlinearly. Crucially, historical delay rates use strictly past-only realized data, and train/val/test splits enforce a **24-hour temporal purge gap** to prevent any future target leakage across boundaries."*

---

## 3. Model Evaluation & Diagnostics (1:15 - 2:15)
> *"Navigating to the **Model Evaluation** tab on the dashboard:  
> - We trained a baseline Logistic Regression model alongside our primary nonlinear Random Forest classifier.  
> - On the held-out test partition, our calibrated Random Forest achieves **>85% accuracy**, an **F1-score > 0.80**, and **ROC-AUC > 0.90**, significantly exceeding majority-class and heuristic baselines.  
> - Probabilities are calibrated via **Sigmoid scaling**, drastically minimizing Brier score loss.  
> - We also trained a secondary Random Forest regressor to continuously estimate waiting hours (evaluated via MAE and RMSE) without mixing classification and regression metrics."*

---

## 4. Interactive Live Prediction & Playbook (2:15 - 3:45)
> *"Let's test a realistic operational scenario in the **Live Interactive Prediction** tab:*
> 1. Select preset: **Severe Storm & Arrival Surge (Rotterdam)**.
>    - Expected Arrivals (24h): `52 vessels` vs Berth Capacity: `26 berths`
>    - Current Queue: `11 vessels` | Utilization: `92%` | Weather Severity: `0.85` (Gale)
> 2. Click **Run Prediction & Diagnosis**:
>    - **Result:** Congestion Probability: `>90%` (Risk Category: **HIGH**).
>    - **Secondary Regressor:** Estimated Wait: `~12-16 hours`.
>    - **Conformal Uncertainty:** Prediction Set = `{1}`, Reliability Tier = **HIGH** (90% confidence singleton).
>    - **Local SHAP Attribution:** Identifies high arrival-to-berth ratio (`+0.25`) and weather disruption (`+0.18`) as dominant positive risk drivers.
>    - **Prescriptive Actions:** The system recommends enacting virtual queuing / slow-steaming advisories, prioritizing bottleneck berths with auxiliary tugs, and enacting heavy weather crane tie-down protocols."*

---

## 5. API Architecture & Automated Test Suite (3:45 - 4:30)
> *"HarborSense AI provides a production-ready FastAPI backend sharing the exact same model artifacts and inference pipeline:  
> - Endpoints include `GET /health`, `GET /metadata`, `GET /metrics`, `POST /predict`, and `POST /predict/batch`.  
> - Request payloads are strictly validated via Pydantic; any attempt to pass future target columns triggers target leakage rejection.  
> - The pytest suite verifies data reproducibility, absence of leakage, schema conformity, model calibration, conformal prediction sets, and API endpoints."*

Launch API:
```bash
make api
# Interactive OpenAPI documentation available at: http://localhost:8000/docs
```

Run Test Suite:
```bash
make test
```

---

## 6. Conclusion & Roadmap (4:30 - 5:00)
> *"In summary, HarborSense AI provides a complete, explainable, and uncertainty-aware port congestion prediction platform. While our synthetic dataset demonstrates realistic queue mechanics, true commercial operational deployment will require streaming AIS vessel tracking, Terminal Operating System (TOS) crane telemetry, and certified pilot boarding ground-truth logs."*

---

## 📌 Reproducible Sample Prediction Payload

```json
{
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
}
```

Curl Execution:
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

