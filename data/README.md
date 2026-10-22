# HarborSense AI Data Dictionary & Synthetic Simulation Assumptions

## 1. Overview
The HarborSense AI dataset (`data/port_congestion.csv`) simulates maritime port operational states and congestion outcomes across 5 major global ports over 50 operating days at 4-hour chronological intervals (300 observations per port, 1,500 total observations).

---

## 2. Data Dictionary

| Column Name | Data Type | Units / Range | Role | Description |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | `string (ISO 8601)` | `YYYY-MM-DDTHH:MM:SS` | Metadata | Prediction reference timestamp $t$. Only telemetry at or before $t$ is used. |
| `port_id` | `categorical (string)` | 5 ports | Feature | Unique port facility identifier (`PORT_ROTTERDAM`, `PORT_SINGAPORE`, `PORT_SHANGHAI`, `PORT_LOS_ANGELES`, `PORT_ANTWERP`). |
| `expected_arrivals_24h` | `float` | `[0, 150]` vessels | Feature | Scheduled / forecasted incoming vessel arrivals across the forward 24h window. |
| `berth_capacity` | `float` | `[1, 70]` berths | Feature | Total operational container/cargo vessel berths available at the terminal. |
| `current_utilization` | `float` | `[0.0, 1.8]` ratio | Feature | Terminal yard and berth utilization ratio ($>1.0$ indicates yard saturation). |
| `current_queue_length` | `integer` | `[0, 120]` vessels | Feature | Number of vessels currently waiting at outer anchorage / roads for berth assignment. |
| `recent_arrivals_6h` | `integer` | `[0, 60]` vessels | Feature | Realized vessel arrivals logged during the prior 6 hours $(t-6\text{h} \to t)$. |
| `weather_severity` | `float` | `[0.0, 1.0]` index | Feature | Adverse meteorological index (0=calm, 0.4=moderate wind/swell, $\ge 0.7$=severe storm/gale). |
| `historical_delay_rate` | `float` | `[0.0, 1.0]` rate | Feature | Rolling fraction of congested periods realized strictly before $t$ (zero future leakage). |
| `hour` | `integer` | `[0, 23]` hour | Feature | Hour of day at prediction timestamp. |
| `day_of_week` | `integer` | `[0, 6]` day | Feature | Day of week at prediction timestamp ($0=\text{Monday}, 6=\text{Sunday}$). |
| `future_median_wait_hours` | `float` | `[0.5, 48.0]` hours | Target (Reg.) | Realized median vessel waiting time in hours across the forward 24-hour window. |
| `congestion` | `integer (binary)` | `{0, 1}` | Target (Cls.) | Ground truth congestion label: `1 if future_median_wait_hours > 6.0 else 0`. |

---

## 3. Simulated Port Profiles

| Port Identifier | Hub Type | Berth Capacity | Base Daily Arrivals | Weather Sensitivity |
| :--- | :--- | :---: | :---: | :---: |
| `PORT_ROTTERDAM` | European Mega Gateway | 26 | 32 | High (North Sea gale/swells) |
| `PORT_SINGAPORE` | Global Transshipment Hub | 36 | 46 | Moderate (Tropical squalls) |
| `PORT_SHANGHAI` | High-Volume Mega Terminal | 42 | 56 | Moderate-High (Monsoon/Typhoon) |
| `PORT_LOS_ANGELES` | Gateway Import Hub | 18 | 23 | Low-Moderate (Pacific fog/swells) |
| `PORT_ANTWERP` | European Chemical / Feeder | 16 | 18 | Moderate (River Scheldt tides) |

---

## 4. Synthetic Generation Assumptions & Principles

1. **Queueing Theoretic Foundation:**  
   Future median waiting hours are determined using M/M/c queuing approximations:
   $$\text{traffic\_pressure} = \frac{\text{current\_queue} \times 1.4 + \text{expected\_arrivals\_24h} \times 0.75}{\max(1.0, \text{effective\_capacity})}$$
   where effective capacity degrades under severe weather:
   $$\text{effective\_capacity} = \text{berth\_capacity} \times \max(0.2, 1.0 - 0.50 \times \text{weather\_sensitivity} \times \text{weather\_severity})$$

2. **Nonlinear Delay Escalation:**  
   When $\text{traffic\_pressure} > 1.0$, waiting delays spike superlinearly ($\text{traffic\_pressure}^{1.75}$), reflecting real-world terminal gridlock.

3. **Stochastic Operational Noise:**  
   Unobserved delays (customs holds, equipment breakdown, pilot dispatch lags) are modeled via multiplicative lognormal noise ($\sigma = 0.30$).

4. **Strict Leakage Prevention:**  
   - `historical_delay_rate` is computed using a rolling window of past realized congestion outcomes prior to timestamp $t$.  
   - Future waiting hours (`future_median_wait_hours`) and `congestion` labels are strictly quarantined from input feature matrices.

5. **Designated Target Definition:**  
   The 6.0-hour threshold for `congestion` is a project specification for the take-home assessment, not an arbitrary universal maritime constant.

