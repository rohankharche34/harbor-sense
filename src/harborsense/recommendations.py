"""Operational recommendations engine for HarborSense AI.

Translates:
- Predicted congestion probability and risk category
- Conformal uncertainty / reliability tier
- Key contributing feature drivers (SHAP local importance)
- Telemetry thresholds (queue length, utilization, weather)
into prioritized, actionable logistics decision-support recommendations.

Important:
Recommendations are decision-support guidelines based on statistical associations,
not proven causal interventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from harborsense.config import ProjectConfig, config


@dataclass
class OperationalAction:
    """A discrete operational action recommended to port dispatchers."""

    category: str  # e.g., "TRAFFIC", "BERTH", "WEATHER", "YARD", "GOVERNANCE"
    urgency: str  # "HIGH", "MEDIUM", "LOW"
    action: str
    rationale: str
    is_causal_guarantee: bool = False  # Always False per guidelines


@dataclass
class RecommendationPlan:
    """Comprehensive recommendation payload for a prediction."""

    risk_category: str
    predicted_probability: float
    primary_actions: List[OperationalAction] = field(default_factory=list)
    contingency_actions: List[OperationalAction] = field(default_factory=list)
    human_in_the_loop_flag: bool = False
    disclaimer: str = (
        "Decision-support recommendation: based on empirical correlation and queue models. "
        "Does not guarantee delay elimination. Human dispatchers retain ultimate authority."
    )


def generate_recommendations(
    predicted_class: int,
    calibrated_prob: float,
    reliability_tier: str,
    raw_features: Dict[str, Any],
    top_driver_features: Optional[List[str]] = None,
    cfg: ProjectConfig = config,
) -> RecommendationPlan:
    """Generate conditional, driver-aware operational recommendations.

    Args:
        predicted_class: 1 (Congestion) or 0 (Normal).
        calibrated_prob: Calibrated likelihood of congestion.
        reliability_tier: "HIGH", "MEDIUM", or "LOW".
        raw_features: Dictionary of raw input features.
        top_driver_features: List of feature names with strongest positive risk push.
        cfg: ProjectConfig instance.

    Returns:
        Structured RecommendationPlan.
    """
    top_drivers = set(top_driver_features or [])

    # Categorize Risk
    if calibrated_prob < cfg.low_risk_threshold:
        risk_category = "LOW"
    elif calibrated_prob < cfg.high_risk_threshold:
        risk_category = "MEDIUM"
    else:
        risk_category = "HIGH"

    primary_actions: List[OperationalAction] = []
    contingency_actions: List[OperationalAction] = []
    human_review_needed = False

    # 1. Uncertainty / Data Quality Escalation
    if reliability_tier == "LOW":
        human_review_needed = True
        primary_actions.append(
            OperationalAction(
                category="GOVERNANCE",
                urgency="HIGH",
                action="Escalate to Senior Port Dispatcher & Verify Telemetry",
                rationale=(
                    "Prediction exhibits high statistical uncertainty (conformal set includes both classes "
                    "or anomalous telemetry). Manual verification of AIS transponder and berth logs is required."
                ),
            )
        )

    # 2. High Risk Protocols
    if risk_category == "HIGH":
        # Traffic / Arrival surge
        if (
            "expected_arrivals_24h" in top_drivers
            or raw_features.get("expected_arrivals_24h", 0)
            > (raw_features.get("berth_capacity", 1) * 1.8)
        ):
            primary_actions.append(
                OperationalAction(
                    category="TRAFFIC",
                    urgency="HIGH",
                    action="Implement Virtual Queuing & Vessel Slow-Steaming Advisories",
                    rationale=(
                        f"Incoming 24h arrival volume ({raw_features.get('expected_arrivals_24h', 0):.0f} vessels) "
                        f"severely stresses berth capacity ({raw_features.get('berth_capacity', 1):.0f} berths). "
                        "Advise inbound carriers to slow-steam to smooth arrival peak."
                    ),
                )
            )

        # Backlog / Existing Queue
        if (
            "current_queue_length" in top_drivers
            or raw_features.get("current_queue_length", 0) > 8
        ):
            primary_actions.append(
                OperationalAction(
                    category="BERTH",
                    urgency="HIGH",
                    action="Prioritize Bottleneck Berths & Deploy Auxiliary Tug Support",
                    rationale=(
                        f"Existing queue of {raw_features.get('current_queue_length', 0):.0f} vessels "
                        "will compound future incoming delays. Expedite pilotage assignments and vessel departures."
                    ),
                )
            )

        # Adverse Weather
        if "weather_severity" in top_drivers or raw_features.get("weather_severity", 0) > 0.60:
            primary_actions.append(
                OperationalAction(
                    category="WEATHER",
                    urgency="HIGH",
                    action="Enact Heavy Weather Protocol & Re-sequence Crane Gangs",
                    rationale=(
                        f"Weather severity is high ({raw_features.get('weather_severity', 0):.2f}). "
                        "Anticipate wind/swell crane safety cutoffs; reallocate labor to yard operations during storm windows."
                    ),
                )
            )

        # High Utilization
        if raw_features.get("current_utilization", 0) > 0.85:
            contingency_actions.append(
                OperationalAction(
                    category="YARD",
                    urgency="MEDIUM",
                    action="Open Overflow Container Staging Areas & Extend Gate Hours",
                    rationale=(
                        f"Current terminal utilization is at {raw_features.get('current_utilization', 0):.1%}. "
                        "Clear quayside bottlenecks by accelerating gate turnarounds."
                    ),
                )
            )

    # 3. Medium Risk Protocols
    elif risk_category == "MEDIUM":
        primary_actions.append(
            OperationalAction(
                category="MONITORING",
                urgency="MEDIUM",
                action="Increase Telemetry Polling to 2-Hour Intervals & Notify Terminals",
                rationale=(
                    f"Congestion likelihood is moderate ({calibrated_prob:.1%}). "
                    "Small fluctuations in arrival times or crane productivity could trigger significant delays."
                ),
            )
        )

        if raw_features.get("weather_severity", 0) > 0.50:
            contingency_actions.append(
                OperationalAction(
                    category="WEATHER",
                    urgency="MEDIUM",
                    action="Monitor Meteorological Feeds for Impending Sea State Deterioration",
                    rationale="Weather is at a threshold where further deterioration will precipitate delays.",
                )
            )

        if raw_features.get("current_queue_length", 0) > 4:
            contingency_actions.append(
                OperationalAction(
                    category="BERTH",
                    urgency="MEDIUM",
                    action="Review Berth Allocation Windows for Next 12 Hours",
                    rationale="Queue backlog requires proactive berth window adjustment.",
                )
            )

    # 4. Low Risk Protocols
    else:
        primary_actions.append(
            OperationalAction(
                category="MONITORING",
                urgency="LOW",
                action="Maintain Standard Operational Cadence",
                rationale=(
                    f"Estimated congestion probability is low ({calibrated_prob:.1%}). "
                    "Current capacity and traffic parameters indicate normal operations. "
                    "Continue routine monitoring; note that unexpected mechanical or marine incidents can still occur."
                ),
            )
        )
        contingency_actions.append(
            OperationalAction(
                category="MAINTENANCE",
                urgency="LOW",
                action="Utilize Operational Slack for Scheduled Crane & Berth Maintenance",
                rationale="Favorable operating conditions afford an opportunistic window for preventative maintenance.",
            )
        )

    return RecommendationPlan(
        risk_category=risk_category,
        predicted_probability=calibrated_prob,
        primary_actions=primary_actions,
        contingency_actions=contingency_actions,
        human_in_the_loop_flag=human_review_needed,
    )

