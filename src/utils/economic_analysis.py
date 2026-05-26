"""
Economic analysis of adversarial attacks on GNN-based anomaly detectors.

Quantifies the cost of attacks from the attacker's perspective:
- Feature modification cost (creating fake profiles, buying accounts)
- Graph modification cost (creating fake connections, botnet coordination)
- Detection risk (probability of being caught)
- ROI analysis (benefit vs. cost)

This provides a practical lens on attack feasibility beyond pure ASR.
"""

import torch
import numpy as np
from typing import Dict


class AttackCostAnalyzer:
    """Quantify the economic cost of adversarial attacks.

    Models the attacker's cost in terms of:
    1. Feature perturbation cost: modifying profile attributes
    2. Edge perturbation cost: creating fake accounts/connections
    3. Detection risk: probability of being detected during attack
    4. Time cost: computational resources required
    """

    # Estimated costs in USD (based on black market pricing)
    COST_PER_FAKE_PROFILE = 0.50        # Cost to create a fake Twitter profile
    COST_PER_CONNECTION = 0.01          # Cost per fake follower/connection
    COST_PER_FEATURE_MOD = 0.10         # Cost to modify a profile feature
    VALUE_PER_EVADED_BOT = 5.00         # Value of one successfully evading bot account
    COST_DETECTED = 50.00               # Cost when detected (account ban, investigation)

    def __init__(
        self,
        cost_per_fake_profile: float = None,
        cost_per_connection: float = None,
        value_per_evaded_bot: float = None,
    ):
        if cost_per_fake_profile is not None:
            self.COST_PER_FAKE_PROFILE = cost_per_fake_profile
        if cost_per_connection is not None:
            self.COST_PER_CONNECTION = cost_per_connection
        if value_per_evaded_bot is not None:
            self.VALUE_PER_EVADED_BOT = value_per_evaded_bot

    def analyze_feature_attack(
        self,
        attack_info: Dict,
        n_targets: int,
        asr: float,
    ) -> Dict:
        """Analyze the cost of a feature-space attack.

        Args:
            attack_info: attack result info dict
            n_targets: number of target nodes
            asr: attack success rate

        Returns:
            cost_analysis: dict with cost breakdown
        """
        pert_norm = attack_info.get("perturbation_norm", 0)
        avg_pert_per_node = attack_info.get("avg_perturbation_per_node", 0)

        # Cost: each feature modification costs money
        # (hiring humans to create realistic fake profiles)
        total_feature_mods = n_targets * max(1, int(avg_pert_per_node * 5))
        feature_cost = total_feature_mods * self.COST_PER_FEATURE_MOD

        # Benefit: value of successfully evaded bots
        n_evaded = int(n_targets * asr)
        benefit = n_evaded * self.VALUE_PER_EVADED_BOT

        # Risk cost: expected cost of detection
        detection_prob = 1 - asr  # fraction that remains detected
        risk_cost = n_targets * detection_prob * self.COST_DETECTED

        total_cost = feature_cost + risk_cost
        roi = (benefit - total_cost) / max(total_cost, 1e-8)

        return {
            "attack_type": "feature",
            "n_targets": n_targets,
            "n_evaded": n_evaded,
            "asr": asr,
            "feature_cost": feature_cost,
            "risk_cost": risk_cost,
            "total_cost": total_cost,
            "benefit": benefit,
            "net_profit": benefit - total_cost,
            "roi": roi,
            "cost_per_evaded_bot": total_cost / max(n_evaded, 1),
            "cost_breakdown": {
                "per_feature_mod": self.COST_PER_FEATURE_MOD,
                "total_feature_mods": total_feature_mods,
                "perturbation_norm": pert_norm,
            },
        }

    def analyze_graph_attack(
        self,
        attack_info: Dict,
        n_targets: int,
        asr: float,
    ) -> Dict:
        """Analyze the cost of a graph-space attack."""
        edges_added = attack_info.get("edges_added", 0)
        edges_removed = attack_info.get("edges_removed", 0)

        # Cost: each fake connection costs money
        connection_cost = edges_added * self.COST_PER_CONNECTION

        # Additional cost: creating bot accounts to form connections
        unique_new_accounts = max(1, edges_added // 10)  # each new account provides ~10 edges
        account_cost = unique_new_accounts * self.COST_PER_FAKE_PROFILE

        n_evaded = int(n_targets * asr)
        benefit = n_evaded * self.VALUE_PER_EVADED_BOT

        detection_prob = 1 - asr
        risk_cost = n_targets * detection_prob * self.COST_DETECTED

        total_cost = connection_cost + account_cost + risk_cost
        roi = (benefit - total_cost) / max(total_cost, 1e-8)

        return {
            "attack_type": "graph",
            "n_targets": n_targets,
            "n_evaded": n_evaded,
            "asr": asr,
            "connection_cost": connection_cost,
            "account_cost": account_cost,
            "risk_cost": risk_cost,
            "total_cost": total_cost,
            "benefit": benefit,
            "net_profit": benefit - total_cost,
            "roi": roi,
            "cost_per_evaded_bot": total_cost / max(n_evaded, 1),
            "edges_added": edges_added,
        }

    def compare_attacks(
        self,
        feature_analysis: Dict,
        graph_analysis: Dict,
    ) -> Dict:
        """Compare feature vs graph attack economics."""
        return {
            "feature_attack": feature_analysis,
            "graph_attack": graph_analysis,
            "recommendation": "feature" if feature_analysis["roi"] > graph_analysis["roi"] else "graph",
            "roi_difference": feature_analysis["roi"] - graph_analysis["roi"],
            "cost_efficiency_ratio": (
                feature_analysis["cost_per_evaded_bot"] /
                max(graph_analysis["cost_per_evaded_bot"], 1e-8)
            ),
        }


def format_economic_report(analyses: Dict) -> str:
    """Format economic analysis as a readable report."""
    lines = ["=" * 60]
    lines.append("ECONOMIC ANALYSIS REPORT")
    lines.append("=" * 60)

    for attack_type, analysis in analyses.items():
        lines.append(f"\n--- {attack_type.upper()} ATTACK ---")
        lines.append(f"  Targets: {analysis['n_targets']}")
        lines.append(f"  Evaded: {analysis['n_evaded']} ({analysis['asr']:.1%})")
        lines.append(f"  Total cost: ${analysis['total_cost']:.2f}")
        lines.append(f"  Benefit: ${analysis['benefit']:.2f}")
        lines.append(f"  Net profit: ${analysis['net_profit']:.2f}")
        lines.append(f"  ROI: {analysis['roi']:.2%}")
        lines.append(f"  Cost per evaded bot: ${analysis['cost_per_evaded_bot']:.2f}")

    if "recommendation" in analyses:
        lines.append(f"\n--- COMPARISON ---")
        lines.append(f"  Recommended attack: {analyses['recommendation']}")
        lines.append(f"  ROI difference: {analyses['roi_difference']:.2%}")

    return "\n".join(lines)
