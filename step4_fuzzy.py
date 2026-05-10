"""
AIDRA - Step 4: Fuzzy Logic — Uncertainty Handling
====================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# 1. FUZZY MEMBERSHIP FUNCTIONS
# ─────────────────────────────────────────

def trapezoid(x, a, b, c, d):
    """
    Trapezoidal Membership Function
    Handles vertical edges correctly.
    """

    # Outside left
    if x < a:
        return 0.0

    # Rising edge
    elif a <= x < b:
        return (x - a) / (b - a) if b != a else 1.0

    # Top plateau
    elif b <= x <= c:
        return 1.0

    # Falling edge
    elif c < x <= d:
        return (d - x) / (d - c) if d != c else 1.0

    # Outside right
    else:
        return 0.0


def triangle(x, a, b, c):
    """Triangular Membership Function"""

    if x <= a or x >= c:
        return 0.0

    elif x == b:
        return 1.0

    elif a < x < b:
        return (x - a) / (b - a)

    else:
        return (c - x) / (c - b)


# ─────────────────────────────────────────
# 2. FUZZIFICATION
# ─────────────────────────────────────────

def fuzzify_hazard(h):
    """hazard_level: 0–10"""

    return {
        "low":    trapezoid(h, 0, 0, 2, 4),
        "medium": triangle(h, 2, 5, 8),
        "high":   trapezoid(h, 6, 8, 10, 10),
    }


def fuzzify_time(t):
    """rescue_time: 0–30"""

    return {
        "fast":   trapezoid(t, 0, 0, 5, 12),
        "medium": triangle(t, 8, 15, 22),
        "slow":   trapezoid(t, 18, 24, 30, 30),
    }


def fuzzify_severity(s):
    """severity: 0–10"""

    return {
        "minor":    trapezoid(s, 0, 0, 2, 4),
        "moderate": triangle(s, 3, 5, 7),
        "critical": trapezoid(s, 6, 8, 10, 10),
    }


# ─────────────────────────────────────────
# 3. FUZZY RULE BASE
# ─────────────────────────────────────────

def apply_rules(fh, ft, fs):

    urgency_rules = []
    risk_rules = []

    # ── URGENCY RULES ──

    r1 = min(fs["critical"], ft["fast"])
    urgency_rules.append((r1, 9.0))

    r2 = min(fs["critical"], ft["slow"])
    urgency_rules.append((r2, 7.0))

    r3 = min(fs["moderate"], ft["fast"])
    urgency_rules.append((r3, 5.5))

    r4 = min(fs["moderate"], ft["slow"])
    urgency_rules.append((r4, 4.0))

    r5 = fs["minor"]
    urgency_rules.append((r5, 2.0))

    r6 = min(fs["critical"], fh["high"])
    urgency_rules.append((r6, 9.5))

    # ── RISK RULES ──

    r7 = min(fh["high"], ft["slow"])
    risk_rules.append((r7, 9.0))

    r8 = min(fh["high"], ft["fast"])
    risk_rules.append((r8, 7.0))

    r9 = min(fh["medium"], ft["medium"])
    risk_rules.append((r9, 5.0))

    r10 = fh["low"]
    risk_rules.append((r10, 2.0))

    r11 = min(fh["medium"], fs["critical"])
    risk_rules.append((r11, 7.5))

    return urgency_rules, risk_rules


# ─────────────────────────────────────────
# 4. DEFUZZIFICATION
# ─────────────────────────────────────────

def defuzzify(rules):

    numerator = sum(s * c for s, c in rules if s > 0)
    denominator = sum(s for s, _ in rules if s > 0)

    return numerator / denominator if denominator > 0 else 0.0


# ─────────────────────────────────────────
# 5. FINAL DECISION
# ─────────────────────────────────────────

def make_decision(urgency, route_risk):

    if urgency >= 7.5:

        if route_risk >= 7.0:
            return (
                "IMMEDIATE_FAST_ROUTE",
                "Victim critically urgent — accept high risk"
            )

        else:
            return (
                "IMMEDIATE_SAFE_ROUTE",
                "Victim critically urgent — safe route available"
            )

    elif urgency >= 4.5:

        if route_risk >= 6.0:
            return (
                "SAFE_ROUTE_DELAY",
                "Moderate urgency but high risk"
            )

        else:
            return (
                "FAST_ROUTE",
                "Moderate urgency and acceptable risk"
            )

    else:

        return (
            "DELAY_LOW_PRIORITY",
            "Low urgency — conserve resources"
        )


# ─────────────────────────────────────────
# 6. FULL FUZZY ASSESSMENT
# ─────────────────────────────────────────

def fuzzy_assess(hazard_level, rescue_time, severity_score):

    fh = fuzzify_hazard(hazard_level)
    ft = fuzzify_time(rescue_time)
    fs = fuzzify_severity(severity_score)

    urgency_rules, risk_rules = apply_rules(fh, ft, fs)

    urgency = defuzzify(urgency_rules)
    route_risk = defuzzify(risk_rules)

    decision, justification = make_decision(
        urgency,
        route_risk
    )

    return {
        "urgency": urgency,
        "route_risk": route_risk,
        "decision": decision,
        "justification": justification,
        "fh": fh,
        "ft": ft,
        "fs": fs
    }


# ─────────────────────────────────────────
# 7. ASSESS ALL VICTIMS
# ─────────────────────────────────────────

SEVERITY_FUZZY = {
    "critical": 9.0,
    "moderate": 5.0,
    "minor": 1.5
}


def assess_all_victims(victims, env):

    from step1_environment import astar

    print(
        f"\n{'Victim':<8}"
        f"{'Severity':<12}"
        f"{'Hazard':>8}"
        f"{'Time':>8}"
        f"{'Urgency':>10}"
        f"{'Risk':>8}  Decision"
    )

    print("-" * 80)

    assessments = []

    for v in victims:

        route = astar(
            env,
            env.base_pos,
            v["pos"],
            avoid_hazard=False
        )

        dist = route["cost"] if route else 15

        hazard_count = sum(
            1 for pos in (route["path"] if route else [])
            if pos in env.hazard_cells
        )

        hazard_level = min(hazard_count * 2.5, 10)

        rescue_time = dist * 2 + hazard_count

        severity_score = SEVERITY_FUZZY[v["severity"]]

        result = fuzzy_assess(
            hazard_level,
            rescue_time,
            severity_score
        )

        result["victim"] = v

        print(
            f"{v['name']:<8}"
            f"{v['severity']:<12}"
            f"{hazard_level:>8.1f}"
            f"{rescue_time:>8}"
            f"{result['urgency']:>10.2f}"
            f"{result['route_risk']:>8.2f}  "
            f"{result['decision']}"
        )

        assessments.append(result)

    return assessments


# ─────────────────────────────────────────
# 8. VISUALISATION
# ─────────────────────────────────────────

def visualise_fuzzy(assessments):

    fig = plt.figure(figsize=(16, 10))

    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        hspace=0.45,
        wspace=0.38
    )

    # Hazard Membership Functions
    ax0 = fig.add_subplot(gs[0, 0])

    x = np.linspace(0, 10, 200)

    ax0.plot(
        x,
        [trapezoid(v, 0, 0, 2, 4) for v in x],
        label="Low",
        lw=2
    )

    ax0.plot(
        x,
        [triangle(v, 2, 5, 8) for v in x],
        label="Medium",
        lw=2
    )

    ax0.plot(
        x,
        [trapezoid(v, 6, 8, 10, 10) for v in x],
        label="High",
        lw=2
    )

    ax0.set_title("Hazard Membership")
    ax0.legend()

    # Victim Scores
    ax1 = fig.add_subplot(gs[1, :2])

    names = [a["victim"]["name"] for a in assessments]

    urgency = [a["urgency"] for a in assessments]

    risk = [a["route_risk"] for a in assessments]

    pos = np.arange(len(names))

    width = 0.35

    ax1.bar(pos - width/2, urgency, width, label="Urgency")

    ax1.bar(pos + width/2, risk, width, label="Risk")

    ax1.set_xticks(pos)

    ax1.set_xticklabels(names)

    ax1.set_ylim(0, 10)

    ax1.set_title("Urgency vs Route Risk")

    ax1.legend()

    # Decision Table
    ax2 = fig.add_subplot(gs[1, 2])

    ax2.axis('off')

    table_data = [
        [
            a["victim"]["name"],
            a["decision"],
            f"{a['urgency']:.1f}",
            f"{a['route_risk']:.1f}"
        ]
        for a in assessments
    ]

    table = ax2.table(
        cellText=table_data,
        colLabels=["Victim", "Decision", "Urgency", "Risk"],
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)

    table.set_fontsize(8)

    table.scale(1, 2)

    fig.suptitle(
        "AIDRA — Step 4: Fuzzy Logic",
        fontsize=14,
        fontweight='bold'
    )

    return fig


# ─────────────────────────────────────────
# 9. RUN DEMO
# ─────────────────────────────────────────

if __name__ == "__main__":

    from step1_environment import DisasterEnvironment

    print("=" * 60)
    print("AIDRA — Step 4: Fuzzy Logic")
    print("=" * 60)

    env = DisasterEnvironment()

    assessments = assess_all_victims(
        env.victims,
        env
    )

    print("\nDETAILED DECISIONS")
    print("=" * 60)

    for a in assessments:

        v = a["victim"]

        print(f"\n{v['name']} ({v['severity']})")

        print(f"Urgency    : {a['urgency']:.2f}")

        print(f"Route Risk : {a['route_risk']:.2f}")

        print(f"Decision   : {a['decision']}")

        print(f"Why        : {a['justification']}")

    # Create figure
    fig = visualise_fuzzy(assessments)

    # Save safely on Windows/Linux/Mac
    output_path = os.path.join(
        os.getcwd(),
        "aidra_fuzzy.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches='tight'
    )

    print(f"\nFuzzy chart saved to:")
    print(output_path)

    plt.show()

    print("\nStep 4 complete! Next: Full Integration")
