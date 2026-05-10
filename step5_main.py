"""
AIDRA - Step 5: Full Integration
=================================
Complete end-to-end simulation:
  1. Environment loads (grid, victims, hazards)
  2. Fuzzy Logic assesses each victim → urgency + risk
  3. ML predicts survival probability per victim
  4. CSP allocates ambulances/team respecting hard constraints
  5. Search (A*) plans routes — fast vs safe trade-off
  6. Dynamic events fire mid-mission → replanning triggered
  7. KPIs computed and logged
  8. Full decision log printed + charts saved
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# STEP 1: Environment + Search Algorithms
# ─────────────────────────────────────────

from step1_environment import (
    DisasterEnvironment,
    astar,
    bfs,
    dfs,
    greedy_best_first,
    visualise_grid
)

# ─────────────────────────────────────────
# STEP 2: CSP Solver
# ─────────────────────────────────────────

from step2_csp import (
    CSPSolver,
    prioritize_victims,
    build_allocation_plan
)

# ─────────────────────────────────────────
# STEP 3: Machine Learning
# ─────────────────────────────────────────

from step3_ml import (
    generate_dataset,
    train_and_evaluate,
    predict_victim_survival,
    SEVERITY_NUM
)

# ─────────────────────────────────────────
# STEP 4: Fuzzy Logic
# ─────────────────────────────────────────

from step4_fuzzy import (
    assess_all_victims,
    SEVERITY_FUZZY
)

# ─────────────────────────────────────────
# 1. DECISION LOG
# ─────────────────────────────────────────

class DecisionLog:

    def __init__(self):
        self.entries = []

    def log(self, step, event, decision, justification):

        entry = {
            "step": step,
            "event": event,
            "decision": decision,
            "justification": justification
        }

        self.entries.append(entry)

        print(f"\n[LOG #{len(self.entries)}] {event}")
        print(f"Decision      : {decision}")
        print(f"Justification : {justification}")

    def print_full(self):

        print("\n" + "=" * 70)
        print("FULL DECISION LOG")
        print("=" * 70)

        for e in self.entries:

            print(f"\nStep {e['step']} | {e['event']}")
            print(f"Decision : {e['decision']}")
            print(f"Why      : {e['justification']}")

# ─────────────────────────────────────────
# 2. KPI TRACKER
# ─────────────────────────────────────────

class KPITracker:

    def __init__(self):

        self.victims_saved = 0
        self.total_victims = 5

        self.rescue_times = []

        self.risk_exposure_score = 0
        self.replan_events = 0

        self.path_costs = {}
        self.nodes_expanded = {}

        self.resource_usage = {
            "ambulance_1": 0,
            "ambulance_2": 0,
            "rescue_team": 0
        }

    def record_rescue(self, victim, route, resource, replanned=False):

        self.victims_saved += 1

        self.rescue_times.append(route["cost"] * 2)

        self.resource_usage[resource] += 1

        hazard_cells_on_route = sum(
            1 for pos in route["path"]
            if pos in {
                (3, 3),
                (3, 4),
                (4, 3),
                (4, 4),
                (4, 5),
                (5, 4)
            }
        )

        self.risk_exposure_score += hazard_cells_on_route

        if replanned:
            self.replan_events += 1

    def record_search_comparison(self, algo, cost, nodes):

        self.path_costs[algo] = cost
        self.nodes_expanded[algo] = nodes

    def avg_rescue_time(self):

        if len(self.rescue_times) == 0:
            return 0

        return np.mean(self.rescue_times)

    def path_optimality_ratio(self):

        if not self.path_costs:
            return {}

        best = min(self.path_costs.values())

        return {
            a: round(c / best, 3)
            for a, c in self.path_costs.items()
        }

    def resource_utilisation(self):

        total = sum(self.resource_usage.values())

        if total == 0:
            return {}

        return {
            r: round(v / total, 3)
            for r, v in self.resource_usage.items()
        }

    def print_summary(self):

        print("\n" + "=" * 60)
        print("KEY PERFORMANCE INDICATORS")
        print("=" * 60)

        print(f"Victims Saved           : {self.victims_saved} / {self.total_victims}")

        print(f"Average Rescue Time     : {self.avg_rescue_time():.1f} min")

        print(f"Risk Exposure           : {self.risk_exposure_score}")

        print(f"Replan Events           : {self.replan_events}")

        print("\nPath Optimality Ratios:")

        for algo, ratio in self.path_optimality_ratio().items():

            print(f"{algo:<10} {ratio}")

        print("\nNodes Expanded:")

        for algo, n in self.nodes_expanded.items():

            print(f"{algo:<10} {n}")

# ─────────────────────────────────────────
# 3. HILL CLIMBING
# ─────────────────────────────────────────

def total_route_cost(order, env):

    cost = 0

    start = env.base_pos

    for v in order:

        r = astar(
            env,
            start,
            v["pos"],
            avoid_hazard=False
        )

        if r:

            cost += r["cost"]

            start = v["pos"]

    return cost

def hill_climbing(victims, env, max_iter=200):

    current_order = victims[:]

    current_cost = total_route_cost(current_order, env)

    cost_history = [current_cost]

    for _ in range(max_iter):

        improved = False

        for i in range(len(current_order)):

            for j in range(i + 1, len(current_order)):

                neighbour = current_order[:]

                neighbour[i], neighbour[j] = \
                    neighbour[j], neighbour[i]

                neighbour_cost = total_route_cost(
                    neighbour,
                    env
                )

                if neighbour_cost < current_cost:

                    current_order = neighbour

                    current_cost = neighbour_cost

                    improved = True

                    break

            if improved:
                break

        cost_history.append(current_cost)

        if not improved:
            break

    return current_order, current_cost, cost_history

# ─────────────────────────────────────────
# 4. SEARCH COMPARISON
# ─────────────────────────────────────────

def compare_search_algorithms(env, start, goal, kpi):

    algos = {

        "BFS": bfs(
            env,
            start,
            goal,
            avoid_hazard=False
        ),

        "DFS": dfs(
            env,
            start,
            goal,
            avoid_hazard=False
        ),

        "Greedy": greedy_best_first(
            env,
            start,
            goal,
            avoid_hazard=False
        ),

        "A*": astar(
            env,
            start,
            goal,
            avoid_hazard=False
        )
    }

    print(f"\n{'Algorithm':<10} {'Cost':>6} {'Nodes':>8}")

    print("-" * 35)

    for name, r in algos.items():

        if r:

            kpi.record_search_comparison(
                name,
                r["cost"],
                r["nodes_expanded"]
            )

            print(
                f"{name:<10} "
                f"{r['cost']:>6} "
                f"{r['nodes_expanded']:>8}"
            )

    return algos

# ─────────────────────────────────────────
# 5. MAIN SIMULATION
# ─────────────────────────────────────────

def run_simulation():

    log = DecisionLog()

    kpi = KPITracker()

    env = DisasterEnvironment()

    print("\n" + "█" * 70)

    print("AIDRA — Full Simulation Run")

    print("█" * 70)

    # ─────────────────────────────────────
    # PHASE 1: ML Training
    # ─────────────────────────────────────

    print("\nPHASE 1: ML Training")

    X, y = generate_dataset(n=300)

    ml_results = train_and_evaluate(X, y)

    best_model = max(
        ml_results,
        key=lambda k: ml_results[k]["f1"]
    )

    print(
        f"Best ML model: {best_model} "
        f"(F1={ml_results[best_model]['f1']:.3f})"
    )

    # ─────────────────────────────────────
    # PHASE 2: Fuzzy Assessment
    # ─────────────────────────────────────

    print("\nPHASE 2: Fuzzy Risk Assessment")

    fuzzy_results = assess_all_victims(
        env.victims,
        env
    )

    # ─────────────────────────────────────
    # PHASE 3: ML Survival Prediction
    # ─────────────────────────────────────

    print("\nPHASE 3: ML Survival Prediction")

    ml_predictions, _ = predict_victim_survival(
        env.victims,
        env,
        ml_results
    )

    # ─────────────────────────────────────
    # PHASE 4: CSP Allocation
    # ─────────────────────────────────────

    print("\nPHASE 4: CSP Allocation")

    victims_ordered = prioritize_victims(
        env.victims,
        env,
        env.hospitals[0]
    )

    solver = CSPSolver(victims_ordered)

    solution = solver.solve()

    print(f"CSP solved in {solver.backtrack_count} backtracks")

    for vid, res in solution.items():

        vname = next(
            v["name"]
            for v in env.victims
            if v["id"] == vid
        )

        print(f"{vname} → {res}")

    log.log(
        4,
        "CSP Allocation",
        str(solution),
        "MRV + Forward Checking used"
    )

    # ─────────────────────────────────────
    # PHASE 5: Hill Climbing
    # ─────────────────────────────────────

    print("\nPHASE 5: Hill Climbing")

    original_cost = total_route_cost(
        victims_ordered,
        env
    )

    hc_order, hc_cost, hc_history = hill_climbing(
        victims_ordered,
        env
    )

    print(f"Original Cost  : {original_cost}")

    print(f"Optimized Cost : {hc_cost}")

    print(f"Improvement    : {original_cost - hc_cost}")

    # ─────────────────────────────────────
    # PHASE 6: Search Comparison
    # ─────────────────────────────────────

    print("\nPHASE 6: Search Comparison")

    start = env.base_pos

    goal = env.victims[0]["pos"]

    algos = compare_search_algorithms(
        env,
        start,
        goal,
        kpi
    )

    # ─────────────────────────────────────
    # PHASE 7: Rescue Missions
    # ─────────────────────────────────────

    print("\nPHASE 7: Rescue Missions")

    all_paths = []

    trip_num = 1

    for v in hc_order:

        vid = v["id"]

        resource = solution[vid]

        fz = next(
            a for a in fuzzy_results
            if a["victim"]["id"] == vid
        )

        use_safe = fz["route_risk"] > 6.0

        hospital = min(
            env.hospitals,
            key=lambda h:
            abs(h[0] - v["pos"][0]) +
            abs(h[1] - v["pos"][1])
        )

        route_to_victim = astar(
            env,
            env.base_pos,
            v["pos"],
            avoid_hazard=use_safe
        )

        if not route_to_victim:

            print(f"No route to {v['name']}")

            continue

        print(f"\nTrip {trip_num}")

        print(f"Victim      : {v['name']}")

        print(f"Resource    : {resource}")

        print(f"Route Type  : {'SAFE' if use_safe else 'FAST'}")

        print(f"Cost        : {route_to_victim['cost']}")

        kpi.record_rescue(
            v,
            route_to_victim,
            resource
        )

        all_paths.append(
            (
                route_to_victim,
                f"T{trip_num}:{v['name']}"
            )
        )

        env.mark_rescued(vid)

        trip_num += 1

        # Dynamic Event

        if trip_num == 3:

            print("\nDYNAMIC EVENT: Road blockage!")

            env.trigger_dynamic_event({
                (2, 6),
                (3, 6)
            })

            kpi.replan_events += 1

    # ─────────────────────────────────────
    # PHASE 8: KPI Summary
    # ─────────────────────────────────────

    print("\nPHASE 8: KPI Summary")

    kpi.print_summary()

    # Decision Log

    log.print_full()

    return (
        env,
        kpi,
        ml_results,
        fuzzy_results,
        all_paths,
        hc_history,
        algos
    )

# ─────────────────────────────────────────
# 6. VISUALIZATION DASHBOARD
# ─────────────────────────────────────────

def visualise_dashboard(
    env,
    kpi,
    ml_results,
    fuzzy_results,
    all_paths,
    hc_history,
    algos
):

    fig = plt.figure(figsize=(18, 12))

    gs = gridspec.GridSpec(
        3,
        4,
        figure=fig,
        hspace=0.5,
        wspace=0.4
    )

    # ─────────────────────────────────────
    # GRID MAP
    # ─────────────────────────────────────

    ax_grid = fig.add_subplot(gs[0, :2])

    color_map = {
        0: "#f5f5f5",
        1: "#2d2d2d",
        2: "#ff6b35",
        3: "#e63946",
        4: "#2a9d8f",
        5: "#457b9d"
    }

    for r in range(env.rows):

        for c in range(env.cols):

            rect = plt.Rectangle(
                [c, env.rows - 1 - r],
                1,
                1,
                facecolor=color_map.get(
                    env.grid[r][c],
                    "#ffffff"
                ),
                edgecolor="#cccccc",
                linewidth=0.4
            )

            ax_grid.add_patch(rect)

    path_colors = [
        "#06d6a0",
        "#ffd166",
        "#118ab2",
        "#ef476f",
        "#8338ec"
    ]

    for i, (result, label) in enumerate(all_paths[:5]):

        xs = [c + 0.5 for r, c in result["path"]]

        ys = [
            env.rows - 1 - r + 0.5
            for r, c in result["path"]
        ]

        ax_grid.plot(
            xs,
            ys,
            color=path_colors[i % 5],
            linewidth=2,
            label=label
        )

    ax_grid.set_xlim(0, env.cols)

    ax_grid.set_ylim(0, env.rows)

    ax_grid.set_title(
        "Grid Map — Rescue Routes",
        fontweight='bold'
    )

    # ─────────────────────────────────────
    # KPI BAR CHART
    # ─────────────────────────────────────

    ax_kpi = fig.add_subplot(gs[0, 2])

    ratios = kpi.path_optimality_ratio()

    names = list(ratios.keys())

    values = list(ratios.values())

    ax_kpi.barh(names, values)

    ax_kpi.set_title(
        "Path Optimality Ratio",
        fontweight='bold'
    )

    # ─────────────────────────────────────
    # NODE EXPANSION
    # ─────────────────────────────────────

    ax_nodes = fig.add_subplot(gs[0, 3])

    algo_names = list(kpi.nodes_expanded.keys())

    algo_vals = list(kpi.nodes_expanded.values())

    ax_nodes.bar(algo_names, algo_vals)

    ax_nodes.set_title(
        "Nodes Expanded",
        fontweight='bold'
    )

    # ─────────────────────────────────────
    # ML RESULTS
    # ─────────────────────────────────────

    ax_ml = fig.add_subplot(gs[1, :2])

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]

    model_names = list(ml_results.keys())

    x = np.arange(len(metric_names))

    width = 0.35

    for i, name in enumerate(model_names):

        vals = [
            ml_results[name][m]
            for m in metric_names
        ]

        ax_ml.bar(
            x + i * width,
            vals,
            width,
            label=name
        )

    ax_ml.set_xticks(x + width / 2)

    ax_ml.set_xticklabels(metric_names)

    ax_ml.legend()

    ax_ml.set_ylim(0, 1.2)

    ax_ml.set_title(
        "ML Model Comparison",
        fontweight='bold'
    )

    # ─────────────────────────────────────
    # FUZZY RESULTS
    # ─────────────────────────────────────

    ax_fz = fig.add_subplot(gs[1, 2])

    fnames = [
        a["victim"]["name"]
        for a in fuzzy_results
    ]

    urgency = [
        a["urgency"]
        for a in fuzzy_results
    ]

    risk = [
        a["route_risk"]
        for a in fuzzy_results
    ]

    xf = np.arange(len(fnames))

    ax_fz.bar(
        xf - 0.2,
        urgency,
        0.35,
        label="Urgency"
    )

    ax_fz.bar(
        xf + 0.2,
        risk,
        0.35,
        label="Risk"
    )

    ax_fz.set_xticks(xf)

    ax_fz.set_xticklabels(fnames)

    ax_fz.legend()

    ax_fz.set_title(
        "Fuzzy Risk & Urgency",
        fontweight='bold'
    )

    # ─────────────────────────────────────
    # RESOURCE USAGE
    # ─────────────────────────────────────

    ax_res = fig.add_subplot(gs[1, 3])

    res_names = list(kpi.resource_usage.keys())

    res_vals = list(kpi.resource_usage.values())

    ax_res.pie(
        res_vals,
        labels=res_names,
        autopct='%1.0f%%'
    )

    ax_res.set_title(
        "Resource Usage",
        fontweight='bold'
    )

    # ─────────────────────────────────────
    # HILL CLIMBING GRAPH
    # ─────────────────────────────────────

    ax_hc = fig.add_subplot(gs[2, :2])

    ax_hc.plot(hc_history)

    ax_hc.set_title(
        "Hill Climbing Convergence",
        fontweight='bold'
    )

    ax_hc.set_xlabel("Iteration")

    ax_hc.set_ylabel("Cost")

    # ─────────────────────────────────────
    # KPI TABLE
    # ─────────────────────────────────────

    ax_tbl = fig.add_subplot(gs[2, 2:])

    ax_tbl.axis('off')

    table_data = [

        ["Victims Saved",
         f"{kpi.victims_saved}/{kpi.total_victims}"],

        ["Avg Rescue Time",
         f"{kpi.avg_rescue_time():.1f} min"],

        ["Risk Exposure",
         str(kpi.risk_exposure_score)],

        ["Replan Events",
         str(kpi.replan_events)]
    ]

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=["KPI", "Value"],
        loc='center',
        cellLoc='center'
    )

    tbl.auto_set_font_size(False)

    tbl.set_fontsize(9)

    tbl.scale(1, 2)

    ax_tbl.set_title(
        "KPI Summary",
        fontweight='bold'
    )

    fig.suptitle(
        "AIDRA Dashboard",
        fontsize=16,
        fontweight='bold'
    )

    return fig

# ─────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":

    env, kpi, ml_results, fuzzy_results, \
    all_paths, hc_history, algos = run_simulation()

    fig = visualise_dashboard(
        env,
        kpi,
        ml_results,
        fuzzy_results,
        all_paths,
        hc_history,
        algos
    )

    # Create output folder safely

    output_dir = "outputs"

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        "aidra_dashboard.png"
    )

    # Save figure safely

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches='tight'
    )

    plt.close(fig)

    print(f"\n✅ Dashboard saved successfully → {output_file}")

    print("✅ AIDRA simulation complete!")

    print("\n📁 Files Ready:")
    print("step1_environment.py")
    print("step2_csp.py")
    print("step3_ml.py")
    print("step4_fuzzy.py")
    print("step5_main.py")
