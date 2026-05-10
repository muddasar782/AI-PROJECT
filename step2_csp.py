"""
AIDRA - Step 2: CSP Resource Allocation Solver
================================================
Constraint Satisfaction Problem:
  - Variables   : each victim needs to be assigned a resource
  - Domain      : ambulance_1, ambulance_2, rescue_team
  - Constraints :
      HC1 - max 2 victims per ambulance at a time
      HC2 - rescue_team can only handle 1 location at a time
      HC3 - critical victims MUST get an ambulance (not rescue team alone)
  - Heuristics  : MRV (Minimum Remaining Values) + Forward Checking
"""

import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from step1_environment import DisasterEnvironment, astar

# ─────────────────────────────────────────
# 1. VICTIM PRIORITY ORDERING
#    (used before CSP to decide rescue order)
# ─────────────────────────────────────────

SEVERITY_SCORE = {"critical": 3, "moderate": 2, "minor": 1}

def prioritize_victims(victims, env, hospital):
    """
    Score each victim by severity + proximity.
    Higher score = rescued sooner.
    """
    scored = []
    for v in victims:
        sev   = SEVERITY_SCORE[v["severity"]]
        route = astar(env, env.base_pos, v["pos"], avoid_hazard=False)
        dist  = route["cost"] if route else 999
        # Formula: severity matters more than distance
        score = (sev * 10) - dist
        scored.append({**v, "score": score, "dist": dist})
    scored.sort(key=lambda x: -x["score"])
    return scored


# ─────────────────────────────────────────
# 2. CSP DEFINITION
# ─────────────────────────────────────────

RESOURCES = ["ambulance_1", "ambulance_2", "rescue_team"]

# Hard constraint limits
MAX_VICTIMS_PER_AMBULANCE = 2
MAX_VICTIMS_PER_TEAM      = 2   # rescue team can handle up to 2 (non-critical)

def get_domain(victim):
    """Critical victims must get an ambulance (not just rescue team)."""
    if victim["severity"] == "critical":
        return ["ambulance_1", "ambulance_2"]
    return RESOURCES[:]   # all resources allowed


class CSPSolver:
    def __init__(self, victims):
        self.victims    = victims
        self.variables  = [v["id"] for v in victims]
        self.domains    = {v["id"]: get_domain(v) for v in victims}
        self.assignment = {}
        self.backtrack_count = 0

    # ── Constraint check ──────────────────
    def is_consistent(self, victim_id, resource, assignment):
        counts = {"ambulance_1": 0, "ambulance_2": 0, "rescue_team": 0}
        for vid, res in assignment.items():
            counts[res] += 1
        # Adding this resource — would it violate limits?
        counts[resource] += 1
        if resource in ["ambulance_1", "ambulance_2"]:
            return counts[resource] <= MAX_VICTIMS_PER_AMBULANCE
        if resource == "rescue_team":
            return counts[resource] <= MAX_VICTIMS_PER_TEAM
        return True

    # ── MRV heuristic: pick variable with smallest domain ──
    def select_unassigned_variable(self, assignment, domains):
        unassigned = [v for v in self.variables if v not in assignment]
        # MRV: choose the one with fewest remaining values
        return min(unassigned, key=lambda v: len(domains[v]))

    # ── Forward checking: prune domains after assignment ──
    def forward_check(self, victim_id, resource, domains, assignment):
        new_domains = copy.deepcopy(domains)
        # Build counts including the current assignment being made
        counts = {"ambulance_1": 0, "ambulance_2": 0, "rescue_team": 0}
        for vid, res in assignment.items():
            counts[res] += 1
        counts[resource] += 1   # include the new assignment

        cap = {"ambulance_1": MAX_VICTIMS_PER_AMBULANCE,
               "ambulance_2": MAX_VICTIMS_PER_AMBULANCE,
               "rescue_team": MAX_VICTIMS_PER_TEAM}

        # If a resource is now at capacity, prune it from remaining domains
        for vid in self.variables:
            if vid not in assignment and vid != victim_id:
                for res, cnt in counts.items():
                    if cnt >= cap[res] and res in new_domains[vid]:
                        new_domains[vid].remove(res)
                if not new_domains[vid]:
                    return None   # domain wipe-out — backtrack
        return new_domains

    # ── Backtracking search ───────────────
    def backtrack(self, assignment, domains):
        if len(assignment) == len(self.variables):
            return assignment   # complete!

        var = self.select_unassigned_variable(assignment, domains)

        for value in list(domains[var]):
            if self.is_consistent(var, value, assignment):
                new_assignment = {**assignment, var: value}
                new_domains = self.forward_check(var, value, domains, assignment)

                if new_domains is not None:
                    result = self.backtrack(new_assignment, new_domains)
                    if result is not None:
                        return result

                self.backtrack_count += 1

        return None   # failure

    def solve(self):
        solution = self.backtrack({}, copy.deepcopy(self.domains))
        return solution


# ─────────────────────────────────────────
# 3. RESOURCE ALLOCATION PLAN
# ─────────────────────────────────────────

def build_allocation_plan(victims_ordered, solution):
    """
    Given the CSP solution (victim_id → resource),
    build human-readable rescue trips.
    """
    ambulance_loads = {"ambulance_1": [], "ambulance_2": [], "rescue_team": []}
    for v in victims_ordered:
        vid      = v["id"]
        resource = solution[vid]
        ambulance_loads[resource].append(v)

    print("\n RESOURCE ALLOCATION PLAN")
    print("=" * 50)
    for resource, assigned in ambulance_loads.items():
        names = [f"{v['name']}({v['severity']})" for v in assigned]
        print(f"  {resource:<14} → {', '.join(names) if names else 'unassigned'}")

    print("\n RESCUE TRIP SCHEDULE")
    print("=" * 50)
    trips = []
    trip_num = 1
    for resource, victims_list in ambulance_loads.items():
        # Group into batches respecting capacity
        cap = MAX_VICTIMS_PER_AMBULANCE if "ambulance" in resource else MAX_VICTIMS_PER_TEAM
        for i in range(0, len(victims_list), cap):
            batch = victims_list[i:i+cap]
            names = [v["name"] for v in batch]
            print(f"  Trip {trip_num}: {resource} picks up {names}")
            trips.append({"trip": trip_num, "resource": resource, "victims": batch})
            trip_num += 1
    return trips


# ─────────────────────────────────────────
# 4. VISUALISE ALLOCATION
# ─────────────────────────────────────────

def visualise_allocation(victims_ordered, solution, trips):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: priority bar chart ──
    ax = axes[0]
    names  = [v["name"] for v in victims_ordered]
    scores = [v["score"] for v in victims_ordered]
    colors = {"critical": "#e63946", "moderate": "#f4a261", "minor": "#2a9d8f"}
    bar_colors = [colors[v["severity"]] for v in victims_ordered]
    bars = ax.barh(names, scores, color=bar_colors, edgecolor='white', height=0.6)
    ax.set_xlabel("Priority Score  (higher = rescued first)", fontsize=10)
    ax.set_title("Victim Rescue Priority Order", fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{score:.1f}", va='center', fontsize=9)
    legend = [mpatches.Patch(color=c, label=s.capitalize()) for s, c in colors.items()]
    ax.legend(handles=legend, fontsize=9)

    # ── Right: resource assignment ──
    ax2 = axes[1]
    resource_colors = {
        "ambulance_1": "#457b9d",
        "ambulance_2": "#1d3557",
        "rescue_team": "#e76f51"
    }
    resource_list = list(resource_colors.keys())
    for i, v in enumerate(victims_ordered):
        res   = solution[v["id"]]
        color = resource_colors[res]
        ax2.barh(v["name"], 1, left=resource_list.index(res),
                 color=color, edgecolor='white', height=0.5)
        ax2.text(resource_list.index(res) + 0.5, i,
                 f"{v['name']}", ha='center', va='center',
                 color='white', fontsize=9, fontweight='bold')

    ax2.set_xticks([0.5, 1.5, 2.5])
    ax2.set_xticklabels(["Ambulance 1", "Ambulance 2", "Rescue Team"], fontsize=10)
    ax2.set_title("CSP Resource Assignment", fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.set_xlim(0, 3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────
# 5. RUN DEMO
# ─────────────────────────────────────────

if __name__ == "__main__":
    env      = DisasterEnvironment()
    hospital = env.hospitals[0]

    print("=" * 55)
    print("  AIDRA — Step 2: CSP Resource Allocation")
    print("=" * 55)

    # Step A: prioritize victims
    victims_ordered = prioritize_victims(env.victims, env, hospital)
    print("\n VICTIM PRIORITY ORDER")
    print("-" * 55)
    print(f"{'Rank':<6}{'Name':<6}{'Severity':<12}{'Score':<10}{'Dist':<8}")
    print("-" * 55)
    for rank, v in enumerate(victims_ordered, 1):
        print(f"{rank:<6}{v['name']:<6}{v['severity']:<12}{v['score']:<10.1f}{v['dist']:<8}")

    # Step B: solve CSP
    solver   = CSPSolver(victims_ordered)
    solution = solver.solve()

    if solution:
        print(f"\nCSP solved! Backtracks needed: {solver.backtrack_count}")
        print("   (with MRV + Forward Checking heuristics)")
        build_allocation_plan(victims_ordered, solution)

        # Step C: compare with/without heuristics (just backtrack count)
        solver_no_heuristic = CSPSolver(victims_ordered)
        solver_no_heuristic.select_unassigned_variable = \
            lambda assignment, domains: next(v for v in solver_no_heuristic.variables
                                              if v not in assignment)
        solver_no_heuristic.solve()

        print("\nCSP COMPARISON")
        print(f"  With MRV + Forward Checking : {solver.backtrack_count} backtracks")
        print(f"  Without heuristics          : {solver_no_heuristic.backtrack_count} backtracks")
    else:
        print("No valid assignment found — insufficient resources!")

    # Visualise
    trips = build_allocation_plan(victims_ordered, solution)
    fig   = visualise_allocation(victims_ordered, solution, trips)
    fig.savefig("aidra_grid.png", dpi=150, bbox_inches='tight')
    print("\nCSP chart saved to aidra_csp.png")
    print("Step 2 complete! Next: ML risk estimation")
