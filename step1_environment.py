"""
AIDRA - Step 1: Grid Environment & Search Algorithms
=====================================================
This file sets up:
  - The grid map (10x10) with victims, hazard zones, blocked roads
  - The intelligent agent and environment model
  - Search algorithms: BFS, DFS, Greedy Best-First, A*
"""

import heapq
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────
# 1. CONSTANTS — cell types on the grid
# ─────────────────────────────────────────
EMPTY    = 0
BLOCKED  = 1   # wall / rubble — impassable
HAZARD   = 2   # high-risk zone (fire/aftershock) — passable but costly
VICTIM   = 3
BASE     = 4
HOSPITAL = 5

# ─────────────────────────────────────────
# 2. ENVIRONMENT — the 10×10 grid map
# ─────────────────────────────────────────
class DisasterEnvironment:
    def __init__(self):
        self.rows = 10
        self.cols = 10

        # Build base grid
        self.grid = np.zeros((self.rows, self.cols), dtype=int)

        # --- Fixed locations ---
        self.base_pos     = (0, 0)          # rescue base (top-left)
        self.hospitals    = [(0, 9), (9, 9)] # two medical centres

        # --- 5 Victims: (row, col, severity, name) ---
        # severity: 'critical', 'moderate', 'minor'
        self.victims = [
            {"id": 1, "pos": (2, 3), "severity": "critical", "name": "V1"},
            {"id": 2, "pos": (5, 7), "severity": "critical", "name": "V2"},
            {"id": 3, "pos": (7, 2), "severity": "moderate", "name": "V3"},
            {"id": 4, "pos": (3, 8), "severity": "moderate", "name": "V4"},
            {"id": 5, "pos": (8, 5), "severity": "minor",    "name": "V5"},
        ]

        # --- Hazard zones (high risk: fire / aftershock) ---
        self.hazard_cells = {(3,3),(3,4),(4,3),(4,4),(4,5),(5,4)}

        # --- Initially blocked roads ---
        self.blocked_cells = {(1,5),(2,5),(6,5),(6,6)}

        # --- Dynamic state ---
        self.dynamic_blocked = set()   # roads blocked mid-mission
        self.rescued = set()           # victim IDs already rescued

        self._build_grid()

    def _build_grid(self):
        """Refresh the numpy grid from current state."""
        self.grid[:] = EMPTY
        for r, c in self.hazard_cells:
            self.grid[r][c] = HAZARD
        for r, c in self.blocked_cells | self.dynamic_blocked:
            self.grid[r][c] = BLOCKED
        for v in self.victims:
            if v["id"] not in self.rescued:
                r, c = v["pos"]
                self.grid[r][c] = VICTIM
        r, c = self.base_pos
        self.grid[r][c] = BASE
        for r, c in self.hospitals:
            self.grid[r][c] = HOSPITAL

    def is_passable(self, pos):
        r, c = pos
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return False
        return self.grid[r][c] != BLOCKED

    def get_neighbors(self, pos):
        """4-directional movement."""
        r, c = pos
        candidates = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
        return [p for p in candidates if self.is_passable(p)]

    def move_cost(self, pos, avoid_hazard=False):
        """
        Cost to enter a cell.
        avoid_hazard=True → hazard cells cost 10x (risk-averse route)
        avoid_hazard=False → hazard cells cost 1 (fastest route)
        """
        r, c = pos
        if self.grid[r][c] == HAZARD:
            return 10 if avoid_hazard else 1
        return 1

    def trigger_dynamic_event(self, new_blocked):
        """Simulate a mid-mission road blockage."""
        self.dynamic_blocked.update(new_blocked)
        self._build_grid()
        print(f"\n⚠️  DYNAMIC EVENT: Roads blocked at {new_blocked} — replanning required!")

    def mark_rescued(self, victim_id):
        self.rescued.add(victim_id)
        self._build_grid()


# ─────────────────────────────────────────
# 3. SEARCH ALGORITHMS
# ─────────────────────────────────────────

def reconstruct_path(came_from, start, goal):
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from[node]
    path.append(start)
    return list(reversed(path))


def bfs(env, start, goal, avoid_hazard=False):
    """Breadth-First Search — finds shortest path (fewest hops)."""
    frontier = deque([start])
    came_from = {start: None}
    nodes_expanded = 0

    while frontier:
        current = frontier.popleft()
        nodes_expanded += 1
        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            cost = sum(env.move_cost(p, avoid_hazard) for p in path[1:])
            return {"path": path, "cost": cost, "nodes_expanded": nodes_expanded, "algorithm": "BFS"}
        for nb in env.get_neighbors(current):
            if nb not in came_from:
                came_from[nb] = current
                frontier.append(nb)
    return None  # no path found


def dfs(env, start, goal, avoid_hazard=False):
    """Depth-First Search — not optimal, but fast to implement."""
    frontier = [start]
    came_from = {start: None}
    nodes_expanded = 0

    while frontier:
        current = frontier.pop()
        nodes_expanded += 1
        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            cost = sum(env.move_cost(p, avoid_hazard) for p in path[1:])
            return {"path": path, "cost": cost, "nodes_expanded": nodes_expanded, "algorithm": "DFS"}
        for nb in env.get_neighbors(current):
            if nb not in came_from:
                came_from[nb] = current
                frontier.append(nb)
    return None


def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def greedy_best_first(env, start, goal, avoid_hazard=False):
    """Greedy Best-First — uses heuristic only, not cost-aware."""
    frontier = [(manhattan(start, goal), start)]
    came_from = {start: None}
    nodes_expanded = 0

    while frontier:
        _, current = heapq.heappop(frontier)
        nodes_expanded += 1
        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            cost = sum(env.move_cost(p, avoid_hazard) for p in path[1:])
            return {"path": path, "cost": cost, "nodes_expanded": nodes_expanded, "algorithm": "Greedy"}
        for nb in env.get_neighbors(current):
            if nb not in came_from:
                came_from[nb] = current
                heapq.heappush(frontier, (manhattan(nb, goal), nb))
    return None


def astar(env, start, goal, avoid_hazard=False):
    """A* — optimal + heuristic guided. Best of both worlds."""
    g_cost = {start: 0}
    frontier = [(manhattan(start, goal), start)]
    came_from = {start: None}
    nodes_expanded = 0

    while frontier:
        f, current = heapq.heappop(frontier)
        nodes_expanded += 1
        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            cost = g_cost[goal]
            return {"path": path, "cost": cost, "nodes_expanded": nodes_expanded, "algorithm": "A*"}
        for nb in env.get_neighbors(current):
            new_g = g_cost[current] + env.move_cost(nb, avoid_hazard)
            if nb not in g_cost or new_g < g_cost[nb]:
                g_cost[nb] = new_g
                came_from[nb] = current
                f_score = new_g + manhattan(nb, goal)
                heapq.heappush(frontier, (f_score, nb))
    return None


# ─────────────────────────────────────────
# 4. VISUALISATION
# ─────────────────────────────────────────

def visualise_grid(env, paths=None, title="AIDRA — Disaster Grid"):
    fig, ax = plt.subplots(figsize=(8, 8))
    color_map = {
        EMPTY:    "#f5f5f5",
        BLOCKED:  "#2d2d2d",
        HAZARD:   "#ff6b35",
        VICTIM:   "#e63946",
        BASE:     "#2a9d8f",
        HOSPITAL: "#457b9d",
    }

    for r in range(env.rows):
        for c in range(env.cols):
            val = env.grid[r][c]
            color = color_map.get(val, "#ffffff")
            rect = plt.Rectangle([c, env.rows-1-r], 1, 1,
                                  facecolor=color, edgecolor="#cccccc", linewidth=0.5)
            ax.add_patch(rect)

    # Label victims
    for v in env.victims:
        if v["id"] not in env.rescued:
            r, c = v["pos"]
            sev_symbol = {"critical": "!!!", "moderate": "!!", "minor": "!"}[v["severity"]]
            ax.text(c+0.5, env.rows-1-r+0.5, f"{v['name']}\n{sev_symbol}",
                    ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    # Label base & hospitals
    br, bc = env.base_pos
    ax.text(bc+0.5, env.rows-1-br+0.5, "BASE", ha='center', va='center',
            fontsize=7, color='white', fontweight='bold')
    for hr, hc in env.hospitals:
        ax.text(hc+0.5, env.rows-1-hr+0.5, "HOSP", ha='center', va='center',
                fontsize=7, color='white', fontweight='bold')

    # Draw paths
    path_colors = ["#06d6a0", "#ffd166", "#118ab2", "#ef476f"]
    if paths:
        for i, (result, label) in enumerate(paths):
            if result and result["path"]:
                color = path_colors[i % len(path_colors)]
                xs = [c+0.5 for r, c in result["path"]]
                ys = [env.rows-1-r+0.5 for r, c in result["path"]]
                ax.plot(xs, ys, color=color, linewidth=2.5, label=label, zorder=5, alpha=0.85)

    # Legend
    legend_patches = [
        mpatches.Patch(color=color_map[EMPTY],    label="Empty"),
        mpatches.Patch(color=color_map[BLOCKED],  label="Blocked"),
        mpatches.Patch(color=color_map[HAZARD],   label="Hazard zone"),
        mpatches.Patch(color=color_map[VICTIM],   label="Victim"),
        mpatches.Patch(color=color_map[BASE],     label="Base"),
        mpatches.Patch(color=color_map[HOSPITAL], label="Hospital"),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8)
    if paths:
        ax.legend(loc='lower right', fontsize=8)

    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_xticklabels(range(env.cols))
    ax.set_yticklabels(range(env.rows-1, -1, -1))
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────
# 5. RUN DEMO
# ─────────────────────────────────────────

if __name__ == "__main__":
    env = DisasterEnvironment()

    print("=" * 55)
    print("  AIDRA — Step 1: Grid & Search Demo")
    print("=" * 55)
    print(f"Grid size : {env.rows}x{env.cols}")
    print(f"Victims   : {len(env.victims)}")
    print(f"Hazards   : {len(env.hazard_cells)} cells")
    print(f"Blocked   : {len(env.blocked_cells)} roads")

    # Pick victim 1 (critical) → nearest hospital
    v1_pos   = env.victims[0]["pos"]   # (2,3)
    hospital = env.hospitals[0]        # (0,9)

    print(f"\nRoute: Base {env.base_pos} → Victim V1 {v1_pos}")
    print("-" * 55)

    start = env.base_pos
    goal  = v1_pos

    results = {
        "BFS":    bfs(env, start, goal, avoid_hazard=False),
        "DFS":    dfs(env, start, goal, avoid_hazard=False),
        "Greedy": greedy_best_first(env, start, goal, avoid_hazard=False),
        "A*":     astar(env, start, goal, avoid_hazard=False),
    }

    print(f"\n{'Algorithm':<10} {'Cost':>6} {'Nodes expanded':>16} {'Path length':>12}")
    print("-" * 48)
    for name, r in results.items():
        if r:
            print(f"{name:<10} {r['cost']:>6} {r['nodes_expanded']:>16} {len(r['path']):>12}")
        else:
            print(f"{name:<10}  No path found")

    # Show trade-off: A* with risk-avoidance vs speed
    print("\n--- Trade-off: Time vs Risk (A*) ---")
    fast   = astar(env, start, goal, avoid_hazard=False)
    safe   = astar(env, start, goal, avoid_hazard=True)
    if fast and safe:
        print(f"Fast route (ignore hazards): cost={fast['cost']}, path={fast['path']}")
        print(f"Safe route (avoid hazards):  cost={safe['cost']}, path={safe['path']}")
        decision = "FAST" if fast["cost"] < safe["cost"] else "SAFE"
        print(f"→ Agent decision: {decision} route selected")
        print(f"  Justification: victim is {'critical' if True else 'minor'} → time priority")

    # Dynamic event: block a road mid-mission
    print("\n--- Dynamic Event Test ---")
    env.trigger_dynamic_event({(1, 3), (1, 4)})
    replan = astar(env, start, goal, avoid_hazard=False)
    if replan:
        print(f"Replanned route: cost={replan['cost']}, path={replan['path']}")
    else:
        print("No path found after blockage — escalating!")

    # Save grid visualisation
    fig = visualise_grid(
        env,
        paths=[
            (fast,   "A* fast"),
            (safe,   "A* safe"),
            (replan, "A* replanned"),
        ],
        title="AIDRA — Grid Map with Search Paths"
    )
    fig.savefig("aidra_grid.png", dpi=150, bbox_inches='tight')
    print(" Grid saved to aidra_grid.png")
    print("Step 1 complete! Next: CSP solver (ambulance allocation)")
