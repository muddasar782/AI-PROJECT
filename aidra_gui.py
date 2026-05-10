"""
AIDRA - Adaptive Intelligent Disaster Response Agent
=====================================================
Tkinter GUI Simulation
- Animated agents moving on a 10x10 grid
- A* pathfinding for route planning
- CSP-based victim prioritization
- Fuzzy logic for route decisions (safe vs fast)
- ML survival probability display
- Dynamic road blockages with replanning
- Full decision log + KPI sidebar
"""

import tkinter as tk
from tkinter import font as tkfont
import heapq
import time
import random

# ══════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════
CELL   = 58
GRID   = 10
W      = GRID * CELL
H      = GRID * CELL

# Cell types
EMPTY    = "empty"
BLOCKED  = "blocked"
HAZARD   = "hazard"
BASE     = "base"
HOSPITAL = "hospital"

COLORS = {
    EMPTY:    "#f0f4f8",
    BLOCKED:  "#2d3436",
    HAZARD:   "#fd7c2a",
    BASE:     "#00b894",
    HOSPITAL: "#0984e3",
}

SEV_COLOR  = {"critical": "#d63031", "moderate": "#e17055", "minor": "#00b894"}
SEV_SCORE  = {"critical": 3, "moderate": 2, "minor": 1}
RES_COLOR  = {"ambulance_1": "#0984e3", "ambulance_2": "#6c5ce7", "rescue_team": "#00b894"}
RES_LABEL  = {"ambulance_1": "AMB1", "ambulance_2": "AMB2", "rescue_team": "TEAM"}


HAZARD_CELLS  = {(3,3),(3,4),(4,3),(4,4),(4,5),(5,4)}
BLOCKED_CELLS = {(1,5),(2,5),(6,5),(6,6)}
BASE_POS      = (0,0)
HOSPITALS     = [(0,9),(9,9)]

VICTIMS_CFG = [
    {"id":1,"pos":(2,3),"severity":"critical","name":"V1","prob":0.20},
    {"id":2,"pos":(5,7),"severity":"critical","name":"V2","prob":0.40},
    {"id":3,"pos":(7,2),"severity":"moderate","name":"V3","prob":0.40},
    {"id":4,"pos":(3,8),"severity":"moderate","name":"V4","prob":0.60},
    {"id":5,"pos":(8,5),"severity":"minor",   "name":"V5","prob":0.40},
]

# ══════════════════════════════════════════
#  A* PATHFINDING
# ══════════════════════════════════════════
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid_blocked, start, goal, hazard_set, avoid_hazard=False):
    pq = [(0, start)]
    came_from = {}
    g = {start: 0}

    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            return list(reversed(path))

        r, c = cur
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if not (0<=nr<GRID and 0<=nc<GRID):
                continue
            if (nr,nc) in grid_blocked:
                continue
            step = 10 if (avoid_hazard and (nr,nc) in hazard_set) else 1
            ng = g[cur] + step
            if (nr,nc) not in g or ng < g[(nr,nc)]:
                g[(nr,nc)] = ng
                came_from[(nr,nc)] = cur
                heapq.heappush(pq, (ng + heuristic((nr,nc), goal), (nr,nc)))
    return []

# ══════════════════════════════════════════
#  VICTIM CLASS
# ══════════════════════════════════════════
class Victim:
    def __init__(self, cfg):
        self.id       = cfg["id"]
        self.pos      = cfg["pos"]
        self.severity = cfg["severity"]
        self.name     = cfg["name"]
        self.prob     = cfg["prob"]
        self.rescued  = False
        self.time_left= {"critical":20,"moderate":40,"minor":60}[self.severity]

# ══════════════════════════════════════════
#  AGENT CLASS
# ══════════════════════════════════════════
class Agent:
    def __init__(self, name, pos, color, label):
        self.name   = name
        self.pos    = list(pos)
        self.color  = color
        self.label  = label
        self.path   = []
        self.target = None
        self.moving = False
        self.route_type = "FAST"

# ══════════════════════════════════════════
#  MAIN SIMULATION APP
# ══════════════════════════════════════════
class AIDRAApp:
    def __init__(self, root):
        self.root  = root
        self.root.title("AIDRA — Adaptive Intelligent Disaster Response Agent")
        self.root.configure(bg="#1e272e")
        self.root.resizable(False, False)

        # Fonts
        self.f_title  = tkfont.Font(family="Courier", size=11, weight="bold")
        self.f_sub    = tkfont.Font(family="Courier", size=8)
        self.f_log    = tkfont.Font(family="Courier", size=8)
        self.f_kpi    = tkfont.Font(family="Courier", size=10, weight="bold")
        self.f_label  = tkfont.Font(family="Courier", size=7, weight="bold")

        # State
        self.running       = False
        self.paused        = False
        self.event_fired   = False
        self.step          = 0
        self.tick          = 0
        self.speed_ms      = 350
        self.blocked       = set(BLOCKED_CELLS)
        self.log_entries   = []
        self.rescued_count = 0
        self.risk_score    = 0
        self.replan_count  = 0
        self.rescue_times  = []
        self.after_id      = None

        self._build_ui()
        self._init_world()
        self._draw_all()

    # ── BUILD UI ──────────────────────────
    def _build_ui(self):
        # Title bar
        title_frame = tk.Frame(self.root, bg="#141d26", pady=6)
        title_frame.pack(fill="x")
        tk.Label(title_frame, text="AIDRA  —  ADAPTIVE INTELLIGENT DISASTER RESPONSE AGENT",
                 font=self.f_title, bg="#141d26", fg="#00b894").pack()
        tk.Label(title_frame, text="AIC-201  ·  Search + CSP + ML + Fuzzy Logic  ·  Dr. Arshad Farhad",
                 font=self.f_sub, bg="#141d26", fg="#636e72").pack()

        # Main area
        main = tk.Frame(self.root, bg="#1e272e")
        main.pack(fill="both", expand=True, padx=8, pady=6)

        # Grid canvas
        self.canvas = tk.Canvas(main, width=W, height=H,
                                bg="#dfe6e9", highlightthickness=2,
                                highlightbackground="#00b894")
        self.canvas.grid(row=0, column=0, rowspan=2, padx=(0,10))

        # Right sidebar
        sidebar = tk.Frame(main, bg="#1e272e", width=300)
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.grid_propagate(False)

        # KPI panel
        self._build_kpi_panel(sidebar)

        # Victim status
        self._build_victim_panel(sidebar)

        # Log panel
        self._build_log_panel(sidebar)

        # Bottom bar: buttons + phase
        bottom = tk.Frame(self.root, bg="#141d26", pady=8)
        bottom.pack(fill="x", padx=8)

        self.btn_run = tk.Button(bottom, text="▶  RUN SIMULATION",
                                 font=self.f_title, bg="#00b894", fg="white",
                                 relief="flat", padx=16, pady=6,
                                 command=self._start_sim, cursor="hand2")
        self.btn_run.pack(side="left", padx=4)

        self.btn_event = tk.Button(bottom, text="🔥  TRIGGER EVENT",
                                   font=self.f_title, bg="#d63031", fg="white",
                                   relief="flat", padx=16, pady=6,
                                   command=self._trigger_event, state="disabled",
                                   cursor="hand2")
        self.btn_event.pack(side="left", padx=4)

        self.btn_pause = tk.Button(bottom, text="⏸  PAUSE",
                                   font=self.f_title, bg="#636e72", fg="white",
                                   relief="flat", padx=16, pady=6,
                                   command=self._toggle_pause, state="disabled",
                                   cursor="hand2")
        self.btn_pause.pack(side="left", padx=4)

        self.btn_reset = tk.Button(bottom, text="↺  RESET",
                                   font=self.f_title, bg="#2d3436", fg="#b2bec3",
                                   relief="flat", padx=16, pady=6,
                                   command=self._reset, cursor="hand2")
        self.btn_reset.pack(side="left", padx=4)

        # Speed slider
        tk.Label(bottom, text="SPEED:", font=self.f_sub,
                 bg="#141d26", fg="#636e72").pack(side="left", padx=(16,4))
        self.speed_var = tk.IntVar(value=3)
        tk.Scale(bottom, from_=1, to=5, orient="horizontal",
                 variable=self.speed_var, bg="#141d26", fg="#b2bec3",
                 highlightthickness=0, troughcolor="#2d3436",
                 command=self._update_speed, length=100).pack(side="left")

        # Phase label
        self.phase_var = tk.StringVar(value="⬡  READY — Press RUN to start simulation")
        self.phase_lbl = tk.Label(bottom, textvariable=self.phase_var,
                                  font=self.f_sub, bg="#141d26", fg="#fdcb6e",
                                  wraplength=280, justify="left")
        self.phase_lbl.pack(side="left", padx=16)

    def _build_kpi_panel(self, parent):
        frame = tk.LabelFrame(parent, text=" KPIs ", font=self.f_label,
                              bg="#1e272e", fg="#00b894",
                              labelanchor="nw", bd=1, relief="solid",
                              highlightbackground="#00b894")
        frame.pack(fill="x", pady=(0,6))

        grid = tk.Frame(frame, bg="#1e272e")
        grid.pack(fill="x", padx=6, pady=4)

        def kpi_cell(parent, row, col, label, var, color):
            f = tk.Frame(parent, bg="#141d26", padx=8, pady=4)
            f.grid(row=row, column=col, padx=3, pady=3, sticky="ew")
            tk.Label(f, text=label, font=self.f_sub,
                     bg="#141d26", fg="#636e72").pack()
            tk.Label(f, textvariable=var, font=self.f_kpi,
                     bg="#141d26", fg=color).pack()
            parent.columnconfigure(col, weight=1)

        self.kv_saved   = tk.StringVar(value="0 / 5")
        self.kv_time    = tk.StringVar(value="—")
        self.kv_risk    = tk.StringVar(value="0")
        self.kv_replan  = tk.StringVar(value="0")
        self.kv_bt      = tk.StringVar(value="0")
        self.kv_hc      = tk.StringVar(value="37→23")

        kpi_cell(grid, 0, 0, "RESCUED",      self.kv_saved,  "#00b894")
        kpi_cell(grid, 0, 1, "AVG TIME",     self.kv_time,   "#0984e3")
        kpi_cell(grid, 1, 0, "RISK SCORE",   self.kv_risk,   "#fd7c2a")
        kpi_cell(grid, 1, 1, "REPLANS",      self.kv_replan, "#e17055")
        kpi_cell(grid, 2, 0, "CSP BKTRACKS", self.kv_bt,     "#00b894")
        kpi_cell(grid, 2, 1, "HC IMPROVE",   self.kv_hc,     "#a29bfe")

    def _build_victim_panel(self, parent):
        frame = tk.LabelFrame(parent, text=" VICTIMS ", font=self.f_label,
                              bg="#1e272e", fg="#d63031",
                              labelanchor="nw", bd=1, relief="solid")
        frame.pack(fill="x", pady=(0,6))

        self.victim_rows = {}
        for v in VICTIMS_CFG:
            row = tk.Frame(frame, bg="#141d26", pady=3, padx=6)
            row.pack(fill="x", padx=4, pady=2)

            dot = tk.Canvas(row, width=12, height=12, bg="#141d26",
                            highlightthickness=0)
            dot.create_oval(1,1,11,11, fill=SEV_COLOR[v["severity"]], outline="")
            dot.pack(side="left", padx=(0,6))

            tk.Label(row, text=f"{v['name']}  {v['severity'].upper()}",
                     font=self.f_sub, bg="#141d26",
                     fg=SEV_COLOR[v["severity"]]).pack(side="left")

            prob_lbl = tk.Label(row,
                                text=f"ML:{int(v['prob']*100)}%",
                                font=self.f_sub, bg="#141d26",
                                fg="#b2bec3")
            prob_lbl.pack(side="right")

            status_lbl = tk.Label(row, text="WAITING",
                                  font=self.f_sub, bg="#141d26", fg="#636e72")
            status_lbl.pack(side="right", padx=8)

            self.victim_rows[v["name"]] = {
                "frame": row, "status": status_lbl, "dot": dot,
                "severity": v["severity"]
            }

    def _build_log_panel(self, parent):
        frame = tk.LabelFrame(parent, text=" DECISION LOG ", font=self.f_label,
                              bg="#1e272e", fg="#fdcb6e",
                              labelanchor="nw", bd=1, relief="solid")
        frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(frame, height=14, width=36,
                                bg="#0d1117", fg="#b2bec3",
                                font=self.f_log, relief="flat",
                                wrap="word", state="disabled",
                                insertbackground="white")
        scroll = tk.Scrollbar(frame, command=self.log_text.yview,
                              bg="#2d3436", troughcolor="#1e272e")
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)

        # Tags for coloring
        self.log_text.tag_config("header",  foreground="#00b894", font=self.f_label)
        self.log_text.tag_config("event",   foreground="#d63031")
        self.log_text.tag_config("success", foreground="#00b894")
        self.log_text.tag_config("warn",    foreground="#fdcb6e")
        self.log_text.tag_config("info",    foreground="#74b9ff")
        self.log_text.tag_config("dim",     foreground="#636e72")

    # ── WORLD INIT ────────────────────────
    def _init_world(self):
        self.victims = [Victim(c) for c in VICTIMS_CFG]
        self.agents  = [
            Agent("ambulance_1", BASE_POS, RES_COLOR["ambulance_1"], "AMB1"),
            Agent("ambulance_2", BASE_POS, RES_COLOR["ambulance_2"], "AMB2"),
            Agent("rescue_team", BASE_POS, RES_COLOR["rescue_team"], "TEAM"),
        ]
        # Assign victims to agents (CSP result)
        self._csp_assign()

    def _csp_assign(self):
        """
        CSP solution (from step2_csp.py):
        V1(critical) → ambulance_1
        V2(critical) → ambulance_2
        V3(moderate) → ambulance_2
        V4(moderate) → ambulance_1
        V5(minor)    → rescue_team
        """
        self.assignment = {
            "V1": "ambulance_1",
            "V2": "ambulance_2",
            "V3": "ambulance_2",
            "V4": "ambulance_1",
            "V5": "rescue_team",
        }
        # Priority order (from step2_csp prioritize_victims)
        self.rescue_order = ["V1","V2","V4","V3","V5"]
        self.queue        = list(self.rescue_order)

        self._add_log("═══ SIMULATION START ═══", "header")
        self._add_log("CSP solved: 0 backtracks (MRV+FC)", "info")
        self._add_log("V1,V4→AMB1 | V2,V3→AMB2 | V5→TEAM", "dim")
        self._add_log("Hill Climbing: 37→23 cost units", "info")
        self._add_log("Rescue order: V1>V2>V4>V3>V5", "dim")

    # ── LOGGING ───────────────────────────
    def _add_log(self, msg, tag="info"):
        self.log_text.configure(state="normal")
        ts = time.strftime("%H:%M:%S")
        prefix = f"[{ts}] " if tag != "header" else "\n"
        self.log_text.insert("end", prefix + msg + "\n", tag)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ── DRAWING ───────────────────────────
    def _draw_all(self):
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_paths()
        self._draw_victims()
        self._draw_agents()
        self._draw_legend()

    def _draw_grid(self):
        for r in range(GRID):
            for c in range(GRID):
                x1,y1 = c*CELL, r*CELL
                x2,y2 = x1+CELL, y1+CELL
                pos = (r,c)

                if pos in self.blocked:
                    color = COLORS[BLOCKED]
                elif pos in HAZARD_CELLS:
                    color = "#ffe0b2"
                elif pos == BASE_POS:
                    color = "#d4f5ee"
                elif pos in HOSPITALS:
                    color = "#d0e8fb"
                else:
                    color = COLORS[EMPTY]

                self.canvas.create_rectangle(x1,y1,x2,y2,
                    fill=color, outline="#b2bec3", width=1)

                # Icons
                cx, cy = x1+CELL//2, y1+CELL//2
                if pos in self.blocked:
                    self.canvas.create_text(cx,cy,text="✖",
                        font=("Courier",14,"bold"), fill="#636e72")
                elif pos in HAZARD_CELLS:
                    self.canvas.create_text(cx,cy,text="🔥",
                        font=("Courier",14))
                elif pos == BASE_POS:
                    self.canvas.create_text(cx,cy-6,text="BASE",
                        font=("Courier",7,"bold"), fill="#00b894")
                    self.canvas.create_text(cx,cy+6,text="⬡",
                        font=("Courier",12,"bold"), fill="#00b894")
                elif pos in HOSPITALS:
                    self.canvas.create_text(cx,cy-6,text="HOSP",
                        font=("Courier",7,"bold"), fill="#0984e3")
                    self.canvas.create_text(cx,cy+6,text="✚",
                        font=("Courier",12,"bold"), fill="#0984e3")

                # Coords
                self.canvas.create_text(x1+5,y1+5,
                    text=f"{r},{c}", font=("Courier",5), fill="#b2bec3")

    def _draw_paths(self):
        colors_path = {
            "ambulance_1": "#0984e3",
            "ambulance_2": "#6c5ce7",
            "rescue_team": "#00b894",
        }
        for agent in self.agents:
            if agent.path:
                pts = [agent.pos] + list(agent.path)
                color = colors_path[agent.name]
                for i in range(len(pts)-1):
                    r1,c1 = pts[i];   r2,c2 = pts[i+1]
                    x1 = c1*CELL+CELL//2; y1 = r1*CELL+CELL//2
                    x2 = c2*CELL+CELL//2; y2 = r2*CELL+CELL//2
                    self.canvas.create_line(x1,y1,x2,y2,
                        fill=color, width=3, dash=(6,3),
                        arrow="last", arrowshape=(8,10,4))

    def _draw_victims(self):
        for v in self.victims:
            if v.rescued:
                continue
            r,c   = v.pos
            x1,y1 = c*CELL+8,  r*CELL+8
            x2,y2 = c*CELL+CELL-8, r*CELL+CELL-8
            color  = SEV_COLOR[v.severity]
            # Outer ring
            self.canvas.create_oval(x1-3,y1-3,x2+3,y2+3,
                outline=color, width=2, fill="")
            # Fill
            self.canvas.create_oval(x1,y1,x2,y2,
                fill=color, outline="white", width=1)
            # Label
            cx,cy = c*CELL+CELL//2, r*CELL+CELL//2
            self.canvas.create_text(cx,cy-5, text=v.name,
                font=("Courier",7,"bold"), fill="white")
            self.canvas.create_text(cx,cy+6,
                text=f"{int(v.prob*100)}%",
                font=("Courier",6), fill="white")
            # Time bar
            bar_w = int((v.time_left / 60) * (CELL-16))
            bar_color = "#00b894" if v.time_left>30 else "#fdcb6e" if v.time_left>15 else "#d63031"
            self.canvas.create_rectangle(
                c*CELL+8, r*CELL+CELL-10,
                c*CELL+8+bar_w, r*CELL+CELL-4,
                fill=bar_color, outline="")

    def _draw_agents(self):
        shape_offset = {"ambulance_1":0, "ambulance_2":4, "rescue_team":-4}
        for agent in self.agents:
            r,c   = agent.pos
            off   = shape_offset[agent.name]
            x1,y1 = c*CELL+10+off, r*CELL+10
            x2,y2 = c*CELL+CELL-10+off, r*CELL+CELL-10
            color  = agent.color
            # Shadow
            self.canvas.create_rectangle(x1+2,y1+2,x2+2,y2+2,
                fill="#2d3436", outline="")
            # Body
            self.canvas.create_rectangle(x1,y1,x2,y2,
                fill=color, outline="white", width=2)
            # Label
            cx = c*CELL+CELL//2+off//2
            cy = r*CELL+CELL//2
            self.canvas.create_text(cx,cy-4, text=agent.label,
                font=("Courier",7,"bold"), fill="white")
            if agent.target:
                self.canvas.create_text(cx,cy+6,
                    text=f"→{agent.target.name}",
                    font=("Courier",6), fill="white")

    def _draw_legend(self):
        items = [
            ("#d4f5ee","#00b894","BASE"),
            ("#d0e8fb","#0984e3","HOSPITAL"),
            ("#d63031","#d63031","CRITICAL"),
            ("#e17055","#e17055","MODERATE"),
            ("#00b894","#00b894","MINOR"),
            ("#ffe0b2","#fd7c2a","HAZARD"),
            ("#2d3436","#2d3436","BLOCKED"),
        ]
        x0, y0 = 6, H - 22
        for i,(fill,outline,label) in enumerate(items):
            x = x0 + i*74
            self.canvas.create_rectangle(x,y0,x+12,y0+12,
                fill=fill, outline=outline, width=1)
            self.canvas.create_text(x+16,y0+6, text=label,
                anchor="w", font=("Courier",6,"bold"), fill="#2d3436")

    # ── SIMULATION LOGIC ──────────────────
    def _start_sim(self):
        if self.running:
            return
        self.running = True
        self.btn_run.configure(state="disabled")
        self.btn_event.configure(state="normal")
        self.btn_pause.configure(state="normal")
        self._assign_next_targets()
        self._tick()

    def _assign_next_targets(self):
        """Assign next victim in queue to the correct agent (CSP assignment)."""
        for agent in self.agents:
            if agent.target is None:
                # Find next victim for this agent
                for vname in list(self.queue):
                    if self.assignment.get(vname) == agent.name:
                        victim = next((v for v in self.victims
                                       if v.name==vname and not v.rescued), None)
                        if victim:
                            avoid = self._fuzzy_use_safe(victim)
                            path  = astar(self.blocked,
                                          tuple(agent.pos),
                                          victim.pos,
                                          HAZARD_CELLS,
                                          avoid_hazard=avoid)
                            if path:
                                agent.target     = victim
                                agent.path       = path
                                agent.route_type = "SAFE" if avoid else "FAST"
                                self.queue.remove(vname)

                                # Update victim status
                                self._set_victim_status(vname, "EN ROUTE", "#fdcb6e")

                                route_str = "safe route (high risk)" if avoid else "fast route"
                                self._add_log(
                                    f"▶ {agent.label}→{vname} [{agent.route_type}]",
                                    "info"
                                )
                                self._add_log(
                                    f"  Fuzzy risk→{route_str}",
                                    "dim"
                                )
                            break

    def _fuzzy_use_safe(self, victim):
        """
        Simplified fuzzy logic decision:
        If victim is minor AND route passes through hazard → use safe route
        Otherwise → use fast route
        """
        fast_path = astar(self.blocked, BASE_POS, victim.pos, HAZARD_CELLS, False)
        hazard_count = sum(1 for p in fast_path if p in HAZARD_CELLS)
        hazard_level = min(hazard_count * 2.5, 10)
        return hazard_level > 6.0 and victim.severity == "minor"

    def _tick(self):
        if not self.running or self.paused:
            return

        self.tick += 1

        # Move each agent one step
        for agent in self.agents:
            if agent.path:
                next_pos = agent.path.pop(0)
                # Check if still valid (dynamic blockage)
                if next_pos in self.blocked:
                    self._replan_agent(agent)
                else:
                    agent.pos = list(next_pos)

            # Check arrival
            if agent.target and tuple(agent.pos) == agent.target.pos:
                self._rescue_victim(agent)

        # Update victim timers
        if self.tick % 2 == 0:
            for v in self.victims:
                if not v.rescued:
                    v.time_left = max(0, v.time_left - 1)

        # Try assigning new targets
        self._assign_next_targets()

        self._draw_all()
        self._update_kpis()

        # Check done
        if all(v.rescued for v in self.victims):
            self._on_complete()
            return

        self.after_id = self.root.after(self.speed_ms, self._tick)

    def _rescue_victim(self, agent):
        v = agent.target
        v.rescued = True
        self.rescued_count += 1
        self.rescue_times.append(
            abs(v.pos[0]-BASE_POS[0]) + abs(v.pos[1]-BASE_POS[1])
        )

        # Count hazard cells on route taken
        path_to_v = astar(self.blocked, BASE_POS, v.pos, HAZARD_CELLS, False)
        hazards = sum(1 for p in path_to_v if p in HAZARD_CELLS)
        if agent.route_type == "FAST":
            self.risk_score += hazards

        agent.target = None
        agent.path   = []

        self._set_victim_status(v.name, "✓ RESCUED", "#00b894")
        self._add_log(f"✓ {v.name} RESCUED by {agent.label}", "success")
        self._add_log(
            f"  Survival: {int(v.prob*100)}% | Risk: {hazards} cells",
            "dim"
        )
        self._update_kpis()

    def _replan_agent(self, agent):
        if not agent.target:
            return
        self.replan_count += 1
        new_path = astar(self.blocked,
                         tuple(agent.pos),
                         agent.target.pos,
                         HAZARD_CELLS, False)
        if new_path:
            agent.path = new_path
            self._add_log(
                f"⚠ REPLAN: {agent.label} rerouted to {agent.target.name}",
                "warn"
            )
        else:
            self._add_log(
                f"✖ No path! {agent.label} waiting...",
                "event"
            )

    def _trigger_event(self):
        if self.event_fired:
            return
        self.event_fired = True

        # Block new roads
        new_blocked = {(2,6),(3,6),(1,3)}
        self.blocked.update(new_blocked)

        self._add_log("🔥 DYNAMIC EVENT FIRED!", "event")
        self._add_log("  Roads (2,6)(3,6)(1,3) BLOCKED", "event")
        self._add_log("  Agents replanning via A*...", "warn")

        # Replan all active agents
        for agent in self.agents:
            if agent.target and agent.path:
                self._replan_agent(agent)

        self.phase_var.set("⚠  AFTERSHOCK! Roads blocked — A* replanning...")
        self.phase_lbl.configure(fg="#d63031")
        self.btn_event.configure(state="disabled")
        self._draw_all()

    def _toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.configure(text="▶  RESUME", bg="#fdcb6e", fg="#2d3436")
            self.phase_var.set("⏸  PAUSED")
        else:
            self.btn_pause.configure(text="⏸  PAUSE", bg="#636e72", fg="white")
            self.phase_var.set("▶  SIMULATION RUNNING...")
            self._tick()

    def _reset(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.running      = False
        self.paused       = False
        self.event_fired  = False
        self.tick         = 0
        self.blocked      = set(BLOCKED_CELLS)
        self.rescued_count= 0
        self.risk_score   = 0
        self.replan_count = 0
        self.rescue_times = []

        self.btn_run.configure(state="normal")
        self.btn_event.configure(state="disabled")
        self.btn_pause.configure(state="disabled",
                                 text="⏸  PAUSE", bg="#636e72", fg="white")
        self.phase_var.set("⬡  READY — Press RUN to start simulation")
        self.phase_lbl.configure(fg="#fdcb6e")

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0","end")
        self.log_text.configure(state="disabled")

        self._init_world()
        self._draw_all()
        self._update_kpis()

        for name in self.victim_rows:
            self._set_victim_status(name, "WAITING", "#636e72")

    def _update_speed(self, val):
        speeds = {1:800, 2:500, 3:350, 4:180, 5:80}
        self.speed_ms = speeds[int(val)]

    def _set_victim_status(self, name, text, color):
        if name in self.victim_rows:
            self.victim_rows[name]["status"].configure(text=text, fg=color)

    def _update_kpis(self):
        self.kv_saved.set(f"{self.rescued_count} / 5")
        avg = sum(self.rescue_times)/len(self.rescue_times)*2 \
              if self.rescue_times else 0
        self.kv_time.set(f"{avg:.1f} min" if avg else "—")
        self.kv_risk.set(str(self.risk_score))
        self.kv_replan.set(str(self.replan_count))
        self.kv_bt.set("0")
        self.kv_hc.set("37→23")

    def _on_complete(self):
        self.running = False
        self.phase_var.set("✓  ALL VICTIMS RESCUED — SIMULATION COMPLETE")
        self.phase_lbl.configure(fg="#00b894")
        self.btn_pause.configure(state="disabled")
        self.btn_run.configure(state="disabled")
        self._add_log("═══ SIMULATION COMPLETE ═══", "header")
        self._add_log("All 5 victims rescued!", "success")
        self._add_log(
            f"Risk={self.risk_score} | Replans={self.replan_count}",
            "dim"
        )


# ══════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = AIDRAApp(root)
    root.mainloop()
