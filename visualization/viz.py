"""
CIPHER multi-robot motion planning visualizer.

Supports both 2D and 3D environments — set `dimensions: 2` or `dimensions: 3`
(default: 3) in the log file header.

Usage:
    python viz.py run.yaml                        # ← → to navigate events, then animation
    python viz.py run.yaml --env problem.yaml     # also draw obstacles from problem file
    python viz.py run.yaml --auto                 # auto-advance (default 1.5s), then animation
    python viz.py run.yaml --auto --delay 0.8
    python viz.py run.yaml --anim-speed 2.0       # animation playback speed multiplier
"""

import argparse
import copy
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import Slider

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROBOT_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#17becf',  # cyan
    '#bcbd22',  # yellow-green
]

OBSTACLE_COLOR = '#444444'
OBSTACLE_ALPHA = 0.55

CELL_COLOR = '#888888'
CELL_LW = 0.8
CELL_ALPHA = 0.55

COUPLED_CELL_COLOR = '#ff8c00'   # dark orange
COUPLED_CELL_LW = 2.5

COLLISION_COLOR = '#ff0000'

ANIM_FPS = 30   # real frames per second during robot animation

# 3-D box: corners indexed by binary (z_high, y_high, x_high):
#   0=mmm  1=Mmm  2=mMm  3=MMm  4=mmM  5=MmM  6=mMM  7=MMM
EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),  # bottom face
    (4, 5), (5, 7), (7, 6), (6, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]

FACES = [
    [0, 1, 3, 2],  # bottom  (z = min)
    [4, 5, 7, 6],  # top     (z = max)
    [0, 1, 5, 4],  # front   (y = min)
    [2, 3, 7, 6],  # back    (y = max)
    [0, 2, 6, 4],  # left    (x = min)
    [1, 3, 7, 5],  # right   (x = max)
]


# ---------------------------------------------------------------------------
# Module-level geometry helpers
# ---------------------------------------------------------------------------

def _corners3d(bmin, bmax):
    """Return (8, 3) array of box corners in binary-xyz order."""
    mn, mx = np.asarray(bmin, float), np.asarray(bmax, float)
    return np.array([
        [mn[0], mn[1], mn[2]],
        [mx[0], mn[1], mn[2]],
        [mn[0], mx[1], mn[2]],
        [mx[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]],
        [mx[0], mn[1], mx[2]],
        [mn[0], mx[1], mx[2]],
        [mx[0], mx[1], mx[2]],
    ])


def _center(bmin, bmax):
    return (np.asarray(bmin, float) + np.asarray(bmax, float)) / 2.0


def interpolate_position(path, t, n_spatial=3):
    """
    Linearly interpolate position at continuous time t along a path.
    Returns an array of length n_spatial.
    """
    cumulative = 0.0
    for i, s in enumerate(path[:-1]):
        dur = float(s['duration'])
        if cumulative + dur >= t:
            alpha = (t - cumulative) / dur if dur > 0 else 0.0
            p0 = np.asarray(s['state'][:n_spatial], float)
            p1 = np.asarray(path[i + 1]['state'][:n_spatial], float)
            return p0 + alpha * (p1 - p0)
        cumulative += dur
    return np.asarray(path[-1]['state'][:n_spatial], float)


def path_duration(path):
    """Return total travel time of a path."""
    return sum(float(s['duration']) for s in path)


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class Visualizer:
    def __init__(self, log_path, env_path=None, auto=False, delay=1.5, anim_speed=1.0):
        self.auto = auto
        self.delay = delay
        self.anim_speed = anim_speed
        self._advance_flag = False
        self._key_direction = 1

        # Load obstacles from problem/environment file
        self.obstacles = []
        if env_path is not None:
            with open(env_path) as f:
                env_data = yaml.safe_load(f)
            for obs in env_data.get('environment', {}).get('obstacles', []):
                if obs.get('type') == 'box':
                    center = np.asarray(obs['center'], float)
                    size   = np.asarray(obs['size'],   float)
                    self.obstacles.append((center - size / 2, center + size / 2))

        # Load log
        with open(log_path) as f:
            data = yaml.safe_load(f)

        header = data['header']
        self.events = data.get('events', [])

        # Dimension mode
        self.is_3d = int(header.get('dimensions', 3)) == 3
        self._n = 3 if self.is_3d else 2

        # Mutable planning state (reset between replays)
        self.robots = {}
        self.robot_colors = {}
        self.cells = {}
        self.low_level_paths = {}
        self.raw_paths = {}
        self.segments = {}
        self.coupled_groups = []
        self.mapf_paths = {}
        # Each entry: {'robots': set, 'resolved': set, 'artists': list[Artist]}
        self.collisions = []

        for i, r in enumerate(header['robots']):
            rid = r['id']
            self.robots[rid] = r
            self.robot_colors[rid] = ROBOT_COLORS[i % len(ROBOT_COLORS)]

        # Save initial cell layout for replay resets
        self._initial_cells = copy.deepcopy(
            {c['id']: c for c in header['grid']['cells']}
        )
        self.cells = copy.deepcopy(self._initial_cells)

        # Figure and axes
        self.fig = plt.figure(figsize=(13, 9))
        self.fig.patch.set_facecolor('white')

        if self.is_3d:
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.grid(False)
            self.ax.set_facecolor('white')
            # Remove pane fills and edges so only our cell wireframes show
            for pane in (self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor('none')
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_aspect('equal')
            self.ax.set_facecolor('white')
            # No background grid — our cell outlines serve that purpose
            self.ax.grid(False)
            for spine in self.ax.spines.values():
                spine.set_visible(False)
            self.ax.tick_params(left=False, bottom=False,
                                labelleft=False, labelbottom=False)

        self.artists = {}

        # Stable axis limits: tight margin (4%) around the workspace
        all_mins = [np.asarray(c['bounds']['min'], float)
                    for c in header['grid']['cells']]
        all_maxs = [np.asarray(c['bounds']['max'], float)
                    for c in header['grid']['cells']]
        env_min = np.min(all_mins, axis=0)
        env_max = np.max(all_maxs, axis=0)
        span = env_max - env_min
        margin = np.where(span > 0, span * 0.04, 0.2)

        self.ax.set_xlim(env_min[0] - margin[0], env_max[0] + margin[0])
        self.ax.set_ylim(env_min[1] - margin[1], env_max[1] + margin[1])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        if self.is_3d:
            self.ax.set_zlim(env_min[2] - margin[2], env_max[2] + margin[2])
            self.ax.set_zlabel('Z')

        self._y_span = float(env_max[1] - env_min[1])

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._draw_all_cells()
        self._draw_obstacles()
        self._draw_starts_goals()
        plt.tight_layout()
        plt.draw()

    # ------------------------------------------------------------------
    # Dimension-agnostic drawing helpers
    # ------------------------------------------------------------------

    def _pos(self, state):
        return np.asarray(state[:self._n], float)

    def _cell_center(self, cell):
        return _center(cell['bounds']['min'], cell['bounds']['max'])[:self._n]

    def _draw_cell_edges(self, bmin, bmax, color, lw):
        arts = []
        if self.is_3d:
            corners = _corners3d(bmin, bmax)
            for i, j in EDGES:
                line, = self.ax.plot3D(
                    [corners[i, 0], corners[j, 0]],
                    [corners[i, 1], corners[j, 1]],
                    [corners[i, 2], corners[j, 2]],
                    color=color, lw=lw, alpha=CELL_ALPHA,
                )
                arts.append(line)
        else:
            x0, y0 = bmin[0], bmin[1]
            x1, y1 = bmax[0], bmax[1]
            line, = self.ax.plot(
                [x0, x1, x1, x0, x0],
                [y0, y0, y1, y1, y0],
                color=color, lw=lw, alpha=CELL_ALPHA,
            )
            arts.append(line)
        return arts

    def _fill_cell(self, bmin, bmax, color, alpha=0.20):
        if self.is_3d:
            corners = _corners3d(bmin, bmax)
            face_verts = [[corners[k].tolist() for k in face] for face in FACES]
            poly = Poly3DCollection(face_verts, alpha=alpha,
                                    linewidths=0.0, edgecolors='none')
            poly.set_facecolor(color)
            self.ax.add_collection3d(poly)
            return poly
        else:
            verts = [
                [bmin[0], bmin[1]], [bmax[0], bmin[1]],
                [bmax[0], bmax[1]], [bmin[0], bmax[1]],
            ]
            patch = MplPolygon(verts, alpha=alpha, facecolor=color, edgecolor='none')
            self.ax.add_patch(patch)
            return patch

    def _plot_path(self, pts, **kwargs):
        if self.is_3d:
            line, = self.ax.plot3D(pts[:, 0], pts[:, 1], pts[:, 2], **kwargs)
        else:
            line, = self.ax.plot(pts[:, 0], pts[:, 1], **kwargs)
        return line

    def _scatter_pt(self, pos, **kwargs):
        if self.is_3d:
            return self.ax.scatter([pos[0]], [pos[1]], [pos[2]], **kwargs)
        else:
            return self.ax.scatter([pos[0]], [pos[1]], **kwargs)

    def _text_pt(self, pos, s, **kwargs):
        if self.is_3d:
            return self.ax.text(pos[0], pos[1], pos[2], s, **kwargs)
        else:
            return self.ax.text(pos[0], pos[1], s, **kwargs)

    def _draw_mapf_line(self, centers, color):
        c = np.array(centers)
        if self.is_3d:
            line, = self.ax.plot3D(c[:, 0], c[:, 1], c[:, 2],
                                   '--', color=color, lw=1.5, alpha=0.75)
        else:
            line, = self.ax.plot(c[:, 0], c[:, 1],
                                 '--', color=color, lw=1.5, alpha=0.75)
        return line

    # ------------------------------------------------------------------
    # Input control
    # ------------------------------------------------------------------

    def _on_key(self, event):
        self._key_direction = -1 if event.key == 'left' else 1
        self._advance_flag = True

    def _wait_for_input(self):
        """Block until a key is pressed. Returns +1 (→ or any key) or -1 (←)."""
        self._advance_flag = False
        self._key_direction = 1
        while not self._advance_flag:
            plt.pause(0.05)
            if not plt.fignum_exists(self.fig.number):
                sys.exit(0)
        return self._key_direction

    # ------------------------------------------------------------------
    # Artist lifecycle
    # ------------------------------------------------------------------

    def _clear_artists(self, key):
        for a in self.artists.get(key, []):
            try:
                a.remove()
            except Exception:
                pass
        self.artists[key] = []

    def _reset_to_initial(self):
        """Clear all drawn artists and restore the initial grid/robot state."""
        for key in list(self.artists.keys()):
            self._clear_artists(key)
        for col in self.collisions:
            for a in col['artists']:
                try:
                    a.remove()
                except Exception:
                    pass
        self.cells = copy.deepcopy(self._initial_cells)
        self.low_level_paths = {}
        self.raw_paths = {}
        self.segments = {}
        self.coupled_groups = []
        self.mapf_paths = {}
        self.collisions = []
        self._draw_all_cells()
        self._draw_obstacles()
        self._draw_starts_goals()

    # ------------------------------------------------------------------
    # Persistent scene elements
    # ------------------------------------------------------------------

    def _draw_obstacles(self):
        self._clear_artists('obstacles')
        arts = []
        for bmin, bmax in self.obstacles:
            arts.append(self._fill_cell(bmin, bmax, OBSTACLE_COLOR, alpha=OBSTACLE_ALPHA))
            arts.extend(self._draw_cell_edges(bmin, bmax, OBSTACLE_COLOR, lw=1.2))
        self.artists['obstacles'] = arts

    def _draw_all_cells(self, coupled_ids=None):
        if coupled_ids is None:
            coupled_ids = set()

        self._clear_artists('cells')
        arts = []

        for cid, cell in self.cells.items():
            bmin = cell['bounds']['min']
            bmax = cell['bounds']['max']
            color = COUPLED_CELL_COLOR if cid in coupled_ids else CELL_COLOR
            lw    = COUPLED_CELL_LW    if cid in coupled_ids else CELL_LW

            arts.extend(self._draw_cell_edges(bmin, bmax, color, lw))
            ctr = self._cell_center(cell)
            arts.append(self._text_pt(ctr, cid,
                                      fontsize=6, color='#444444',
                                      ha='center', va='center'))

        self.artists['cells'] = arts

    def _draw_starts_goals(self):
        self._clear_artists('starts_goals')
        arts = []
        for rid, r in self.robots.items():
            color = self.robot_colors[rid]
            s_pos = self._pos(r['start'])
            g_pos = self._pos(r['goal'])
            kw = {'depthshade': False} if self.is_3d else {}
            arts.append(self._scatter_pt(s_pos, c=[color], marker='^',
                                         s=120, zorder=5, **kw))
            arts.append(self._scatter_pt(g_pos, c=[color], marker='*',
                                         s=200, zorder=5, **kw))
        self.artists['starts_goals'] = arts

    def _status_text(self, msg):
        self.fig.suptitle(msg, fontsize=11, y=0.99)
        plt.draw()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def handle_mapf(self, event):
        self._clear_artists('mapf_fills')
        self._clear_artists('mapf_lines')
        fills, lines = [], []

        self.mapf_paths = dict(event.get('paths', {}))
        self.coupled_groups = []

        for rid, cell_ids in self.mapf_paths.items():
            color = self.robot_colors.get(rid, 'gray')
            centers = []
            for cid in cell_ids:
                cell = self.cells.get(cid)
                if cell is None:
                    continue
                fills.append(self._fill_cell(cell['bounds']['min'],
                                             cell['bounds']['max'],
                                             color, alpha=0.20))
                centers.append(self._cell_center(cell))
            if len(centers) >= 2:
                lines.append(self._draw_mapf_line(centers, color))

        self.artists['mapf_fills'] = fills
        self.artists['mapf_lines'] = lines
        self._draw_all_cells()

    def handle_raw_trajectory(self, event):
        self._clear_artists('raw_paths')
        self.raw_paths = dict(event.get('paths', {}))
        arts = []
        for rid, path in self.raw_paths.items():
            color = self.robot_colors.get(rid, 'gray')
            if not path:
                continue
            pts = np.array([self._pos(s['state']) for s in path])
            if len(pts) >= 2:
                arts.append(self._plot_path(pts, color=color, lw=1.8,
                                            alpha=0.45, linestyle='--',
                                            label=f'{rid} (raw)'))
        self.artists['raw_paths'] = arts
        self._draw_starts_goals()

    def handle_low_level_paths(self, event):
        """'control' field is present in the log but ignored by the visualizer."""
        self._clear_artists('paths')
        self._clear_artists('segment_markers')
        new_paths = event.get('paths', {})
        self.low_level_paths.update(new_paths)
        # Invalidate segments only for robots whose paths changed
        for rid in new_paths:
            self.segments.pop(rid, None)
        # Resolve collisions whose affected robots all have new paths
        updated = set(new_paths.keys())
        remaining = []
        for col in self.collisions:
            col['resolved'] |= updated & col['robots']
            if col['resolved'] >= col['robots']:
                for a in col['artists']:
                    try:
                        a.remove()
                    except Exception:
                        pass
            else:
                remaining.append(col)
        self.collisions = remaining
        arts = []

        for rid, path in self.low_level_paths.items():
            color = self.robot_colors.get(rid, 'gray')
            if not path:
                continue
            pts = np.array([self._pos(s['state']) for s in path])
            if len(pts) >= 2:
                arts.append(self._plot_path(pts, color=color, lw=2.2,
                                            alpha=0.9, label=rid))

        self.artists['paths'] = arts
        self._draw_starts_goals()

    def handle_segments(self, event):
        self._clear_artists('segment_markers')
        self.segments.update(event.get('segments', {}))
        arts = []

        for rid, segs in self.segments.items():
            color = self.robot_colors.get(rid, 'gray')
            path = self.low_level_paths.get(rid)
            if not path:
                continue
            marked = set()
            for seg in segs:
                for idx in seg:
                    if idx not in marked and 0 <= idx < len(path):
                        marked.add(idx)
                        pos = self._pos(path[idx]['state'])
                        kw = {'depthshade': False} if self.is_3d else {}
                        arts.append(self._scatter_pt(
                            pos, c=[color], s=90, marker='o',
                            edgecolors='black', linewidths=1.0,
                            zorder=4, alpha=0.95, **kw,
                        ))

        self.artists['segment_markers'] = arts

    def handle_collision(self, event):
        robots_involved = event.get('robots', [])
        t = float(event.get('time', 0.0))
        arts = []
        positions = []

        for rid in robots_involved:
            path = self.low_level_paths.get(rid)
            if not path:
                continue
            pos = interpolate_position(path, t, self._n)
            positions.append(pos)
            kw = {'depthshade': False} if self.is_3d else {}
            arts.append(self._scatter_pt(
                pos, c=[COLLISION_COLOR], s=350, marker='o',
                alpha=0.75, zorder=6,
                edgecolors='darkred', linewidths=2.0, **kw,
            ))

        if positions:
            label_pos = positions[0].copy()
            label = f'COLLISION t={t:.2f}s\n({", ".join(robots_involved)})'
            if self.is_3d:
                zlim = self.ax.get_zlim()
                label_pos[2] += (zlim[1] - zlim[0]) * 0.08
            else:
                label_pos[1] += self._y_span * 0.06
            arts.append(self._text_pt(label_pos, label,
                                      color='red', fontsize=8,
                                      fontweight='bold',
                                      ha='center', va='bottom'))

        self.collisions.append({
            'robots':   set(robots_involved),
            'resolved': set(),
            'artists':  arts,
        })

    def handle_grid_update(self, event):
        for cid in event.get('removed', []):
            self.cells.pop(cid, None)
        for c in event.get('cells', []):
            self.cells[c['id']] = c
        self.coupled_groups = []
        self._draw_all_cells()

    def handle_coupled_planning(self, event):
        self.coupled_groups = event.get('groups', [])
        coupled_ids = {cid
                       for group in self.coupled_groups
                       for cid in group.get('cells', [])}
        self._draw_all_cells(coupled_ids=coupled_ids)

    # ------------------------------------------------------------------
    # Robot animation
    # ------------------------------------------------------------------

    def _animate_final_paths(self):
        """Animate robots as moving dots along their current low-level paths (or raw if no opt)."""
        paths_to_animate = self.low_level_paths or self.raw_paths
        if not paths_to_animate:
            return

        total_time = max(
            (path_duration(p) for p in paths_to_animate.values() if p),
            default=0.0,
        )
        if total_time <= 0:
            return

        dt_real = 1.0 / ANIM_FPS
        dt_sim  = self.anim_speed / ANIM_FPS

        dim_str = '3D' if self.is_3d else '2D'

        # Reserve bottom space for the scrub slider
        self.fig.subplots_adjust(bottom=0.12)
        ax_slider = self.fig.add_axes([0.1, 0.04, 0.8, 0.03])
        slider = Slider(ax_slider, 'Time (s)', 0.0, total_time,
                        valinit=0.0, color='steelblue')

        self._status_text(
            f'[{dim_str}] Animating — drag slider to scrub, any key to finish'
        )
        plt.draw()

        # Reset advance flag so a previous keypress doesn't skip immediately
        self._advance_flag = False
        self._scrub_time = None

        robot_markers = {}

        def draw_robots_at(t):
            for rid, path in paths_to_animate.items():
                pos = interpolate_position(path, min(t, path_duration(path)), self._n)
                color = self.robot_colors.get(rid, 'gray')
                if rid in robot_markers:
                    try:
                        robot_markers[rid].remove()
                    except Exception:
                        pass
                kw = {'depthshade': False} if self.is_3d else {}
                robot_markers[rid] = self._scatter_pt(
                    pos, c=[color], s=180, marker='o',
                    zorder=7, edgecolors='black', linewidths=1.5, **kw,
                )

        def on_slider_changed(val):
            self._scrub_time = val
            draw_robots_at(val)
            self.fig.canvas.draw_idle()

        slider.on_changed(on_slider_changed)

        t = 0.0
        while t <= total_time + dt_sim:
            if not plt.fignum_exists(self.fig.number):
                return
            if self._advance_flag:
                break

            # Jump to scrubbed position if the user dragged the slider
            if self._scrub_time is not None:
                t = self._scrub_time
                self._scrub_time = None

            # Sync slider without re-triggering the callback
            slider.eventson = False
            slider.set_val(t)
            slider.eventson = True

            draw_robots_at(t)
            plt.draw()
            plt.pause(dt_real)
            t += dt_sim

        # Keep scrubber active until a key is pressed
        self._advance_flag = False
        self._status_text(
            f'[{dim_str}] Drag slider to scrub — press any key to continue'
        )
        while not self._advance_flag:
            if not plt.fignum_exists(self.fig.number):
                return
            plt.pause(0.05)

        # Remove animation markers and slider, restore layout
        for marker in robot_markers.values():
            try:
                marker.remove()
            except Exception:
                pass
        ax_slider.remove()
        self.fig.subplots_adjust(bottom=0.05)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        n = len(self.events)
        dim_str = '3D' if self.is_3d else '2D'

        handlers = {
            'mapf':             self.handle_mapf,
            'raw_trajectory':   self.handle_raw_trajectory,
            'low_level_paths':  self.handle_low_level_paths,
            'segments':         self.handle_segments,
            'collision':        self.handle_collision,
            'grid_update':      self.handle_grid_update,
            'coupled_planning': self.handle_coupled_planning,
        }

        def apply_event(event):
            etype = event.get('type', 'unknown')
            handler = handlers.get(etype)
            if handler:
                handler(event)
            else:
                print(f'Warning: unknown event type "{etype}"', file=sys.stderr)

        def replay_to(target_i):
            """Reset to initial state and replay events 0 … target_i."""
            self._reset_to_initial()
            for j in range(target_i + 1):
                apply_event(self.events[j])

        if self.auto:
            # ---- Auto mode: forward-only ----
            for i, event in enumerate(self.events):
                if not plt.fignum_exists(self.fig.number):
                    return
                etype = event.get('type', 'unknown')
                self._status_text(f'[{dim_str}] Event {i+1}/{n}: {etype}  (auto)')
                apply_event(event)
                plt.draw()
                plt.pause(self.delay)
        else:
            # ---- Manual mode: ← → bidirectional navigation ----
            self._status_text(
                f'CIPHER [{dim_str}] — {n} events  '
                f'(← previous  →/any key next)'
            )
            plt.draw()

            i = -1   # -1 = initial state, no events applied yet

            while True:
                if not plt.fignum_exists(self.fig.number):
                    return

                direction = self._wait_for_input()
                new_i = i + direction

                if new_i >= n:          # stepped past the last event → animate
                    break
                if new_i < -1:          # can't go before the initial state
                    new_i = -1

                if new_i == i:
                    continue

                i = new_i

                if i == -1:
                    self._reset_to_initial()
                    self._status_text(
                        f'CIPHER [{dim_str}] — {n} events  '
                        f'(← previous  →/any key next)'
                    )
                else:
                    replay_to(i)
                    etype = self.events[i].get('type', 'unknown')
                    self._status_text(
                        f'[{dim_str}] Event {i+1}/{n}: {etype}  '
                        f'(← previous  →/any key next)'
                    )

                plt.draw()

        # ---- Animation ----
        if self.low_level_paths:
            self._animate_final_paths()

        self._status_text(f'[{dim_str}] Done — close window to exit.')
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='CIPHER multi-robot motion planning visualizer'
    )
    p.add_argument('log', help='Path to YAML solution/log file')
    p.add_argument('--env', default=None, metavar='ENV_YAML',
                   help='Path to problem/environment YAML file (to draw obstacles)')
    p.add_argument('--auto', action='store_true',
                   help='Auto-advance between events (default: key-press)')
    p.add_argument('--delay', type=float, default=1.5,
                   help='Seconds between events in --auto mode (default: 1.5)')
    p.add_argument('--anim-speed', type=float, default=1.0,
                   help='Animation playback speed multiplier (default: 1.0)')
    return p.parse_args()


def main():
    args = parse_args()
    viz = Visualizer(args.log, env_path=args.env, auto=args.auto,
                     delay=args.delay, anim_speed=args.anim_speed)
    viz.run()


if __name__ == '__main__':
    main()
