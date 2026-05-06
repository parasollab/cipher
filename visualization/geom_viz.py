"""
Geometric planner output visualizer.

Visualizes environments and paths produced by geometric planners (e.g. decoupled_rrt).
Takes a separate environment YAML and an optional planner output YAML.

Usage:
    python geom_viz.py env.yaml                        # show environment only
    python geom_viz.py env.yaml output.yaml            # show environment + paths + animation
    python geom_viz.py env.yaml output.yaml --anim-speed 2.0
"""

import argparse
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Constants (match viz.py style)
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

ANIM_FPS = 30

# Robot collision geometry, matching FCL shapes in db-CBS/src/robots.cpp
ROBOT_GEOMETRY = {
    'single_integrator_0':               {'shape': 'sphere', 'radius': 0.1},
    'double_integrator_0':               {'shape': 'sphere', 'radius': 0.15},
    'unicycle_first_order_0':            {'shape': 'box',    'length': 0.5, 'width': 0.25},
    'unicycle_first_order_0_sphere':     {'shape': 'sphere', 'radius': 0.4},
    'unicycle_second_order_0':           {'shape': 'box',    'length': 0.5, 'width': 0.25},
    'car_first_order_0':                 {'shape': 'box',    'length': 0.5, 'width': 0.25},
    'car_first_order_with_1_trailers_0': {'shape': 'box',    'length': 0.5, 'width': 0.25},
}

# Fallback geometry when robot type is unknown
_DEFAULT_GEOM = {'shape': 'sphere', 'radius': 0.2}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------

class GeomVisualizer:
    def __init__(self, env_path, output_path=None, anim_speed=1.0, save_mp4=None):
        self.anim_speed = anim_speed
        self.save_mp4 = save_mp4
        self._advance_flag = False

        # Load environment
        with open(env_path) as f:
            env_data = yaml.safe_load(f)

        env = env_data['environment']
        self.env_min = np.array(env['min'], dtype=float)
        self.env_max = np.array(env['max'], dtype=float)
        self.obstacles = env.get('obstacles', []) or []
        self.robots_cfg = env_data.get('robots', [])

        # Load planner output (optional)
        if output_path is not None:
            with open(output_path) as f:
                out_data = yaml.safe_load(f)

            if not out_data.get('solved', False):
                print('ERROR: planner output indicates problem was not solved.', file=sys.stderr)
                sys.exit(1)

            self.planning_time = out_data.get('planning_time', None)
            result = out_data.get('result', [])
            self.paths = [r['states'] for r in result if 'states' in r]

            if len(self.paths) != len(self.robots_cfg):
                print(
                    f'Warning: {len(self.robots_cfg)} robots in env but '
                    f'{len(self.paths)} paths in output.',
                    file=sys.stderr,
                )
        else:
            self.planning_time = None
            self.paths = []

        self.total_steps = max((len(p) for p in self.paths), default=0)

        # Figure
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('white')
        self.ax.grid(False)
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.tick_params(left=True, bottom=True,
                            labelleft=True, labelbottom=True)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')

        span = self.env_max - self.env_min
        margin = np.where(span > 0, span * 0.06, 0.5)
        self.ax.set_xlim(self.env_min[0] - margin[0], self.env_max[0] + margin[0])
        self.ax.set_ylim(self.env_min[1] - margin[1], self.env_max[1] + margin[1])

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._draw_scene()
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        plt.draw()

    def _on_key(self, event):
        self._advance_flag = True

    # ------------------------------------------------------------------
    # Scene drawing
    # ------------------------------------------------------------------

    def _draw_scene(self):
        """Draw environment boundary, obstacles, paths, starts, and goals."""
        # Workspace boundary
        bx = [self.env_min[0], self.env_max[0], self.env_max[0],
              self.env_min[0], self.env_min[0]]
        by = [self.env_min[1], self.env_min[1], self.env_max[1],
              self.env_max[1], self.env_min[1]]
        self.ax.plot(bx, by, color='#333333', lw=1.5, zorder=1)

        # Obstacles
        for obs in self.obstacles:
            if not isinstance(obs, dict):
                print(f'Warning: unknown obstacle format, skipping: {obs}', file=sys.stderr)
                continue
            if 'min' in obs and 'max' in obs:
                omin = obs['min']
                omax = obs['max']
                w = omax[0] - omin[0]
                h = omax[1] - omin[1]
                ox, oy = omin[0], omin[1]
            elif obs.get('type') == 'box' and 'center' in obs and 'size' in obs:
                cx, cy = obs['center'][0], obs['center'][1]
                sw, sh = obs['size'][0], obs['size'][1]
                w, h = sw, sh
                ox, oy = cx - sw / 2.0, cy - sh / 2.0
            else:
                print(f'Warning: unknown obstacle format, skipping: {obs}', file=sys.stderr)
                continue
            rect = mpatches.Rectangle(
                (ox, oy), w, h,
                facecolor='#888888', edgecolor='#444444', lw=1.0,
                alpha=0.7, zorder=2,
            )
            self.ax.add_patch(rect)

        # Paths (only when output was provided)
        for i, path in enumerate(self.paths):
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            pts = np.array([[s[0], s[1]] for s in path], dtype=float)
            if len(pts) >= 2:
                self.ax.plot(pts[:, 0], pts[:, 1],
                             color=color, lw=2.2, alpha=0.9, zorder=3,
                             label=f'Robot {i}')

        # Starts and goals from env config
        legend_handles = []
        for i, robot in enumerate(self.robots_cfg):
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            geom  = ROBOT_GEOMETRY.get(robot.get('type', ''), _DEFAULT_GEOM)
            start = robot.get('start', [])
            goal  = robot.get('goal',  [])
            plotted = False
            if len(start) >= 2:
                x, y = start[0], start[1]
                yaw = start[2] if len(start) > 2 else 0.0
                patch = self._make_robot_outline_patch(geom, x, y, yaw, color, is_start=True)
                self.ax.add_patch(patch)
                plotted = True
            if len(goal) >= 2:
                x, y = goal[0], goal[1]
                yaw = goal[2] if len(goal) > 2 else 0.0
                patch = self._make_robot_outline_patch(geom, x, y, yaw, color, is_start=False)
                self.ax.add_patch(patch)
                plotted = True
            if plotted and not self.paths:
                legend_handles.append(
                    mpatches.Patch(color=color, label=f'Robot {i}')
                )

        # Legend + title
        if self.paths:
            self.ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                           borderaxespad=0, fontsize=9)
        elif legend_handles:
            start_proxy = mpatches.Patch(facecolor='lightgray', edgecolor='gray',
                                         linestyle='--', lw=2.0, label='Start')
            goal_proxy  = mpatches.Patch(facecolor='lightgray', edgecolor='gray',
                                         linestyle='-',  lw=2.5, label='Goal')
            self.ax.legend(
                handles=legend_handles + [start_proxy, goal_proxy],
                loc='upper left', bbox_to_anchor=(1.01, 1),
                borderaxespad=0, fontsize=9,
            )

        if self.paths:
            title = 'Geometric Planner — Paths'
            if self.planning_time is not None:
                title += f'  (planning time: {self.planning_time:.4f}s)'
        else:
            title = 'Environment — starts (dashed outline) and goals (solid outline)'
        self.ax.set_title(title, fontsize=11)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _make_robot_outline_patch(self, geom, x, y, yaw, color, is_start):
        """Create a shaded outline patch for a start (dashed) or goal (solid) pose."""
        from matplotlib.transforms import Affine2D
        face_alpha = 0.18 if is_start else 0.28
        r, g, b, _ = mcolors.to_rgba(color)
        facecolor = (r, g, b, face_alpha)
        ls = '--' if is_start else '-'
        lw = 2.0 if is_start else 2.5
        if geom['shape'] == 'sphere':
            patch = mpatches.Circle(
                (x, y), geom['radius'],
                facecolor=facecolor, edgecolor=color, lw=lw,
                linestyle=ls, zorder=4,
            )
        else:  # box
            length, width = geom['length'], geom['width']
            patch = mpatches.FancyBboxPatch(
                (-length / 2, -width / 2), length, width,
                boxstyle='square,pad=0',
                facecolor=facecolor, edgecolor=color, lw=lw,
                linestyle=ls, zorder=4,
            )
            patch.set_transform(
                Affine2D().rotate(yaw).translate(x, y) + self.ax.transData
            )
        return patch

    def _make_robot_patch(self, geom, x, y, yaw, color):
        """Create a matplotlib patch matching the robot's FCL collision geometry."""
        if geom['shape'] == 'sphere':
            patch = mpatches.Circle(
                (x, y), geom['radius'],
                facecolor=color, edgecolor='black', lw=1.2,
                alpha=0.85, zorder=7,
            )
        else:  # box
            length, width = geom['length'], geom['width']
            patch = mpatches.FancyBboxPatch(
                (-length / 2, -width / 2), length, width,
                boxstyle='square,pad=0',
                facecolor=color, edgecolor='black', lw=1.2,
                alpha=0.85, zorder=7,
            )
            from matplotlib.transforms import Affine2D
            patch.set_transform(
                Affine2D().rotate(yaw).translate(x, y) + self.ax.transData
            )
        return patch

    def _animate(self):
        if self.total_steps == 0 or not self.paths:
            return

        dt_real = 1.0 / (ANIM_FPS * self.anim_speed)

        # Pre-resolve geometry for each robot
        robot_geoms = []
        for i, robot_cfg in enumerate(self.robots_cfg):
            robot_type = robot_cfg.get('type', '')
            robot_geoms.append(ROBOT_GEOMETRY.get(robot_type, _DEFAULT_GEOM))

        self.fig.suptitle('Animating paths — press any key to skip', fontsize=10, y=0.99)
        plt.draw()

        self._advance_flag = False
        patches = {}

        for step in range(self.total_steps):
            if not plt.fignum_exists(self.fig.number):
                return
            if self._advance_flag:
                break

            for i, path in enumerate(self.paths):
                color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
                state = path[min(step, len(path) - 1)]
                x, y = state[0], state[1]
                yaw = state[2] if len(state) > 2 else 0.0
                geom = robot_geoms[i] if i < len(robot_geoms) else _DEFAULT_GEOM

                if i in patches:
                    try:
                        patches[i].remove()
                    except Exception:
                        pass

                patch = self._make_robot_patch(geom, x, y, yaw, color)
                self.ax.add_patch(patch)
                patches[i] = patch

            plt.draw()
            plt.pause(dt_real)

        for p in patches.values():
            try:
                p.remove()
            except Exception:
                pass

    def _save_mp4(self, path):
        from matplotlib.animation import FuncAnimation, FFMpegWriter

        robot_geoms = [
            ROBOT_GEOMETRY.get(rc.get('type', ''), _DEFAULT_GEOM)
            for rc in self.robots_cfg
        ]
        active_patches = {}

        def update(step):
            for i, path in enumerate(self.paths):
                color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
                state = path[min(step, len(path) - 1)]
                x, y = state[0], state[1]
                yaw = state[2] if len(state) > 2 else 0.0
                geom = robot_geoms[i] if i < len(robot_geoms) else _DEFAULT_GEOM
                if i in active_patches:
                    try:
                        active_patches[i].remove()
                    except Exception:
                        pass
                patch = self._make_robot_patch(geom, x, y, yaw, color)
                self.ax.add_patch(patch)
                active_patches[i] = patch
            return list(active_patches.values())

        fps = ANIM_FPS * self.anim_speed
        anim = FuncAnimation(
            self.fig, update, frames=self.total_steps,
            interval=1000.0 / fps, blit=False,
        )
        print(f'Saving animation to {path} ...', flush=True)
        anim.save(path, writer=FFMpegWriter(fps=fps))
        print('Done.', flush=True)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        if self.paths:
            if self.save_mp4:
                self._save_mp4(self.save_mp4)
            else:
                self._animate()
                self.fig.suptitle('Done — close window to exit.', fontsize=10, y=0.99)
                plt.draw()
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Geometric planner output visualizer'
    )
    p.add_argument('env',    help='Path to environment YAML (e.g. straight.yaml)')
    p.add_argument('output', nargs='?', default=None,
                   help='Path to planner output YAML (optional; omit to show env only)')
    p.add_argument('--anim-speed', type=float, default=1.0,
                   help='Animation playback speed multiplier (default: 1.0)')
    p.add_argument('--save-mp4', metavar='FILE', default=None,
                   help='Save animation to FILE (e.g. out.mp4) instead of showing interactively')
    return p.parse_args()


def main():
    args = parse_args()
    viz = GeomVisualizer(args.env, args.output, anim_speed=args.anim_speed, save_mp4=args.save_mp4)
    viz.run()


if __name__ == '__main__':
    main()
