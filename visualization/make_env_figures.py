"""
Generate publication-quality static environment figures.

Shows obstacle layout with colored start ('S') and goal ('G') positions for each robot.
No legend required — 'S' and 'G' labels are self-explanatory.

Usage:
    # By scenario name (looks up experiments/<scenario>/seeds/<robots>/<seed>.yaml)
    python make_env_figures.py --scenarios open rooms low_clutter medium_clutter high_clutter \
        --robots 16 --seed 1 --output-dir figs/

    # By explicit YAML path
    python make_env_figures.py path/to/seed.yaml --output-dir figs/
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml
from matplotlib.transforms import Affine2D

# 20 perceptually distinct colors; cycles if more than 20 robots
_CMAP = matplotlib.colormaps['tab20']
DISPLAY_COLORS = [_CMAP(i) for i in range(20)]

ROBOT_GEOMETRY = {
    'single_integrator_0':               {'shape': 'sphere', 'radius': 0.1},
    'double_integrator_0':               {'shape': 'sphere', 'radius': 0.15},
    'unicycle_first_order_0':            {'shape': 'box',    'length': 0.5, 'width': 0.25},
    'unicycle_first_order_0_sphere':     {'shape': 'sphere', 'radius': 0.4},
    'unicycle_second_order_0':           {'shape': 'box',    'length': 0.5, 'width': 0.25},
    'car_first_order_0':                 {'shape': 'box',    'length': 0.5, 'width': 0.25},
    'car_first_order_with_1_trailers_0': {'shape': 'box',    'length': 0.5, 'width': 0.25},
}
_DEFAULT_GEOM = {'shape': 'sphere', 'radius': 0.2}


def _scale_geom(geom, display_radius):
    """Return a copy of geom with dimensions replaced by display_radius."""
    if geom['shape'] == 'sphere':
        return {'shape': 'sphere', 'radius': display_radius}
    # Box: scale uniformly so the short dimension equals display_radius
    factor = display_radius / min(geom['length'], geom['width']) * 2
    return {
        'shape': 'box',
        'length': geom['length'] * factor,
        'width':  geom['width']  * factor,
    }


def _make_patch(geom, x, y, yaw, color, is_start, ax):
    """Circle or rotated box patch for a start (dashed) or goal (solid) position."""
    rgba = list(color[:3]) + [0.18 if is_start else 0.35]
    ls = '--' if is_start else '-'
    lw = 1.8 if is_start else 2.2
    if geom['shape'] == 'sphere':
        return mpatches.Circle(
            (x, y), geom['radius'],
            facecolor=rgba, edgecolor=color[:3], lw=lw, linestyle=ls, zorder=4,
        )
    # box
    length, width = geom['length'], geom['width']
    patch = mpatches.FancyBboxPatch(
        (-length / 2, -width / 2), length, width,
        boxstyle='square,pad=0',
        facecolor=rgba, edgecolor=color[:3], lw=lw, linestyle=ls, zorder=4,
    )
    patch.set_transform(Affine2D().rotate(yaw).translate(x, y) + ax.transData)
    return patch


def draw_env_figure(yaml_path, output_path, display_radius=1.5, figsize=(4, 4), fmt='pdf'):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    env = data['environment']
    env_min = np.array(env['min'], dtype=float)
    env_max = np.array(env['max'], dtype=float)
    obstacles = env.get('obstacles', []) or []
    robots = data.get('robots', [])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    span = env_max - env_min
    margin = span * 0.03
    ax.set_xlim(env_min[0] - margin[0], env_max[0] + margin[0])
    ax.set_ylim(env_min[1] - margin[1], env_max[1] + margin[1])

    # Workspace boundary
    bx = [env_min[0], env_max[0], env_max[0], env_min[0], env_min[0]]
    by = [env_min[1], env_min[1], env_max[1], env_max[1], env_min[1]]
    ax.plot(bx, by, color='#222222', lw=1.5, zorder=1)

    # Obstacles
    for obs in obstacles:
        if not isinstance(obs, dict):
            continue
        if 'min' in obs and 'max' in obs:
            omin, omax = obs['min'], obs['max']
            ox, oy = omin[0], omin[1]
            w, h = omax[0] - omin[0], omax[1] - omin[1]
        elif obs.get('type') == 'box' and 'center' in obs and 'size' in obs:
            cx, cy = obs['center'][0], obs['center'][1]
            w, h = obs['size'][0], obs['size'][1]
            ox, oy = cx - w / 2, cy - h / 2
        else:
            continue
        ax.add_patch(mpatches.Rectangle(
            (ox, oy), w, h,
            facecolor='#aaaaaa', edgecolor='#555555', lw=1.0, zorder=2,
        ))

    # Robots
    for i, robot in enumerate(robots):
        color = DISPLAY_COLORS[i % len(DISPLAY_COLORS)]
        raw_geom = ROBOT_GEOMETRY.get(robot.get('type', ''), _DEFAULT_GEOM)
        geom = _scale_geom(raw_geom, display_radius)

        start = robot.get('start', [])
        goal  = robot.get('goal',  [])

        if len(start) >= 2:
            sx, sy = start[0], start[1]
            syaw = start[2] if len(start) > 2 else 0.0
            ax.add_patch(_make_patch(geom, sx, sy, syaw, color, is_start=True, ax=ax))

        if len(goal) >= 2:
            gx, gy = goal[0], goal[1]
            gyaw = goal[2] if len(goal) > 2 else 0.0
            ax.add_patch(_make_patch(geom, gx, gy, gyaw, color, is_start=False, ax=ax))

    plt.tight_layout(pad=0.1)
    fig.savefig(output_path, format=fmt, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'Saved {output_path}')


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    p = argparse.ArgumentParser(description='Generate environment figures for paper')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('yamls', nargs='*', default=[],
                     help='Explicit YAML file paths (alternative to --scenarios)')
    src.add_argument('--scenarios', nargs='+',
                     help='Scenario names; looks up experiments/<s>/seeds/<robots>/<seed>.yaml')
    p.add_argument('--robots', type=int, default=16,
                   help='Robot count when using --scenarios (default: 16)')
    p.add_argument('--seed', type=int, default=1,
                   help='Seed number when using --scenarios (default: 1)')
    p.add_argument('--output-dir', default='.', help='Output directory (default: .)')
    p.add_argument('--format', choices=['pdf', 'png'], default='pdf',
                   help='Output format (default: pdf)')
    p.add_argument('--display-radius', type=float, default=1.5,
                   help='Display radius for robot markers in env units (default: 1.5)')
    p.add_argument('--figsize', type=float, nargs=2, default=[4, 4],
                   metavar=('W', 'H'), help='Figure size in inches (default: 4 4)')
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []  # list of (yaml_path, stem)

    if args.scenarios:
        exp_dir = project_root / 'experiments'
        for scenario in args.scenarios:
            yaml_path = exp_dir / scenario / 'seeds' / str(args.robots) / f'{args.seed}.yaml'
            if not yaml_path.exists():
                print(f'Warning: {yaml_path} not found, skipping.', file=sys.stderr)
                continue
            jobs.append((yaml_path, scenario))
    else:
        for path_str in args.yamls:
            p_path = Path(path_str)
            if not p_path.exists():
                print(f'Warning: {p_path} not found, skipping.', file=sys.stderr)
                continue
            jobs.append((p_path, p_path.stem))

    if not jobs:
        print('No valid inputs found.', file=sys.stderr)
        sys.exit(1)

    fmt = args.format
    for yaml_path, stem in jobs:
        output_path = out_dir / f'{stem}.{fmt}'
        draw_env_figure(
            yaml_path, output_path,
            display_radius=args.display_radius,
            figsize=tuple(args.figsize),
            fmt=fmt,
        )


if __name__ == '__main__':
    main()
