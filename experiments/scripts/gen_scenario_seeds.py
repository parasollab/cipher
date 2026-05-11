import argparse
import copy
from pathlib import Path

import numpy as np
import yaml


# Geometry for robot types not in dynobench models directory
_GEOMETRY_FALLBACK = {
    'car_first_order_0': {'shape': 'box', 'size': [0.5, 0.25]},
}


def _load_geometry(robot_type, models_dir):
    yaml_path = models_dir / f"{robot_type}.yaml"
    if yaml_path.exists():
        cfg = yaml.safe_load(open(yaml_path))
    elif robot_type in _GEOMETRY_FALLBACK:
        cfg = _GEOMETRY_FALLBACK[robot_type]
    else:
        raise ValueError(f"No geometry info for robot type '{robot_type}'")
    return cfg['shape'], cfg.get('radius'), cfg.get('size')


def _bounding_radius(shape, radius, size):
    if shape == 'sphere':
        return radius
    return np.sqrt((size[0] / 2) ** 2 + (size[1] / 2) ** 2)


def _sphere_vs_box(px, py, r, cx, cy, sx, sy):
    nearest_x = max(cx - sx / 2, min(px, cx + sx / 2))
    nearest_y = max(cy - sy / 2, min(py, cy + sy / 2))
    return (px - nearest_x) ** 2 + (py - nearest_y) ** 2 < r ** 2


def _obb_vs_box(px, py, theta, rw, rh, cx, cy, sx, sy):
    """OBB-AABB separating axis test (4 axes)."""
    dx = px - cx
    dy = py - cy
    c = np.cos(theta)
    s = np.sin(theta)
    if abs(dx) >= sx / 2 + abs(c) * rw / 2 + abs(s) * rh / 2:
        return False
    if abs(dy) >= sy / 2 + abs(s) * rw / 2 + abs(c) * rh / 2:
        return False
    if abs(c * dx + s * dy) >= rw / 2 + abs(c) * sx / 2 + abs(s) * sy / 2:
        return False
    if abs(-s * dx + c * dy) >= rh / 2 + abs(s) * sx / 2 + abs(c) * sy / 2:
        return False
    return True


def _collides(px, py, theta, shape, radius, size, obstacles):
    for obs in obstacles:
        cx, cy = obs['center']
        sx, sy = obs['size']
        if shape == 'sphere':
            if _sphere_vs_box(px, py, radius, cx, cy, sx, sy):
                return True
        else:
            if _obb_vs_box(px, py, theta, size[0], size[1], cx, cy, sx, sy):
                return True
    return False


def _build_state(robot_type, x, y, theta):
    if robot_type in ('unicycle_first_order_0_sphere', 'unicycle_first_order_0', 'car_first_order_0'):
        return [x, y, theta]
    elif robot_type == 'unicycle_second_order_0':
        return [x, y, theta, 0.0]
    elif robot_type == 'double_integrator_0':
        return [x, y, 0.0, 0.0]
    else:
        raise ValueError(f"Unknown state format for robot type '{robot_type}'")


def _point_to_box_dist(px, py, cx, cy, sx, sy):
    dx = max(abs(px - cx) - sx / 2, 0.0)
    dy = max(abs(py - cy) - sy / 2, 0.0)
    return np.sqrt(dx * dx + dy * dy)


def _min_obs_dist(px, py, obstacles):
    if not obstacles:
        return np.inf
    return min(
        _point_to_box_dist(px, py, o['center'][0], o['center'][1], o['size'][0], o['size'][1])
        for o in obstacles
    )


def _ccw(ax, ay, bx, by, cx, cy):
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)


def _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    return (
        _ccw(ax, ay, cx, cy, dx, dy) != _ccw(bx, by, cx, cy, dx, dy) and
        _ccw(ax, ay, bx, by, cx, cy) != _ccw(ax, ay, bx, by, dx, dy)
    )


def _crossing_fraction(starts, goals):
    n = len(starts)
    n_pairs = n * (n - 1) // 2
    if n_pairs == 0:
        return 0.0
    crossings = sum(
        _segments_intersect(
            starts[i][0], starts[i][1], goals[i][0], goals[i][1],
            starts[j][0], starts[j][1], goals[j][0], goals[j][1],
        )
        for i in range(n) for j in range(i + 1, n)
    )
    return crossings / n_pairs


def _cell(x, y, cell_size):
    return (int(x / cell_size), int(y / cell_size))


def _sample(rng, env_min, env_max, margin):
    x = rng.uniform(env_min[0] + margin, env_max[0] - margin)
    y = rng.uniform(env_min[1] + margin, env_max[1] - margin)
    theta = rng.uniform(-np.pi, np.pi)
    return x, y, theta


def _generate(robots, obstacles, env_min, env_max, min_robot_dist, min_sg_dist, max_attempts, rng, models_dir, min_obs_clearance=0.0, min_cross_dist=0.0, min_boundary_margin=0.0, cell_size=5.0):
    geometries = [
        (lambda g: (g[0], g[1], g[2], _bounding_radius(*g)))(_load_geometry(r['type'], models_dir))
        for r in robots
    ]

    starts, goals = [], []

    for i, (robot, (shape, radius, size, br)) in enumerate(zip(robots, geometries)):
        margin = br + 0.05

        for _ in range(max_attempts):
            x, y, theta = _sample(rng, env_min, env_max, margin)
            if _collides(x, y, theta, shape, radius, size, obstacles):
                continue
            if (x < env_min[0] + min_boundary_margin or x > env_max[0] - min_boundary_margin or
                    y < env_min[1] + min_boundary_margin or y > env_max[1] - min_boundary_margin):
                continue
            if _min_obs_dist(x, y, obstacles) < min_obs_clearance:
                continue
            if any(np.hypot(x - sx, y - sy) < min_robot_dist for sx, sy, _ in starts):
                continue
            if any(np.hypot(x - gx, y - gy) < min_cross_dist for gx, gy, _ in goals):
                continue
            if any(_cell(x, y, cell_size) == _cell(gx, gy, cell_size) for gx, gy, _ in goals):
                continue
            starts.append((x, y, theta))
            break
        else:
            raise RuntimeError(f"Could not place start for robot {i} after {max_attempts} attempts")

        sx0, sy0, _ = starts[i]
        for _ in range(max_attempts):
            x, y, theta = _sample(rng, env_min, env_max, margin)
            if _collides(x, y, theta, shape, radius, size, obstacles):
                continue
            if (x < env_min[0] + min_boundary_margin or x > env_max[0] - min_boundary_margin or
                    y < env_min[1] + min_boundary_margin or y > env_max[1] - min_boundary_margin):
                continue
            if _min_obs_dist(x, y, obstacles) < min_obs_clearance:
                continue
            if np.hypot(x - sx0, y - sy0) < min_sg_dist:
                continue
            if any(np.hypot(x - gx, y - gy) < min_robot_dist for gx, gy, _ in goals):
                continue
            if any(np.hypot(x - sx, y - sy) < min_cross_dist for sx, sy, _ in starts[:-1]):
                continue
            if any(_cell(x, y, cell_size) == _cell(sx, sy, cell_size) for sx, sy, _ in starts[:-1]):
                continue
            goals.append((x, y, theta))
            break
        else:
            raise RuntimeError(f"Could not place goal for robot {i} after {max_attempts} attempts")

    return starts, goals


def _pick_types(n_robots, base_robots, robot_types_arg, rng):
    """Return a list of robot type strings, one per robot."""
    if robot_types_arg is not None:
        if len(robot_types_arg) == 1:
            return [robot_types_arg[0]] * n_robots
        return [str(rng.choice(robot_types_arg)) for _ in range(n_robots)]
    if base_robots:
        base_types = [r['type'] for r in base_robots]
        return [base_types[i % len(base_types)] for i in range(n_robots)]
    raise ValueError("No robot types available: specify --robot-types")


def _find_env_base(scenario_dir, scenario):
    """Find a base yaml in scenario_dir to borrow the environment block from.

    Priority: {scenario}.yaml (environment-only file) > any numbered base yaml.
    """
    env_only = scenario_dir / f'{scenario}.yaml'
    if env_only.exists():
        return env_only
    candidates = sorted(scenario_dir.glob(f'{scenario}[0-9]*.yaml'))
    candidates = [p for p in candidates if '_' not in p.stem]
    return candidates[0] if candidates else None


def gen_seed_file(base_cfg, n_robots, seed, models_dir, min_robot_dist, min_sg_dist, max_attempts, robot_types_arg=None, min_obs_clearance=0.0, min_cross_dist=0.0, max_crossing_fraction=1.0, max_global_attempts=100, min_boundary_margin=2.0, cell_size=5.0):
    env = base_cfg['environment']
    obstacles = env.get('obstacles', [])
    rng = np.random.default_rng(seed)

    types = _pick_types(n_robots, base_cfg.get('robots', []), robot_types_arg, rng)
    robots = [{'type': t} for t in types]

    for _ in range(max_global_attempts):
        starts, goals = _generate(
            robots, obstacles,
            env['min'], env['max'],
            min_robot_dist, min_sg_dist, max_attempts,
            rng, models_dir,
            min_obs_clearance=min_obs_clearance,
            min_cross_dist=min_cross_dist,
            min_boundary_margin=min_boundary_margin,
            cell_size=cell_size,
        )
        if _crossing_fraction(starts, goals) <= max_crossing_fraction:
            break
    else:
        print(f"  Warning: seed {seed} ({n_robots} robots) could not meet crossing fraction cap "
              f"after {max_global_attempts} attempts; using last result")

    new_cfg = copy.deepcopy(base_cfg)
    new_cfg['robots'] = [
        {
            'type': types[i],
            'start': _build_state(types[i], *starts[i]),
            'goal': _build_state(types[i], *goals[i]),
        }
        for i in range(n_robots)
    ]
    return new_cfg


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    models_dir = project_root / 'db-CBS' / 'dynoplan' / 'dynobench' / 'models'

    parser = argparse.ArgumentParser(description="Pre-generate per-seed start/goal configs for experiment scenarios.")
    parser.add_argument('--scenarios', nargs='+', required=True)
    parser.add_argument('--robots', nargs='+', type=int, default=[2, 4, 8, 16])
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--robot-types', nargs='+', default=None,
                        help="Robot type(s) to use. One type: all robots use it. "
                             "Multiple types: each robot is randomly assigned one. "
                             "Default: preserve types from base yaml.")
    parser.add_argument('--min-robot-dist', type=float, default=3.0,
                        help="Minimum distance between any two robot positions (default: 3.0)")
    parser.add_argument('--min-obs-clearance', type=float, default=1.0,
                        help="Minimum distance from robot center to nearest obstacle surface (default: 1.0)")
    parser.add_argument('--min-cross-dist', type=float, default=2.0,
                        help="Minimum distance between any start and any other robot's goal (default: 2.0)")
    parser.add_argument('--max-crossing-fraction', type=float, default=0.35,
                        help="Reject seeds where more than this fraction of path pairs cross (default: 0.35)")
    parser.add_argument('--min-boundary-margin', type=float, default=2.0,
                        help="Minimum distance from robot center to environment boundary (default: 2.0)")
    parser.add_argument('--cell-size', type=float, default=5.0,
                        help="Planner grid cell size used for cross-cell checks (default: 5.0)")
    parser.add_argument('--min-sg-dist', type=float, default=3.0,
                        help="Minimum distance between a robot's start and goal (default: 3.0)")
    parser.add_argument('--max-attempts', type=int, default=10000,
                        help="Max rejection-sampling attempts per robot (default: 10000)")
    args = parser.parse_args()

    for scenario in args.scenarios:
        scenario_dir = project_root / 'experiments' / scenario
        for n in args.robots:
            base_path = scenario_dir / f'{scenario}{n}.yaml'
            if base_path.exists():
                base_cfg = yaml.safe_load(open(base_path))
            else:
                fallback = _find_env_base(scenario_dir, scenario)
                if fallback is None:
                    print(f"No base yaml found in {scenario_dir}, skipping {n} robots")
                    continue
                base_cfg = yaml.safe_load(open(fallback))
                if args.robot_types is None and not base_cfg.get('robots'):
                    print(f"Warning: {fallback.name} has no robots and --robot-types not set; skipping {n} robots")
                    continue
                print(f"Using environment from {fallback.name}")

            out_dir = scenario_dir / 'seeds' / str(n)
            out_dir.mkdir(parents=True, exist_ok=True)

            for seed in range(1, args.seeds + 1):
                out_path = out_dir / f'{seed}.yaml'
                new_cfg = gen_seed_file(
                    base_cfg, n, seed, models_dir,
                    args.min_robot_dist, args.min_sg_dist, args.max_attempts,
                    robot_types_arg=args.robot_types,
                    min_obs_clearance=args.min_obs_clearance,
                    min_cross_dist=args.min_cross_dist,
                    max_crossing_fraction=args.max_crossing_fraction,
                    min_boundary_margin=args.min_boundary_margin,
                    cell_size=args.cell_size,
                )
                with open(out_path, 'w') as f:
                    yaml.dump(new_cfg, f, default_flow_style=None)
            print(f"Generated {scenario} seeds 1-{args.seeds} for {n} robots → seeds/{n}/")


if __name__ == '__main__':
    main()
