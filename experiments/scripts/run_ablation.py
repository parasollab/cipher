"""
Ablation study for geometric_cipher on the boxes_70x70 environment.

Sweeps over region_size, robot_cell_size_ratio, and mapf_method (CBS vs A*).
Results are saved to experiments/results/ablation.csv with columns:
  region_size, robot_cell_size_ratio, mapf_method, robots, seed,
  solved, planning_time, timed_out, makespan, sum_of_costs
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

EXECUTABLE = 'geometric_cipher'
SCENARIO = 'boxes_70x70'


def compute_makespan(output):
    longest = 0
    for path_node in output.get('result', []):
        path = path_node.get('states', [])
        length = sum(
            np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
            for i in range(1, len(path))
        )
        longest = max(longest, length)
    return longest


def compute_sum_of_costs(output):
    total = 0
    for path_node in output.get('result', []):
        path = path_node.get('states', [])
        total += sum(
            np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
            for i in range(1, len(path))
        )
    return total


def run_planner(executable, problem_file, output_file, config_data, timeout, seed, config_dir):
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / f'config_{seed}.yaml'
    config_data['seed'] = seed
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    cmd = [executable, '-i', str(problem_file), '-o', str(output_file), '-c', str(config_file)]
    print(f"    cmd: {' '.join(cmd)}")

    STATS_KEYS = [
        'total_conflicts_encountered', 'total_conflicts_resolved',
        'decomp_refine_attempts', 'decomp_refine_successes',
        'expansion_attempts', 'expansion_successes',
        'composite_attempts', 'composite_successes',
        'decoupled_attempts', 'decoupled_successes',
        'time_mapf', 'time_guided', 'time_decomp', 'time_conflict_resolution',
    ]
    result = {
        'solved': False,
        'planning_time': timeout,
        'timed_out': False,
        'error': None,
        'makespan': None,
        'sum_of_costs': None,
        'failure_reason': None,
        **{k: None for k in STATS_KEYS},
    }

    try:
        start = time.time()
        subprocess.run(cmd, timeout=timeout + 30, capture_output=True, text=True)
        elapsed = time.time() - start

        if Path(output_file).exists():
            with open(output_file) as f:
                output = yaml.safe_load(f)
            solved = output.get('solved', output.get('success', bool(output.get('result'))))
            result['solved'] = bool(solved)
            pt = output.get('planning_time')
            result['planning_time'] = pt if pt is not None else elapsed
            result['failure_reason'] = output.get('failure_reason')
            if result['solved']:
                result['makespan'] = compute_makespan(output)
                result['sum_of_costs'] = compute_sum_of_costs(output)
            s = output.get('resolution_stats', {}) or {}
            result['total_conflicts_encountered'] = s.get('total_conflicts_encountered')
            result['total_conflicts_resolved']    = s.get('total_conflicts_resolved')
            result['decomp_refine_attempts']      = s.get('decomposition_refinement_attempts')
            result['decomp_refine_successes']     = s.get('decomposition_refinement_successes')
            result['expansion_attempts']          = s.get('expansion_attempts')
            result['expansion_successes']         = s.get('expansion_successes')
            result['composite_attempts']          = s.get('composite_planner_attempts')
            result['composite_successes']         = s.get('composite_planner_successes')
            result['decoupled_attempts']          = s.get('decoupled_planner_attempts')
            result['decoupled_successes']         = s.get('decoupled_planner_successes')
            result['time_mapf']                   = s.get('time_mapf_seconds')
            result['time_guided']                 = s.get('time_guided_planning_seconds')
            result['time_decomp']                 = s.get('time_decomposition_seconds')
            result['time_conflict_resolution']    = s.get('time_conflict_resolution_seconds')
        else:
            result['planning_time'] = elapsed
            result['error'] = 'No output file'
    except subprocess.TimeoutExpired:
        result['timed_out'] = True
        result['error'] = 'Timed out'
    except Exception as e:
        result['error'] = str(e)

    return result


def _fmt(v):
    if v is None:
        return ''
    if isinstance(v, float):
        return f'{v:.5f}'
    return str(v)


def write_result(summary_file, region_size, ratio, mapf_method, robots, seed, result):
    key = f"{region_size},{ratio},{mapf_method},{robots},{seed},"
    line = ','.join([
        str(region_size), str(ratio), mapf_method, str(robots), str(seed),
        str(result['solved']), _fmt(result['planning_time']),
        str(result['timed_out']), _fmt(result['makespan']), _fmt(result['sum_of_costs']),
        str(result['failure_reason'] or ''),
        _fmt(result['total_conflicts_encountered']),
        _fmt(result['total_conflicts_resolved']),
        _fmt(result['decomp_refine_attempts']),
        _fmt(result['decomp_refine_successes']),
        _fmt(result['expansion_attempts']),
        _fmt(result['expansion_successes']),
        _fmt(result['composite_attempts']),
        _fmt(result['composite_successes']),
        _fmt(result['decoupled_attempts']),
        _fmt(result['decoupled_successes']),
        _fmt(result['time_mapf']),
        _fmt(result['time_guided']),
        _fmt(result['time_decomp']),
        _fmt(result['time_conflict_resolution']),
    ]) + '\n'
    with open(summary_file) as f:
        lines = f.readlines()
    lines = [l for l in lines if not l.startswith(key)]
    lines.append(line)
    with open(summary_file, 'w') as f:
        f.writelines(lines)


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(
        description="Ablation study for geometric_cipher on boxes_70x70.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--region-sizes', nargs='+', type=float, default=[3, 5, 7, 10],
                        metavar='RS', help="Values of region_size to sweep")
    parser.add_argument('--ratios', nargs='+', type=float, default=[1.0, 1.3, 1.6, 2.0],
                        metavar='R', help="Values of robot_cell_size_ratio to sweep")
    parser.add_argument('--mapf-methods', nargs='+', default=['cbs', 'astar'],
                        choices=['cbs', 'astar'], metavar='M', help="MAPF methods to compare")
    parser.add_argument('--robots', nargs='+', type=int, default=None,
                        help="Robot counts (default: all discovered from seeds/)")
    parser.add_argument('--seeds', type=int, default=10, help="Seeds per configuration")
    parser.add_argument('--timeout', type=float, default=120.0, help="Planner timeout (s)")
    parser.add_argument('--output', default='ablation.csv', help="Output CSV in experiments/results/")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing CSV")
    args = parser.parse_args()

    scenario_dir = project_root / 'experiments' / SCENARIO
    executable = str(project_root / 'build' / EXECUTABLE)
    base_config_path = project_root / 'examples' / 'config' / 'geometric_cipher.yaml'
    summary_file = project_root / 'experiments' / 'results' / args.output

    if args.robots is not None:
        robot_counts = args.robots
    else:
        seeds_dir = scenario_dir / 'seeds'
        robot_counts = sorted(
            int(d.name) for d in seeds_dir.iterdir() if d.is_dir() and d.name.isdigit()
        )
    print(f"Robot counts: {robot_counts}")
    print(f"region_sizes: {args.region_sizes}")
    print(f"ratios:       {args.ratios}")
    print(f"mapf_methods: {args.mapf_methods}")
    print(f"Seeds: {args.seeds}, Timeout: {args.timeout}s")

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    if summary_file.exists() and not args.overwrite:
        print(f"\nAppending to {summary_file}")
    else:
        with open(summary_file, 'w') as f:
            f.write(
                "region_size,robot_cell_size_ratio,mapf_method,robots,seed,"
                "solved,planning_time,timed_out,makespan,sum_of_costs,"
                "failure_reason,"
                "total_conflicts_encountered,total_conflicts_resolved,"
                "decomp_refine_attempts,decomp_refine_successes,"
                "expansion_attempts,expansion_successes,"
                "composite_attempts,composite_successes,"
                "decoupled_attempts,decoupled_successes,"
                "time_mapf,time_guided,time_decomp,time_conflict_resolution\n"
            )
        print(f"\nCreated {summary_file}")

    base_config = yaml.safe_load(open(base_config_path))

    total = len(args.region_sizes) * len(args.ratios) * len(args.mapf_methods)
    idx = 0
    for region_size in args.region_sizes:
        for ratio in args.ratios:
            for mapf_method in args.mapf_methods:
                idx += 1
                label = f"rs{int(region_size)}_ratio{ratio}_{mapf_method}"
                print(f"\n[{idx}/{total}] {label}")

                config = {**base_config, 'region_size': int(region_size),
                          'robot_cell_size_ratio': ratio, 'mapf_method': mapf_method}

                for robots in robot_counts:
                    success_count = 0
                    for seed in range(1, args.seeds + 1):
                        output_file = str(
                            project_root / 'experiments' / 'results' / SCENARIO / 'ablation'
                            / label / str(robots) / f'{seed}.yaml'
                        )
                        problem_file = scenario_dir / 'seeds' / str(robots) / f'{seed}.yaml'
                        if not problem_file.exists():
                            problem_file = scenario_dir / f'{SCENARIO}{robots}.yaml'
                        config_dir = (
                            project_root / 'experiments' / 'configs' / 'ablation'
                            / label / str(robots)
                        )

                        result = run_planner(
                            executable, str(problem_file), output_file,
                            config.copy(), args.timeout, seed, config_dir,
                        )
                        status = 'OK' if result['solved'] else ('TIMEOUT' if result['timed_out'] else 'FAIL')
                        print(f"  robots={robots:2d} seed={seed:2d}: {status}  t={result['planning_time']:.2f}s")
                        if result['solved']:
                            success_count += 1

                        write_result(summary_file, region_size, ratio, mapf_method, robots, seed, result)

                    print(f"  -> robots={robots}: {success_count}/{args.seeds} solved")
                    if success_count == 0:
                        print(f"  Early stop for {label} at {robots} robots.")
                        break

    print(f"\nDone. Results saved to {summary_file}")


if __name__ == '__main__':
    main()
