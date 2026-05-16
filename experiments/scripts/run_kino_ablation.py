"""
Ablation study for kinodynamic_cipher.

Sweeps over region_size, robot_cell_size_ratio, and mapf_method (CBS vs A*).
Results are saved to experiments/results/kino_ablation.csv with columns:
  region_size, robot_cell_size_ratio, mapf_method, robots, seed,
  solved, planning_time, timed_out, makespan, sum_of_costs, ...

Runs independent (config × robots × seed) trials in parallel.
"""

import argparse
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

EXECUTABLE = 'kinodynamic_cipher'
SCENARIO = 'open_70x70'

_active_procs: set = set()
_procs_lock = threading.Lock()
_shutdown = threading.Event()

STATS_KEYS = [
    'total_conflicts_encountered', 'total_conflicts_resolved',
    'decomp_refine_attempts', 'decomp_refine_successes',
    'expansion_attempts', 'expansion_successes',
    'composite_attempts', 'composite_successes',
    'decoupled_attempts', 'decoupled_successes',
    'time_mapf', 'time_guided', 'time_decomp', 'time_conflict_resolution',
]


def compute_makespan(output):
    longest = 0
    for path_node in output.get('result', []):
        path = path_node.get('states') or []
        length = sum(
            np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
            for i in range(1, len(path))
        )
        longest = max(longest, length)
    return longest


def compute_sum_of_costs(output):
    total = 0
    for path_node in output.get('result', []):
        path = path_node.get('states') or []
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
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with _procs_lock:
            _active_procs.add(proc)
        try:
            proc.wait(timeout=timeout + 30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        finally:
            with _procs_lock:
                _active_procs.discard(proc)
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


def _make_row(region_size, ratio, mapf_method, robots, seed, result):
    return ','.join([
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


def write_result(summary_file, lock, region_size, ratio, mapf_method, robots, seed, result):
    key = f"{region_size},{ratio},{mapf_method},{robots},{seed},"
    row = _make_row(region_size, ratio, mapf_method, robots, seed, result)
    with lock:
        with open(summary_file) as f:
            lines = f.readlines()
        lines = [l for l in lines if not l.startswith(key)]
        lines.append(row)
        with open(summary_file, 'w') as f:
            f.writelines(lines)


def run_trial(args_tuple):
    """Entry point for each parallel trial."""
    (executable, problem_file, output_file, config, timeout, seed,
     config_dir, region_size, ratio, mapf_method, robots) = args_tuple
    if _shutdown.is_set():
        return region_size, ratio, mapf_method, robots, seed, None
    result = run_planner(executable, problem_file, output_file, config, timeout, seed, config_dir)
    return region_size, ratio, mapf_method, robots, seed, result


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(
        description="Ablation study for kinodynamic_cipher.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--region-sizes', nargs='+', type=float, default=[1, 2, 3, 5],
                        metavar='RS', help="Values of region_size to sweep")
    parser.add_argument('--ratios', nargs='+', type=float, default=[1.0, 1.3, 1.6, 2.0],
                        metavar='R', help="Values of robot_cell_size_ratio to sweep")
    parser.add_argument('--mapf-methods', nargs='+', default=['cbs', 'astar'],
                        choices=['cbs', 'astar'], metavar='M', help="MAPF methods to compare")
    parser.add_argument('--robots', nargs='+', type=int, default=None,
                        help="Robot counts (default: all discovered from seeds/)")
    parser.add_argument('--seeds', type=int, default=10, help="Seeds per configuration")
    parser.add_argument('--timeout', type=float, default=120.0, help="Planner timeout (s)")
    parser.add_argument('--workers', type=int, default=4,
                        help="Number of parallel planner processes")
    parser.add_argument('--output', default='kino_ablation.csv',
                        help="Output CSV in experiments/results/")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing CSV")
    args = parser.parse_args()

    scenario_dir = project_root / 'experiments' / SCENARIO
    executable = str(project_root / 'build' / EXECUTABLE)
    base_config_path = project_root / 'examples' / 'config' / 'kinodynamic_cipher.yaml'
    summary_file = project_root / 'experiments' / 'results' / args.output

    if args.robots is not None:
        robot_counts = args.robots
    else:
        seeds_dir = scenario_dir / 'seeds'
        robot_counts = sorted(
            int(d.name) for d in seeds_dir.iterdir() if d.is_dir() and d.name.isdigit()
        )
    print(f"Robot counts:  {robot_counts}")
    print(f"region_sizes:  {args.region_sizes}")
    print(f"ratios:        {args.ratios}")
    print(f"mapf_methods:  {args.mapf_methods}")
    print(f"Seeds: {args.seeds}, Timeout: {args.timeout}s, Workers: {args.workers}")

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

    # Build all independent trials upfront
    trials = []
    for region_size in args.region_sizes:
        for ratio in args.ratios:
            for mapf_method in args.mapf_methods:
                label = f"rs{int(region_size)}_ratio{ratio}_{mapf_method}"
                config = {**base_config, 'region_size': int(region_size),
                          'robot_cell_size_ratio': ratio, 'mapf_method': mapf_method}
                for robots in robot_counts:
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
                        trials.append((
                            executable, str(problem_file), output_file, config.copy(),
                            args.timeout, seed, str(config_dir),
                            region_size, ratio, mapf_method, robots,
                        ))

    total = len(trials)
    print(f"\nTotal trials: {total}  (parallelism: {args.workers})")

    lock = threading.Lock()
    completed = 0

    def _handle_interrupt(sig, frame):
        print("\nInterrupt received — killing running planners...")
        _shutdown.set()
        with _procs_lock:
            for proc in _active_procs:
                proc.kill()

    signal.signal(signal.SIGINT, _handle_interrupt)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_trial, t): t for t in trials}
        for future in as_completed(futures):
            completed += 1
            region_size, ratio, mapf_method, robots, seed, result = future.result()
            if result is None:
                continue
            status = 'OK' if result['solved'] else ('TIMEOUT' if result['timed_out'] else 'FAIL')
            label = f"rs{region_size}_ratio{ratio}_{mapf_method}"
            print(f"  [{completed}/{total}] {label} robots={robots:2d} seed={seed}: "
                  f"{status}  t={result['planning_time']:.2f}s")
            write_result(summary_file, lock, region_size, ratio, mapf_method, robots, seed, result)

    if _shutdown.is_set():
        print("\nAborted. Partial results saved to {summary_file}")
    else:
        print(f"\nDone. Results saved to {summary_file}")


if __name__ == '__main__':
    main()
