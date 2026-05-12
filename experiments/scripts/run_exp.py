import argparse
import os
import subprocess
import time
from pathlib import Path
import yaml
import numpy as np

METHOD_EXECUTABLES = {
    'geometric_cipher':   'geometric_cipher',
    'coupled_rrt':        'geometric_coupled_rrt',
    'decoupled_rrt':      'geometric_decoupled_rrt',
    'srrt':               'srrt',
    'drrt':               'drrt',
    'arc':                'arc',
    'kino_coupled_rrt':   'kinodynamic_coupled_rrt',
    'kino_decoupled_rrt': 'kinodynamic_decoupled_rrt',
    'kcbs':               'db-CBS/main_kcbs',
    'db_cbs':             'db-CBS/db_cbs',
    'k_arc':              'k_arc',
    'kinodynamic_cipher': 'kinodynamic_cipher',
}

def get_extra_args(method, output_full_path, timeout):
    if method == 'kcbs':
        base = Path(output_full_path)
        stats_path = base.parent / (base.stem + '_stats.yaml')
        return ['--timelimit', str(int(timeout)), '--stats', str(stats_path)]
    elif method == 'db_cbs':
        base = Path(output_full_path)
        return [
            '--joint', str(base.parent / (base.stem + '_joint.yaml')),
            '--optimization', str(base.parent / (base.stem + '_opt.yaml')),
        ]
    return []

def get_env(method):
    env = os.environ.copy()
    if method in ('k_arc', 'kinodynamic_cipher'):
        lib_path = '/home/courtney/cipher/install/lib'
        existing = env.get('LD_LIBRARY_PATH', '')
        env['LD_LIBRARY_PATH'] = f"{lib_path}:{existing}" if existing else lib_path
    return env

def compute_makespan(output):
    # load the yaml plan file and compute the makespan
    paths = output.get('result', [])
    longest = 0
    for path_node in paths:
        path = path_node.get('states')
        prev_state = None
        path_length = 0
        for state in path:
            if prev_state is not None:
                path_length += np.linalg.norm(np.array(state) - np.array(prev_state))
            prev_state = state
        longest = max(longest, path_length)

    return longest

def compute_sum_of_costs(output):
    # load the yaml plan file and compute the sum of costs
    paths = output.get('result', [])
    total_cost = 0
    for path_node in paths:
        path = path_node.get('states')
        prev_state = None
        path_length = 0
        for state in path:
            if prev_state is not None:
                path_length += np.linalg.norm(np.array(state) - np.array(prev_state))
            prev_state = state
        total_cost += path_length

    return total_cost

def run_planner(executable, problem_file, output_file, config_file, timeout, seed, out_config_dir, extra_args=[], env=None):
    """
    Run a planner executable and return the result.

    Returns:
        dict with keys: solved, planning_time, timed_out, error
    """
    # Make a new config file with the seed and write it to the out_config_dir
    init_config = yaml.safe_load(open(config_file))
    init_config['seed'] = seed
    out_config_file = Path(out_config_dir) / f'config_{seed}.yaml'

    print(f"Running planner with seed {seed} and config {out_config_file}...")
    
    # Ensure the config output directory exists
    out_config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_config_file, 'w') as f:
        yaml.dump(init_config, f)

    # Ensure the output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        executable,
        '-i', str(problem_file),
        '-o', str(output_file),
        '-c', str(out_config_file),
    ] + extra_args

    print(f"Executing command: {' '.join(cmd)}")

    result = {
        'solved': False,
        'planning_time': timeout,
        'timed_out': False,
        'error': None,
        'makespan': None,
        'sum_of_costs': None
    }

    try:
        start = time.time()
        proc = subprocess.run(
            cmd,
            timeout=timeout + 30,  # Extra buffer for cleanup
            capture_output=True,
            text=True,
            env=env
        )
        elapsed = time.time() - start

        # Parse output file
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                output = yaml.safe_load(f)
            solved = output.get('solved', output.get('success', None))
            if solved is None:  # kcbs/db_cbs omit this field; infer from result presence
                solved = bool(output.get('result'))
            result['solved'] = solved
            pt = output.get('planning_time')
            if pt is not None:
                result['planning_time'] = pt
            else:
                stats_path = Path(output_file).parent / (Path(output_file).stem + '_stats.yaml')
                if stats_path.exists():
                    with open(stats_path) as sf:
                        stats_data = yaml.safe_load(sf)
                    entries = (stats_data or {}).get('stats') or []
                    result['planning_time'] = entries[-1]['t'] if entries else elapsed
                else:
                    result['planning_time'] = elapsed
            if result['solved']:
                result['makespan'] = compute_makespan(output)
                result['sum_of_costs'] = compute_sum_of_costs(output)
        else:
            result['planning_time'] = elapsed
            result['error'] = 'No output file generated'

    except subprocess.TimeoutExpired:
        result['timed_out'] = True
        result['error'] = 'Process timed out'
    except Exception as e:
        result['error'] = str(e)

    return result

def run_method(method, scenario, executable, problem_file, output_file, config_file, timeout, num_robots, num_seeds, summary_file, base_output_dir=None):
    """
    Run the planner for a range of robot counts and print results.
    Stops early if the planner fails to solve all instances for a given robot count.
    """
    env = get_env(method)
    for robots in num_robots:
        success_count = 0
        for seed in range(1, num_seeds + 1):
            output_full_path = f"{output_file}/{robots}/{seed}.yaml"
            extra_args = get_extra_args(method, output_full_path, timeout)
            scenario_dir = Path(problem_file).parent
            seed_problem = scenario_dir / 'seeds' / str(robots) / f'{seed}.yaml'
            base_problem = Path(problem_file + str(robots) + '.yaml')
            chosen_problem = seed_problem if seed_problem.exists() else base_problem
            result = run_planner(executable, str(chosen_problem), output_full_path, config_file, timeout, seed, f"{base_output_dir}/experiments/configs/{method}/{robots}", extra_args=extra_args, env=env)
            print(f"\tResult: {result['solved']}, Time: {result['planning_time']:.2f}s, Timed out: {result['timed_out']}, Error: {result['error']}")

            if result['solved']:
                success_count += 1

            # Write results to summary file, overwriting any existing entry for this key
            new_line = f"{method},{scenario},{robots},{seed},{result['solved']},{result['planning_time']:.5f},{result['timed_out']},{result['makespan']},{result['sum_of_costs']}\n"
            with open(summary_file, 'r') as f:
                lines = f.readlines()
            key_prefix = f"{method},{scenario},{robots},{seed},"
            lines = [l for l in lines if not l.startswith(key_prefix)]
            lines.append(new_line)
            with open(summary_file, 'w') as f:
                f.writelines(lines)

        print(f"Successfully solved {success_count}/{num_seeds} instances for {robots} robots.")

        if success_count == 0:
            print(f"Stopping early for {robots} robots due to zero success rate.")
            break

def discover_robot_counts(scenario_dir, scenario):
    """Return sorted list of robot counts that have seed files or base yamls."""
    counts = set()
    seeds_dir = scenario_dir / 'seeds'
    if seeds_dir.exists():
        for d in seeds_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                counts.add(int(d.name))
    for p in scenario_dir.glob(f'{scenario}[0-9]*.yaml'):
        if '_' not in p.stem:
            suffix = p.stem[len(scenario):]
            if suffix.isdigit():
                counts.add(int(suffix))
    return sorted(counts)


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(description="Run multi-robot planning experiments.")
    parser.add_argument('--scenarios', nargs='+', required=True, help="Scenario names")
    parser.add_argument('--methods', nargs='+', required=True, choices=list(METHOD_EXECUTABLES.keys()), help="Method names (k_arc requires install/lib on LD_LIBRARY_PATH, handled automatically)")
    parser.add_argument('--output', default='summary.csv', help="Output CSV filename in experiments/results/ (default: summary.csv)")
    parser.add_argument('--robots', nargs='+', type=int, default=None, help="Robot counts to run (default: all counts discovered from seeds/ and base yamls)")
    parser.add_argument('--seeds', type=int, default=10, help="Number of seeds per configuration (default: 10)")
    parser.add_argument('--timeout', type=float, default=600.0, help="Planner timeout in seconds (default: 600)")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite the output file instead of appending")
    args = parser.parse_args()

    methods = [{'name': m, 'executable': str(project_root / 'build' / METHOD_EXECUTABLES[m])} for m in args.methods]

    summary_file = project_root / 'experiments' / 'results' / args.output

    if summary_file.exists() and not args.overwrite:
        print(f"Summary file {summary_file} already exists. Appending results.")
    else:
        with open(summary_file, 'w') as f:
            f.write("method,scenario,robots,seed,solved,planning_time,timed_out,makespan,sum_of_costs\n")

    for scenario in args.scenarios:
        if args.robots is not None:
            robot_counts = args.robots
        else:
            scenario_dir = project_root / 'experiments' / scenario
            robot_counts = discover_robot_counts(scenario_dir, scenario)
            if not robot_counts:
                print(f"No robot counts found for scenario '{scenario}', skipping.")
                continue
            print(f"Discovered robot counts for '{scenario}': {robot_counts}")
        for method in methods:
            run_method(
                method['name'],
                scenario,
                method['executable'],
                str(project_root / 'experiments' / scenario / scenario),
                str(project_root / 'experiments' / 'results' / scenario / f'{method["name"]}'),
                str(project_root / 'examples' / 'config' / f'{method["name"]}.yaml'),
                timeout=args.timeout,
                num_robots=robot_counts,
                num_seeds=args.seeds,
                summary_file=summary_file,
                base_output_dir=project_root
            )

if __name__ == "__main__":
    main()
