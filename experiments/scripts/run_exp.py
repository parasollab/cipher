import subprocess
from pathlib import Path
import yaml
import numpy as np

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

def run_planner(executable, problem_file, output_file, config_file, timeout, seed, out_config_dir):
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
    ]

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
        proc = subprocess.run(
            cmd,
            timeout=timeout + 30,  # Extra buffer for cleanup
            capture_output=True,
            text=True
        )

        # Parse output file
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                output = yaml.safe_load(f)
                result['solved'] = output.get('solved', output.get('success', False))
                result['planning_time'] = output.get('planning_time', timeout)
                if result['solved']:
                    result['makespan'] = compute_makespan(output)
                    result['sum_of_costs'] = compute_sum_of_costs(output)
        else:
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
    for robots in num_robots:
        success_count = 0
        for seed in range(1, num_seeds + 1):
            output_full_path = f"{output_file}/{robots}/{seed}.yaml"
            result = run_planner(executable, problem_file + str(robots) + '.yaml', output_full_path, config_file, timeout, seed, f"{base_output_dir}/experiments/configs/{method}/{robots}")
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

def main():
    # Get the directory of the current script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    overwrite_results = False

    scenarios = [
        # "narrow",
        # "narrow_kino",
        # "open",
        # "rooms",
        # "rooms_kino",
        # "cross",
        "cross_kino",
        # "low_clutter",
        # "medium_clutter",
        # "high_clutter"
    ]

    methods = [
    # {
    #     'name': 'coupled_rrt',
    #     'executable': str(project_root / 'build' / 'geometric_coupled_rrt'),
    # }, 
    # {
    #     'name': 'decoupled_rrt',
    #     'executable': str(project_root / 'build' / 'geometric_decoupled_rrt'),
    # },
    # {
    #     'name': 'srrt',
    #     'executable': str(project_root / 'build' / 'srrt'),
    # },
    # {
    #     'name': 'drrt',
    #     'executable': str(project_root / 'build' / 'drrt'),
    # },
    # {
    #     'name': 'arc',
    #     'executable': str(project_root / 'build' / 'arc'),
    # },
    {
        'name': 'kino_coupled_rrt',
        'executable': str(project_root / 'build' / 'kinodynamic_coupled_rrt'),
    },
    {
        'name': 'kino_decoupled_rrt',
        'executable': str(project_root / 'build' / 'kinodynamic_decoupled_rrt'),
    }
    ]

    summary_file = project_root / 'experiments' / 'results' / 'kino_summary.csv'

    # Write header to summary file if it doesn't exist or if we're overwriting results
    if summary_file.exists() and not overwrite_results:
        print(f"Summary file {summary_file} already exists. Appending results.")
    else:
        with open(summary_file, 'w') as f:
            f.write("method,scenario,robots,seed,solved,planning_time,timed_out,makespan,sum_of_costs\n")

    for scenario in scenarios:
        for method in methods:
            run_method(
                method['name'],
                scenario,
                method['executable'],
                str(project_root / 'experiments' / scenario / scenario),
                str(project_root / 'experiments' / 'results' / scenario / f'{method["name"]}'),
                str(project_root / 'examples' / 'config' / f'{method["name"]}.yaml'),
                timeout=600,  # 10 minutes
                num_robots=[2, 4, 8, 16],
                num_seeds=10,
                summary_file=summary_file,
                base_output_dir=project_root
            )

if __name__ == "__main__":
    main()
