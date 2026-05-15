import argparse
import sys
from pathlib import Path
import yaml
import numpy as np


def compute_makespan(output):
    paths = output.get('result', [])
    longest = 0
    for path_node in paths:
        path = path_node.get('states') or []
        prev_state = None
        path_length = 0
        for state in path:
            if prev_state is not None:
                path_length += np.linalg.norm(np.array(state) - np.array(prev_state))
            prev_state = state
        longest = max(longest, path_length)
    return longest


def compute_sum_of_costs(output):
    paths = output.get('result', [])
    total_cost = 0
    for path_node in paths:
        path = path_node.get('states') or []
        prev_state = None
        path_length = 0
        for state in path:
            if prev_state is not None:
                path_length += np.linalg.norm(np.array(state) - np.array(prev_state))
            prev_state = state
        total_cost += path_length
    return total_cost


def gather_results(scenarios, methods, robots, seeds, output_file, timeout=600.0):
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / 'results'

    rows = []
    for scenario in scenarios:
        for method in methods:
            for num_robots in robots:
                for seed in range(1, seeds + 1):
                    yaml_path = results_dir / scenario / method / str(num_robots) / f'{seed}.yaml'
                    if not yaml_path.exists():
                        print(f"WARNING: missing {yaml_path}", file=sys.stderr)
                        continue

                    with open(yaml_path, 'r') as f:
                        output = yaml.safe_load(f)

                    solved = output.get('solved', output.get('success', None))
                    if solved is None:
                        solved = bool(output.get('result'))

                    planning_time = output.get('planning_time')
                    if planning_time is None:
                        stats_path = yaml_path.parent / (yaml_path.stem + '_stats.yaml')
                        if stats_path.exists():
                            with open(stats_path) as sf:
                                stats_data = yaml.safe_load(sf)
                            entries = (stats_data or {}).get('stats') or []
                            planning_time = entries[-1]['t'] if entries else timeout
                        else:
                            planning_time = timeout

                    timed_out = (not solved) and (planning_time >= timeout)

                    if solved:
                        makespan = compute_makespan(output)
                        sum_of_costs = compute_sum_of_costs(output)
                    else:
                        makespan = None
                        sum_of_costs = None

                    rows.append(f"{method},{scenario},{num_robots},{seed},{solved},{planning_time:.5f},{timed_out},{makespan},{sum_of_costs}\n")

    with open(output_file, 'w') as f:
        f.write("method,scenario,robots,seed,solved,planning_time,timed_out,makespan,sum_of_costs\n")
        f.writelines(rows)

    print(f"Wrote {len(rows)} rows to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate a summary CSV from existing result YAML files.")
    parser.add_argument('--scenarios', nargs='+', required=True, help="Scenario names")
    parser.add_argument('--methods', nargs='+', required=True, help="Method names")
    parser.add_argument('--robots', nargs='+', type=int, default=None, help="Robot counts (default: all that exist in results)")
    parser.add_argument('--seeds', type=int, default=10, help="Number of seeds per configuration (default: 10)")
    parser.add_argument('--output', required=True, help="Output CSV filename (written to experiments/results/)")
    parser.add_argument('--timeout', type=float, default=600.0, help="Timeout used during runs, for timed_out inference (default: 600)")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / 'results'
    output_file = results_dir / args.output

    if args.robots is None:
        robot_set = set()
        for scenario in args.scenarios:
            for method in args.methods:
                method_dir = results_dir / scenario / method
                if method_dir.is_dir():
                    for d in method_dir.iterdir():
                        if d.is_dir() and d.name.isdigit():
                            robot_set.add(int(d.name))
        robots = sorted(robot_set)
        if not robots:
            parser.error("No robot count directories found; specify --robots explicitly.")
    else:
        robots = args.robots

    gather_results(
        scenarios=args.scenarios,
        methods=args.methods,
        robots=robots,
        seeds=args.seeds,
        output_file=output_file,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
