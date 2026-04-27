import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

name_map = {
    'kino_coupled_rrt': 'Coupled RRT',
    'kino_decoupled_rrt': 'Decoupled RRT',
    'coupled_rrt': 'Coupled RRT',
    'decoupled_rrt': 'Decoupled RRT',
    'srrt': 'sRRT',
    'drrt': 'MRdRRT',
    'arc': 'ARC'
}

def plot_time(df, scenario, methods, num_robots, output_dir, file_name):
    # Filter the DataFrame for the given scenario, methods, and number of robots
    df_filtered = df[(df['scenario'] == scenario) & (df['method'].isin(methods)) & (df['robots'].isin(num_robots))]

    # Create a boxplot for each method
    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(x='robots', y='planning_time', hue='method', data=df_filtered, err_style="bars", marker='o')
    ax.set_title('Planning Time')
    ax.set_xlabel('Number of Robots')
    ax.set_ylabel('Time (s)')
    ax.set_yscale('log')  # Use logarithmic scale for better visibility of differences
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    labels = [name_map.get(l, l) for l in labels]
    ax.legend(handles, labels)

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, file_name))
    print(f"Saved plot to {os.path.join(output_dir, file_name)}")
    plt.close()

def plot_success_rate(df, scenario, methods, num_robots, output_dir, file_name):
    # Filter the DataFrame for the given scenario, methods, and number of robots
    df_filtered = df[(df['scenario'] == scenario) & (df['method'].isin(methods)) & (df['robots'].isin(num_robots))]

    # Create a line plot for success rate
    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(x='robots', y='solved', hue='method', data=df_filtered, err_style=None, marker='o')
    ax.set_title('Success Rate')
    ax.set_xlabel('Number of Robots')
    ax.set_ylabel('Success Rate')
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    labels = [name_map.get(l, l) for l in labels]
    ax.legend(handles, labels)

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, file_name))
    print(f"Saved plot to {os.path.join(output_dir, file_name)}")
    plt.close()

def plot_makespan(df, scenario, methods, num_robots, output_dir, file_name):
    # Filter the DataFrame for the given scenario, methods, and number of robots
    df_filtered = df[(df['scenario'] == scenario) & (df['method'].isin(methods)) & (df['robots'].isin(num_robots))]

    # Create a line plot for makespan
    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(x='robots', y='makespan', hue='method', data=df_filtered, err_style="bars", marker='o')
    ax.set_title('Makespan')
    ax.set_xlabel('Number of Robots')
    ax.set_ylabel('Makespan (s)')
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    labels = [name_map.get(l, l) for l in labels]
    ax.legend(handles, labels)

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, file_name))
    print(f"Saved plot to {os.path.join(output_dir, file_name)}")
    plt.close()

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    kino = True

    summary_file = os.path.join(project_root, 'experiments', 'results', 'summary.csv' if not kino else 'kino_summary.csv')
    df = pd.read_csv(summary_file)

    num_robots = [2, 4, 8, 16]

    scenarios = [
        "narrow",
        "open",
        "rooms",
        "cross",
        "low_clutter",
        "medium_clutter",
        "high_clutter"
    ] if not kino else [
        "narrow",
        "open",
        "rooms",
        "cross",
        "low_clutter",
        "medium_clutter",
        "high_clutter",
        "open_second",
        "narrow_second",
        "rooms_second",
        "cross_second",
        "low_clutter_second",
        "medium_clutter_second",
        "high_clutter_second",
        "open_double",
        "narrow_double",
        "rooms_double",
        "cross_double",
        "low_clutter_double",
        "medium_clutter_double",
        "high_clutter_double"
    ]

    methods = [
        'coupled_rrt',
        'decoupled_rrt',
        'srrt',
        'drrt',
        'arc'
    ] if not kino else [
        'kino_coupled_rrt',
        'kino_decoupled_rrt',
        'kcbs'
    ]

    output_dir = os.path.join(project_root, 'experiments', 'plots')
    time_output_dir = os.path.join(output_dir, 'time')
    success_output_dir = os.path.join(output_dir, 'success_rate')
    makespan_output_dir = os.path.join(output_dir, 'makespan')
    os.makedirs(time_output_dir, exist_ok=True)
    os.makedirs(success_output_dir, exist_ok=True)
    os.makedirs(makespan_output_dir, exist_ok=True)
    for scenario in scenarios:
        plot_time(df, scenario, methods, num_robots, time_output_dir, f'{scenario}.png' if not kino else f'kino_{scenario}.png')
        plot_success_rate(df, scenario, methods, num_robots, success_output_dir, f'{scenario}.png' if not kino else f'kino_{scenario}.png')
        plot_makespan(df, scenario, methods, num_robots, makespan_output_dir, f'{scenario}.png' if not kino else f'kino_{scenario}.png')

if __name__ == "__main__":
    main()