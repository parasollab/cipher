import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import pandas as pd
import os

name_map = {
    'kino_coupled_rrt': 'Coupled RRT',
    'kino_decoupled_rrt': 'Decoupled RRT',
    'coupled_rrt': 'Coupled RRT',
    'decoupled_rrt': 'Decoupled RRT',
    'srrt': 'sRRT',
    'drrt': 'MRdRRT',
    'arc': 'ARC',
    'kcbs': 'KCBS',
    'k_arc': 'K-ARC'
}

CONFIGS = [
    {
        'label': 'geometric',
        'summary_file': 'summary.csv',
        'scenarios': [
            'narrow', 'open', 'rooms', 'cross',
            'low_clutter', 'medium_clutter', 'high_clutter',
        ],
        'methods': ['coupled_rrt', 'decoupled_rrt', 'srrt', 'drrt', 'arc'],
    },
    {
        'label': 'kinodynamic',
        'summary_file': 'kino_summary.csv',
        'scenarios': [
            'narrow', 'open', 'rooms', 'cross',
            'low_clutter', 'medium_clutter', 'high_clutter',
            'narrow_second', 'open_second', 'rooms_second', 'cross_second',
            'low_clutter_second', 'medium_clutter_second', 'high_clutter_second',
            'narrow_double', 'open_double', 'rooms_double', 'cross_double',
            'low_clutter_double', 'medium_clutter_double', 'high_clutter_double',
        ],
        'methods': ['kino_coupled_rrt', 'kino_decoupled_rrt', 'kcbs', 'k_arc'],
    },
]


def _draw_time(df, scenario, methods, num_robots, ax):
    df_filtered = df[(df['scenario'] == scenario) & (df['method'].isin(methods)) & (df['robots'].isin(num_robots))]
    sns.lineplot(x='robots', y='planning_time', hue='method', data=df_filtered, err_style="bars", marker='o', ax=ax)
    ax.set_title('Planning Time')
    ax.set_xlabel('Number of Robots')
    ax.set_ylabel('Time (s)')
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    labels = [name_map.get(l, l) for l in labels]
    ax.legend(handles, labels)


def _draw_success_rate(df, scenario, methods, num_robots, ax):
    df_filtered = df[(df['scenario'] == scenario) & (df['method'].isin(methods)) & (df['robots'].isin(num_robots))]
    sns.lineplot(x='robots', y='solved', hue='method', data=df_filtered, err_style=None, marker='o', ax=ax)
    ax.set_title('Success Rate')
    ax.set_xlabel('Number of Robots')
    ax.set_ylabel('Success Rate')
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    labels = [name_map.get(l, l) for l in labels]
    ax.legend(handles, labels)


def _draw_makespan(df, scenario, methods, num_robots, ax):
    df_filtered = df[(df['scenario'] == scenario) & (df['method'].isin(methods)) & (df['robots'].isin(num_robots))]
    sns.lineplot(x='robots', y='makespan', hue='method', data=df_filtered, err_style="bars", marker='o', ax=ax)
    ax.set_title('Makespan')
    ax.set_xlabel('Number of Robots')
    ax.set_ylabel('Makespan (s)')
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    labels = [name_map.get(l, l) for l in labels]
    ax.legend(handles, labels)


def plot_time(df, scenario, methods, num_robots, output_dir, file_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_time(df, scenario, methods, num_robots, ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, file_name))
    print(f"Saved plot to {os.path.join(output_dir, file_name)}")
    plt.close(fig)


def plot_success_rate(df, scenario, methods, num_robots, output_dir, file_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_success_rate(df, scenario, methods, num_robots, ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, file_name))
    print(f"Saved plot to {os.path.join(output_dir, file_name)}")
    plt.close(fig)


def plot_makespan(df, scenario, methods, num_robots, output_dir, file_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_makespan(df, scenario, methods, num_robots, ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, file_name))
    print(f"Saved plot to {os.path.join(output_dir, file_name)}")
    plt.close(fig)


def generate_pdf(configs, results_dir, num_robots, pdf_path):
    drawers = [_draw_time, _draw_success_rate, _draw_makespan]
    with pdf_backend.PdfPages(pdf_path) as pdf:
        for cfg in configs:
            df = pd.read_csv(os.path.join(results_dir, cfg['summary_file']))
            for scenario in cfg['scenarios']:
                fig, axes = plt.subplots(1, 3, figsize=(22, 5))
                fig.suptitle(f"{cfg['label']} — {scenario}", fontsize=13)
                for ax, draw_fn in zip(axes, drawers):
                    draw_fn(df, scenario, cfg['methods'], num_robots, ax)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    print(f"Saved PDF to {pdf_path}")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    num_robots = [2, 4, 8, 16]
    base_plot_dir = os.path.join(project_root, 'experiments', 'plots')
    results_dir = os.path.join(project_root, 'experiments', 'results')

    metrics = [
        ('time', plot_time),
        ('success_rate', plot_success_rate),
        ('makespan', plot_makespan),
    ]

    for cfg in CONFIGS:
        label = cfg['label']
        df = pd.read_csv(os.path.join(results_dir, cfg['summary_file']))
        for metric, plot_fn in metrics:
            out_dir = os.path.join(base_plot_dir, metric, label)
            os.makedirs(out_dir, exist_ok=True)
            for scenario in cfg['scenarios']:
                plot_fn(df, scenario, cfg['methods'], num_robots, out_dir, f'{scenario}.png')

    os.makedirs(base_plot_dir, exist_ok=True)
    generate_pdf(CONFIGS, results_dir, num_robots, os.path.join(base_plot_dir, 'all_plots.pdf'))

if __name__ == "__main__":
    main()
