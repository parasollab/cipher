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
    'k_arc': 'K-ARC',
    'geometric_cipher': 'CIPHER',
    'kinodynamic_cipher': 'CIPHER',
}

# CIPHER first, then all other methods in a stable order
ALL_DISPLAY_NAMES = ['CIPHER', 'Coupled RRT', 'Decoupled RRT', 'sRRT', 'MRdRRT', 'ARC', 'KCBS', 'K-ARC']
METHOD_PALETTE = dict(zip(ALL_DISPLAY_NAMES, sns.color_palette("tab10", len(ALL_DISPLAY_NAMES))))

CONFIGS = [
    {
        'label': 'geometric',
        'summary_file': 'summary.csv',
        'scenarios': [
            # 'open', 'rooms',
            # 'low_clutter', 'medium_clutter', 'high_clutter', 
            'open_70x70', 'rooms_70x70',
            'low_clutter_70x70', 'medium_clutter_70x70', 'high_clutter_70x70',
        ],
        'methods': ['coupled_rrt', 'decoupled_rrt', 'srrt', 'drrt', 'arc', 'geometric_cipher'],
    },
    {
        'label': 'kinodynamic',
        'summary_file': 'kino_summary.csv',
        'scenarios': [
            # 'narrow', 'open', 'rooms', 'cross',
            # 'low_clutter', 'medium_clutter', 'high_clutter',
            # 'narrow_second', 'open_second', 'rooms_second', 'cross_second',
            # 'low_clutter_second', 'medium_clutter_second', 'high_clutter_second',
            # 'narrow_double', 'open_double', 'rooms_double', 'cross_double',
            # 'low_clutter_double', 'medium_clutter_double', 'high_clutter_double',
            'open_70x70', 'rooms_70x70',
            'low_clutter_70x70', 'medium_clutter_70x70', 'high_clutter_70x70',
            'open_70x70_second', 'rooms_70x70_second',
            'low_clutter_70x70_second', 'medium_clutter_70x70_second', 'high_clutter_70x70_second',
            'open_70x70_double', 'rooms_70x70_double',
            'low_clutter_70x70_double', 'medium_clutter_70x70_double', 'high_clutter_70x70_double',
        ],
        'methods': ['kino_coupled_rrt', 'kino_decoupled_rrt', 'kcbs', 'k_arc', 'kinodynamic_cipher'],
    },
]


def _draw_time(df, scenario, methods, num_robots, ax):
    robot_mask = df['robots'].isin(num_robots) if num_robots is not None else True
    df_filtered = df[(df['scenario'] == scenario) & (df['method'].isin(methods)) & robot_mask & (df['solved'] == True)].copy()
    df_filtered = df_filtered[~((df_filtered['method'] == 'srrt') & (df_filtered['planning_time'] >= 480))]
    df_filtered['planning_time'] = df_filtered['planning_time'].replace(0, 1e-4)
    df_filtered['method'] = df_filtered['method'].map(lambda m: name_map.get(m, m))
    present = [m for m in ALL_DISPLAY_NAMES if m in df_filtered['method'].unique()]
    ax.set_title('Planning Time')
    ax.set_xlabel('Number of Robots')
    ax.set_ylabel('Time (s)')
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    if df_filtered.empty or not present:
        return
    sns.lineplot(x='robots', y='planning_time', hue='method', hue_order=present, palette=METHOD_PALETTE, data=df_filtered, err_style="bars", marker='o', ax=ax)
    ax.legend()


def _draw_success_rate(df, scenario, methods, num_robots, ax):
    robot_mask = df['robots'].isin(num_robots) if num_robots is not None else True
    df_filtered = df[(df['scenario'] == scenario) & (df['method'].isin(methods)) & robot_mask].copy()
    df_filtered['method'] = df_filtered['method'].map(lambda m: name_map.get(m, m))
    present = [m for m in ALL_DISPLAY_NAMES if m in df_filtered['method'].unique()]
    ax.set_title('Success Rate')
    ax.set_xlabel('Number of Robots')
    ax.set_ylabel('Success Rate')
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    if df_filtered.empty or not present:
        return
    sns.lineplot(x='robots', y='solved', hue='method', hue_order=present, palette=METHOD_PALETTE, data=df_filtered, err_style=None, marker='o', ax=ax)
    ax.legend()


def _draw_makespan(df, scenario, methods, num_robots, ax):
    robot_mask = df['robots'].isin(num_robots) if num_robots is not None else True
    df_filtered = df[(df['scenario'] == scenario) & (df['method'].isin(methods)) & robot_mask].copy()
    df_filtered['method'] = df_filtered['method'].map(lambda m: name_map.get(m, m))
    present = [m for m in ALL_DISPLAY_NAMES if m in df_filtered['method'].unique()]
    ax.set_title('Makespan')
    ax.set_xlabel('Number of Robots')
    ax.set_ylabel('Makespan (s)')
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    if df_filtered.empty or not present:
        return
    sns.lineplot(x='robots', y='makespan', hue='method', hue_order=present, palette=METHOD_PALETTE, data=df_filtered, err_style="bars", marker='o', ax=ax)
    ax.legend()


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
            csv_path = os.path.join(results_dir, cfg['summary_file'])
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
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
    num_robots = None  # None means all robot counts present in the data
    base_plot_dir = os.path.join(project_root, 'experiments', 'plots')
    results_dir = os.path.join(project_root, 'experiments', 'results')

    metrics = [
        ('time', plot_time),
        ('success_rate', plot_success_rate),
        ('makespan', plot_makespan),
    ]

    active_configs = []
    for cfg in CONFIGS:
        csv_path = os.path.join(results_dir, cfg['summary_file'])
        if not os.path.exists(csv_path):
            print(f"Skipping '{cfg['label']}' (missing {csv_path})")
            continue
        active_configs.append(cfg)
        label = cfg['label']
        df = pd.read_csv(csv_path)
        for metric, plot_fn in metrics:
            out_dir = os.path.join(base_plot_dir, metric, label)
            os.makedirs(out_dir, exist_ok=True)
            for scenario in cfg['scenarios']:
                plot_fn(df, scenario, cfg['methods'], num_robots, out_dir, f'{scenario}.png')

    os.makedirs(base_plot_dir, exist_ok=True)
    if active_configs:
        generate_pdf(active_configs, results_dir, num_robots, os.path.join(base_plot_dir, 'all_plots.pdf'))

if __name__ == "__main__":
    main()
