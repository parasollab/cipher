import argparse
import os
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend

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

ALL_DISPLAY_NAMES = ['CIPHER', 'Coupled RRT', 'Decoupled RRT', 'sRRT', 'MRdRRT', 'ARC', 'KCBS', 'K-ARC']
METHOD_PALETTE = dict(zip(ALL_DISPLAY_NAMES, sns.color_palette("tab10", len(ALL_DISPLAY_NAMES))))

LOG_SCALE_METRICS = {
    'planning_time',
    'time_mapf_seconds',
    'time_guided_planning_seconds',
    'time_decomposition_seconds',
    'time_conflict_resolution_seconds',
}

METRIC_LABELS = {
    'success_rate': 'Success Rate',
    'planning_time': 'Planning Time (s)',
    'total_conflicts_encountered': 'Conflicts Encountered',
    'total_conflicts_resolved': 'Conflicts Resolved',
    'decomposition_refinement_attempts': 'Decomp. Refinement Attempts',
    'decomposition_refinement_successes': 'Decomp. Refinement Successes',
    'expansion_attempts': 'Expansion Attempts',
    'expansion_successes': 'Expansion Successes',
    'composite_planner_attempts': 'Composite Planner Attempts',
    'composite_planner_successes': 'Composite Planner Successes',
    'decoupled_planner_attempts': 'Decoupled Planner Attempts',
    'decoupled_planner_successes': 'Decoupled Planner Successes',
    'time_mapf_seconds': 'MAPF Time (s)',
    'time_guided_planning_seconds': 'Guided Planning Time (s)',
    'time_decomposition_seconds': 'Decomposition Time (s)',
    'time_conflict_resolution_seconds': 'Conflict Resolution Time (s)',
}


def discover_data(scenario_dir, robots_filter):
    rows = []
    if not os.path.isdir(scenario_dir):
        raise SystemExit(f"Scenario directory not found: {scenario_dir}")

    for method in sorted(os.listdir(scenario_dir)):
        method_dir = os.path.join(scenario_dir, method)
        if not os.path.isdir(method_dir):
            continue
        for n_robots_str in sorted(os.listdir(method_dir)):
            try:
                n_robots = int(n_robots_str)
            except ValueError:
                continue
            if robots_filter and n_robots not in robots_filter:
                continue
            robots_dir = os.path.join(method_dir, n_robots_str)
            for fname in sorted(os.listdir(robots_dir)):
                if not fname.endswith('.yaml'):
                    continue
                stem = os.path.splitext(fname)[0]
                try:
                    seed = int(stem)
                except ValueError:
                    continue
                fpath = os.path.join(robots_dir, fname)
                with open(fpath) as f:
                    data = yaml.safe_load(f)
                row = {
                    'method': method,
                    'robots': n_robots,
                    'seed': seed,
                    'solved': bool(data.get('solved', False)),
                    'planning_time': data.get('planning_time'),
                }
                for k, v in (data.get('resolution_stats') or {}).items():
                    row[k] = v
                rows.append(row)

    if not rows:
        raise SystemExit(f"No YAML files found under {scenario_dir}")
    return pd.DataFrame(rows)


def build_plot_df(df):
    # success_rate: mean of solved per method+robots (all seeds)
    sr = df.groupby(['method', 'robots'])['solved'].mean().reset_index()
    sr.rename(columns={'solved': 'success_rate'}, inplace=True)

    # other metrics: solved runs only, drop solved/seed columns
    solved_df = df[df['solved']].drop(columns=['solved', 'seed'])
    agg = solved_df.groupby(['method', 'robots']).mean(numeric_only=True).reset_index()

    merged = sr.merge(agg, on=['method', 'robots'], how='outer')
    return merged


def plot_all(df, scenario, output_path):
    df = df.copy()
    df['method'] = df['method'].map(lambda m: name_map.get(m, m))

    numeric_cols = [c for c in df.columns if c not in ('method', 'robots')]
    # order: success_rate first, then planning_time, then rest
    ordered = ['success_rate', 'planning_time'] + [
        c for c in numeric_cols if c not in ('success_rate', 'planning_time')
    ]
    metrics = [m for m in ordered if m in df.columns and df[m].notna().any()]

    ncols = min(4, len(metrics))
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    fig.suptitle(scenario, fontsize=14)
    axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]

    present_methods = [m for m in ALL_DISPLAY_NAMES if m in df['method'].unique()]
    unknown = [m for m in df['method'].unique() if m not in ALL_DISPLAY_NAMES]
    palette = {**METHOD_PALETTE, **dict(zip(unknown, sns.color_palette("Set2", len(unknown))))}

    for ax, metric in zip(axes_flat, metrics):
        subset = df[['method', 'robots', metric]].dropna(subset=[metric])
        label = METRIC_LABELS.get(metric, metric.replace('_', ' ').title())
        ax.set_title(label)
        ax.set_xlabel('Number of Robots')
        ax.set_ylabel(label)
        ax.grid(True, which='both', ls='--', linewidth=0.5)

        if metric in LOG_SCALE_METRICS:
            ax.set_yscale('log')
        if metric == 'success_rate':
            ax.set_ylim(0, 1.05)

        if subset.empty:
            continue

        order = present_methods + [m for m in unknown if m in subset['method'].unique()]
        hue_order = [m for m in order if m in subset['method'].unique()]
        sns.lineplot(
            x='robots', y=metric, hue='method',
            hue_order=hue_order, palette=palette,
            data=subset, err_style='bars', marker='o', ax=ax,
        )
        ax.legend(fontsize=7)

    for ax in list(axes_flat)[len(metrics):]:
        ax.set_visible(False)

    fig.tight_layout()

    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.pdf':
        with pdf_backend.PdfPages(output_path) as pdf:
            pdf.savefig(fig)
    else:
        fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot cipher output metrics for a scenario.')
    parser.add_argument('scenario', help='Scenario name, e.g. boxes_70x70')
    parser.add_argument('--robots', type=int, nargs='+', metavar='N', help='Robot counts to include (default: all)')
    parser.add_argument('--results-dir', default=None, metavar='DIR', help='Path to results directory')
    parser.add_argument('--output', default=None, metavar='FILE', help='Output file (default: <scenario>_metrics.pdf)')
    args = parser.parse_args()

    if args.results_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.results_dir = os.path.join(script_dir, '..', 'results')

    scenario_dir = os.path.join(args.results_dir, args.scenario)
    output_path = args.output or f'{args.scenario}_metrics.pdf'

    df_raw = discover_data(scenario_dir, set(args.robots) if args.robots else None)
    df_plot = build_plot_df(df_raw)
    plot_all(df_plot, args.scenario, output_path)


if __name__ == '__main__':
    main()
