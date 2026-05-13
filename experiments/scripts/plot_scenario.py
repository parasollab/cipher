import argparse
import os
from collections import OrderedDict

import numpy as np
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

CIPHER_RAW_METHODS = {'geometric_cipher', 'kinodynamic_cipher'}

LOG_SCALE_METRICS = {
    'planning_time',
    'time_setup_decomposition_seconds',
    'time_transition_feasibility_seconds',
    'time_mapf_seconds',
    'time_guided_planning_seconds',
    'time_cbs_decomposition_seconds',
    'time_refinement_decomposition_seconds',
    'time_check_conflicts_seconds',
    'time_conflict_resolution_seconds',
    'time_hierarchical_resolution_seconds',
    'time_full_composite_resolution_seconds',
    'time_extract_bounds_seconds',
    'time_integrate_paths_seconds',
    'time_recheck_conflicts_seconds',
    'time_decoupled_fallback_seconds',
    'time_coupled_fallback_seconds',
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
    'time_setup_decomposition_seconds': 'Setup / Decomp Time (s)',
    'time_transition_feasibility_seconds': 'Transition Feasibility Time (s)',
    'time_mapf_seconds': 'MAPF Time (s)',
    'time_guided_planning_seconds': 'Guided Planning Time (s)',
    'time_cbs_decomposition_seconds': 'CBS Decomp Time (s)',
    'time_refinement_decomposition_seconds': 'Refinement Decomp Time (s)',
    'time_check_conflicts_seconds': 'Conflict Detection Time (s)',
    'time_conflict_resolution_seconds': 'Conflict Resolution Time (s)',
    'time_hierarchical_resolution_seconds': 'Hierarchical Resolution Time (s)',
    'time_full_composite_resolution_seconds': 'Full Composite Resolution Time (s)',
    'time_extract_bounds_seconds': 'Extract Bounds Time (s)',
    'time_integrate_paths_seconds': 'Integrate Paths Time (s)',
    'time_recheck_conflicts_seconds': 'Recheck Conflicts Time (s)',
    'time_decoupled_fallback_seconds': 'Decoupled Fallback Time (s)',
    'time_coupled_fallback_seconds': 'Coupled Fallback Time (s)',
}

# Top-level phases: mutually exclusive at the plan() call-site level.
# time_conflict_resolution_seconds is inclusive; sub-phases are its drill-down.
TIMING_PHASES = OrderedDict([
    ('Setup',               ['time_setup_decomposition_seconds',
                             'time_transition_feasibility_seconds']),
    ('MAPF',                ['time_mapf_seconds']),
    ('CBS Decomp',          ['time_cbs_decomposition_seconds']),
    ('Guided Planning',     ['time_guided_planning_seconds']),
    ('Conflict Detection',  ['time_check_conflicts_seconds']),
    ('Conflict Resolution', ['time_conflict_resolution_seconds']),
    ('Fallback',            ['time_decoupled_fallback_seconds',
                             'time_coupled_fallback_seconds']),
])

# Sub-phases inside conflict resolution for the drill-down bar.
# 'Other CR' = time_conflict_resolution_seconds - sum(these), clipped to 0.
CR_SUBPHASES = OrderedDict([
    ('Hierarchical',    ['time_hierarchical_resolution_seconds']),
    ('Full Composite',  ['time_full_composite_resolution_seconds']),
    ('Decomposition',   ['time_refinement_decomposition_seconds']),
    ('Extract Bounds',  ['time_extract_bounds_seconds']),
    ('Integrate Paths', ['time_integrate_paths_seconds']),
    ('Recheck',         ['time_recheck_conflicts_seconds']),
])

PHASE_PALETTE = dict(zip(
    list(TIMING_PHASES.keys()) + ['Other'],
    sns.color_palette("tab20", len(TIMING_PHASES) + 1),
))

CR_PALETTE = dict(zip(
    list(CR_SUBPHASES.keys()) + ['Other CR'],
    sns.color_palette("Set2", len(CR_SUBPHASES) + 1),
))


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
    sr = df.groupby(['method', 'robots'])['solved'].mean().reset_index()
    sr.rename(columns={'solved': 'success_rate'}, inplace=True)

    solved_df = df[df['solved']].drop(columns=['solved', 'seed'])
    agg = solved_df.groupby(['method', 'robots']).mean(numeric_only=True).reset_index()

    merged = sr.merge(agg, on=['method', 'robots'], how='outer')
    return merged


def build_line_grid_fig(df, scenario):
    df = df.copy()
    df['method'] = df['method'].map(lambda m: name_map.get(m, m))

    numeric_cols = [c for c in df.columns if c not in ('method', 'robots')]
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
    return fig


def _phase_means(df_cipher, phase_dict):
    """Return (robot_counts, {phase: array_of_means}) for each phase in phase_dict."""
    robot_counts = sorted(df_cipher['robots'].unique())
    result = {}
    for phase, fields in phase_dict.items():
        vals = []
        for n in robot_counts:
            sub = df_cipher[df_cipher['robots'] == n]
            val = sum(sub[f].mean() for f in fields if f in sub.columns and sub[f].notna().any())
            vals.append(val)
        result[phase] = np.array(vals)
    return robot_counts, result


def _fallback_mask(df):
    """True for rows where any fallback planner was used."""
    mask = pd.Series(False, index=df.index)
    for col in ('decoupled_planner_attempts', 'coupled_planner_attempts',
                'time_decoupled_fallback_seconds', 'time_coupled_fallback_seconds'):
        if col in df.columns:
            mask |= df[col].fillna(0) > 0
    return mask


def _draw_phase_bars(ax, x, robot_counts, phase_vals, other_vals,
                     normalizer, palette, phase_order,
                     other_label, add_legend, ylabel, title):
    """Draw one stacked bar subplot. normalizer array = denominator for % mode; None = absolute."""
    bottom = np.zeros(len(robot_counts))
    for phase in phase_order + [other_label]:
        vals = other_vals if phase == other_label else phase_vals[phase]
        if normalizer is not None:
            vals = np.where(normalizer > 0, 100.0 * vals / normalizer, 0.0)
        ax.bar(x, vals, bottom=bottom, width=0.7, label=phase, color=palette[phase])
        bottom += vals

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if normalizer is not None:
        ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in robot_counts])
    ax.set_xlabel('Number of Robots')
    ax.grid(axis='y', ls='--', linewidth=0.5)
    if add_legend:
        ax.legend(loc='upper left', fontsize=8, ncol=2)


def build_timing_breakdown_fig(df_raw, scenario):
    cipher_df = df_raw[df_raw['method'].isin(CIPHER_RAW_METHODS) & df_raw['solved']].copy()
    if cipher_df.empty:
        return None

    # Split into fallback / no-fallback groups
    fb_mask = _fallback_mask(cipher_df)
    groups = []
    if (~fb_mask).any():
        groups.append(('No Fallback', cipher_df[~fb_mask]))
    if fb_mask.any():
        groups.append(('With Fallback', cipher_df[fb_mask]))

    # Decide whether to show the CR drill-down (based on all CIPHER data)
    all_rc, all_pv = _phase_means(cipher_df, TIMING_PHASES)
    all_pt = np.array([cipher_df[cipher_df['robots'] == n]['planning_time'].mean() for n in all_rc])
    all_cr = all_pv.get('Conflict Resolution', np.zeros(len(all_rc)))
    show_cr = bool(np.any(all_cr > 0.05 * all_pt))

    nrows = 3 if show_cr else 2
    ncols = len(groups)
    fig_w = max(8, sum(max(4, len(gdf['robots'].unique()) * 0.9) for _, gdf in groups))
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, 5 * nrows), squeeze=False)
    fig.suptitle(f'{scenario} — CIPHER Timing Breakdown', fontsize=13)

    phase_order = list(TIMING_PHASES.keys())
    cr_order = list(CR_SUBPHASES.keys())

    for col, (group_label, gdf) in enumerate(groups):
        rc, pv = _phase_means(gdf, TIMING_PHASES)
        rc = list(rc)
        pt = np.array([gdf[gdf['robots'] == n]['planning_time'].mean() for n in rc])
        phase_sum = sum(pv.values())
        other = np.maximum(0, pt - phase_sum)
        x = np.arange(len(rc))
        add_legend = (col == 0)

        # Row 0: absolute, with total planning time annotated above each bar
        _draw_phase_bars(
            axes[0][col], x, rc, pv, other,
            normalizer=None, palette=PHASE_PALETTE,
            phase_order=phase_order, other_label='Other',
            add_legend=add_legend,
            ylabel='Time (s)', title=f'Absolute — {group_label}',
        )
        tops = sum(pv.values()) + other
        ymax = tops.max() if tops.max() > 0 else 1
        for i, (total, top) in enumerate(zip(pt, tops)):
            axes[0][col].text(x[i], top + 0.01 * ymax, f'{total:.1f}s',
                              ha='center', va='bottom', fontsize=7)

        # Row 1: normalized
        _draw_phase_bars(
            axes[1][col], x, rc, pv, other,
            normalizer=pt, palette=PHASE_PALETTE,
            phase_order=phase_order, other_label='Other',
            add_legend=add_legend,
            ylabel='% of Planning Time', title=f'Relative — {group_label}',
        )

        # Row 2: CR drill-down (conditional)
        if show_cr:
            cr_vals = pv.get('Conflict Resolution', np.zeros(len(rc)))
            _, cr_sub = _phase_means(gdf, CR_SUBPHASES)
            other_cr = np.maximum(0, cr_vals - sum(cr_sub.values()))
            _draw_phase_bars(
                axes[2][col], x, rc, cr_sub, other_cr,
                normalizer=None, palette=CR_PALETTE,
                phase_order=cr_order, other_label='Other CR',
                add_legend=add_legend,
                ylabel='Time (s)', title=f'CR Drill-down — {group_label}',
            )

    fig.tight_layout()
    return fig


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

    if args.output:
        output_path = args.output
    else:
        plots_dir = os.path.join(script_dir, '..', 'plots', 'scenario_metrics')
        os.makedirs(plots_dir, exist_ok=True)
        output_path = os.path.join(plots_dir, f'{args.scenario}_metrics.pdf')

    df_raw = discover_data(scenario_dir, set(args.robots) if args.robots else None)
    df_plot = build_plot_df(df_raw)

    figs = [build_line_grid_fig(df_plot, args.scenario)]
    timing_fig = build_timing_breakdown_fig(df_raw, args.scenario)
    if timing_fig is not None:
        figs.append(timing_fig)

    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.pdf':
        with pdf_backend.PdfPages(output_path) as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches='tight')
    else:
        figs[0].savefig(output_path)

    for fig in figs:
        plt.close(fig)

    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
