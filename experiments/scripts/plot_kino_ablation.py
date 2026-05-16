"""
Plot ablation results for kinodynamic_cipher.

Reads experiments/results/kino_ablation.csv and produces an 8-page PDF plus
individual PNGs in experiments/plots/kino_ablation/.

Pages:
  1. Region size → success rate   (cols: CBS | A*)
  2. Region size → planning time  (cols: CBS | A*)
  3. Ratio → success rate         (cols: CBS | A*)
  4. Ratio → planning time        (cols: CBS | A*)
  5. CBS vs A*                    (cols: success rate | planning time)
  6. Heatmap success rate         (cols: CBS | A*)
  7. Heatmap planning time        (cols: CBS | A*)
  8. Makespan                     (cols: CBS | A*)
  9. Time breakdown (stacked bar) (cols: CBS | A*)
  10. Conflict resolution          (cols: CBS | A*)
  11. Strategy usage               (cols: CBS | A*)
"""

import argparse
import os
from pathlib import Path

import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'axes.titlesize': 15,
})

# ── colour helpers ────────────────────────────────────────────────────────────

MAPF_PALETTE = {'cbs': '#2196F3', 'astar': '#FF9800'}
MAPF_LABELS  = {'cbs': 'CBS', 'astar': 'A*'}


def _seq_palette(values):
    """Map a sorted list of numeric values to a tab10 qualitative palette."""
    cmap = plt.get_cmap('tab10')
    return {v: cmap(i / 10.0) for i, v in enumerate(sorted(values))}


# ── individual draw functions ─────────────────────────────────────────────────

def _success_by_param(ax, df, param, fixed_val, mapf_method, robot_counts):
    """Line plot: success rate vs robots, hue = param value."""
    ref_mask = pd.Series(True, index=df.index)
    if fixed_val is not None:
        other = 'robot_cell_size_ratio' if param == 'region_size' else 'region_size'
        ref_mask &= (df[other] == fixed_val)
    if robot_counts is not None:
        ref_mask &= df['robots'].isin(robot_counts)
    all_values = sorted(df[ref_mask][param].unique())
    palette = _seq_palette(all_values)

    mask = (df['mapf_method'] == mapf_method)
    if fixed_val is not None:
        other = 'robot_cell_size_ratio' if param == 'region_size' else 'region_size'
        mask &= (df[other] == fixed_val)
    if robot_counts is not None:
        mask &= df['robots'].isin(robot_counts)
    sub = df[mask].copy()
    sub['solved'] = sub['solved'].astype(float)

    ax.set_title(f"{MAPF_LABELS[mapf_method]}")
    ax.set_xlabel('Robots')
    ax.set_ylabel('Success rate')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, ls='--', lw=0.5)

    for v in all_values:
        d = sub[sub[param] == v]
        if d.empty:
            continue
        grp = d.groupby('robots')['solved'].mean().reset_index()
        ax.plot(grp['robots'], grp['solved'], marker='o', label=str(v), color=palette[v])
    ax.legend(title=param.replace('_', ' '))


def _time_by_param(ax, df, param, fixed_val, mapf_method, robot_counts):
    """Line plot: planning time (log) vs robots, hue = param value."""
    ref_mask = pd.Series(True, index=df.index)
    if fixed_val is not None:
        other = 'robot_cell_size_ratio' if param == 'region_size' else 'region_size'
        ref_mask &= (df[other] == fixed_val)
    if robot_counts is not None:
        ref_mask &= df['robots'].isin(robot_counts)
    all_values = sorted(df[ref_mask][param].unique())
    palette = _seq_palette(all_values)

    mask = (df['mapf_method'] == mapf_method) & (df['solved'] == True)
    if fixed_val is not None:
        other = 'robot_cell_size_ratio' if param == 'region_size' else 'region_size'
        mask &= (df[other] == fixed_val)
    if robot_counts is not None:
        mask &= df['robots'].isin(robot_counts)
    sub = df[mask].copy()

    ax.set_title(f"{MAPF_LABELS[mapf_method]}")
    ax.set_xlabel('Robots')
    ax.set_ylabel('Planning time (s)')
    ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', lw=0.5)

    for v in all_values:
        d = sub[sub[param] == v]
        if d.empty:
            ax.plot([], [], marker='o', linestyle='--', alpha=0.5,
                    label=f'{v} (no data)', color=palette[v])
            continue
        grp = d.groupby('robots')['planning_time'].agg(['mean', 'std']).reset_index()
        ax.errorbar(grp['robots'], grp['mean'], yerr=grp['std'],
                    marker='o', capsize=3, label=str(v), color=palette[v])
    ax.legend(title=param.replace('_', ' '))


def _cbs_vs_astar(axes, df, default_region_size, default_ratio, robot_counts):
    """Two panels: success rate and planning time, hue = mapf_method."""
    mask = (df['region_size'] == default_region_size) & (df['robot_cell_size_ratio'] == default_ratio)
    if robot_counts is not None:
        mask &= df['robots'].isin(robot_counts)
    sub = df[mask].copy()

    ax_sr, ax_t = axes

    ax_sr.set_title('Success Rate')
    ax_sr.set_xlabel('Robots')
    ax_sr.set_ylabel('Success rate')
    ax_sr.set_ylim(-0.05, 1.05)
    ax_sr.grid(True, ls='--', lw=0.5)

    ax_t.set_title('Planning Time')
    ax_t.set_xlabel('Robots')
    ax_t.set_ylabel('Planning time (s)')
    ax_t.set_yscale('log')
    ax_t.grid(True, which='both', ls='--', lw=0.5)

    if sub.empty:
        return

    sub['solved_f'] = sub['solved'].astype(float)
    for method in ['cbs', 'astar']:
        d = sub[sub['mapf_method'] == method]
        if d.empty:
            continue
        grp_sr = d.groupby('robots')['solved_f'].mean().reset_index()
        ax_sr.plot(grp_sr['robots'], grp_sr['solved_f'],
                   marker='o', label=MAPF_LABELS[method], color=MAPF_PALETTE[method])

        d_solved = d[d['solved'] == True]
        if not d_solved.empty:
            grp_t = d_solved.groupby('robots')['planning_time'].agg(['mean', 'std']).reset_index()
            ax_t.errorbar(grp_t['robots'], grp_t['mean'], yerr=grp_t['std'],
                          marker='o', capsize=3, label=MAPF_LABELS[method], color=MAPF_PALETTE[method])

    ax_sr.legend()
    ax_t.legend()


def _heatmap(ax, df, metric, mapf_method, robot_counts, fmt='.2f', cmap='YlGnBu'):
    """Heatmap: region_size × robot_cell_size_ratio → mean metric."""
    mask = (df['mapf_method'] == mapf_method)
    if robot_counts is not None:
        mask &= df['robots'].isin(robot_counts)
    if metric == 'planning_time':
        mask &= (df['solved'] == True)
    sub = df[mask].copy()

    ax.set_title(MAPF_LABELS[mapf_method])

    if sub.empty:
        return

    pivot = sub.pivot_table(
        index='robot_cell_size_ratio', columns='region_size',
        values=metric, aggfunc='mean',
    )
    pivot = pivot.sort_index(ascending=False)
    pivot = pivot[sorted(pivot.columns)]

    sns.heatmap(pivot, ax=ax, annot=True, fmt=fmt, cmap=cmap,
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax.set_xlabel('region_size')
    ax.set_ylabel('robot_cell_size_ratio')


def _makespan_by_param(ax, df, param, fixed_val, mapf_method, robot_counts):
    """Line plot: makespan vs robots, hue = param value (solved only)."""
    ref_mask = pd.Series(True, index=df.index)
    if fixed_val is not None:
        other = 'robot_cell_size_ratio' if param == 'region_size' else 'region_size'
        ref_mask &= (df[other] == fixed_val)
    if robot_counts is not None:
        ref_mask &= df['robots'].isin(robot_counts)
    all_values = sorted(df[ref_mask][param].unique())
    palette = _seq_palette(all_values)

    mask = (df['mapf_method'] == mapf_method) & df['makespan'].notna()
    if fixed_val is not None:
        other = 'robot_cell_size_ratio' if param == 'region_size' else 'region_size'
        mask &= (df[other] == fixed_val)
    if robot_counts is not None:
        mask &= df['robots'].isin(robot_counts)
    sub = df[mask].copy()

    ax.set_title(f"{MAPF_LABELS[mapf_method]}")
    ax.set_xlabel('Robots')
    ax.set_ylabel('Makespan')
    ax.grid(True, ls='--', lw=0.5)

    if sub.empty:
        return
    for v in all_values:
        d = sub[sub[param] == v]
        if d.empty:
            continue
        grp = d.groupby('robots')['makespan'].agg(['mean', 'std']).reset_index()
        ax.errorbar(grp['robots'], grp['mean'], yerr=grp['std'],
                    marker='o', capsize=3, label=str(v), color=palette[v])
    ax.legend(title=param.replace('_', ' '))


# ── resolution_stats draw functions ──────────────────────────────────────────

TIME_COMPONENTS = [
    ('time_mapf',                'MAPF'),
    ('time_guided',              'Guided planning'),
    ('time_decomp',              'Decomposition'),
    ('time_conflict_resolution', 'Conflict resolution'),
]

STRATEGY_PAIRS = [
    ('decomp_refine_attempts',  'decomp_refine_successes',  'Decomp refine'),
    ('expansion_attempts',      'expansion_successes',      'Expansion'),
    ('composite_attempts',      'composite_successes',      'Composite'),
    ('decoupled_attempts',      'decoupled_successes',      'Decoupled'),
]


def _time_breakdown(ax, df, mapf_method, robot_counts):
    """Stacked bar: mean time per component, grouped by robot count."""
    mask = df['mapf_method'] == mapf_method
    if robot_counts is not None:
        mask &= df['robots'].isin(robot_counts)
    sub = df[mask].copy()

    ax.set_title(MAPF_LABELS[mapf_method])
    ax.set_xlabel('Robots')
    ax.set_ylabel('Mean time (s)')
    ax.grid(True, axis='y', ls='--', lw=0.5)

    cols = [c for c, _ in TIME_COMPONENTS if c in sub.columns]
    if sub.empty or not cols:
        return

    grp = sub.groupby('robots')[[c for c, _ in TIME_COMPONENTS if c in sub.columns]].mean()
    robots = grp.index.tolist()
    x = np.arange(len(robots))
    width = 0.6
    bottom = np.zeros(len(robots))
    colors = sns.color_palette('Set2', len(TIME_COMPONENTS))
    for (col, label), color in zip(TIME_COMPONENTS, colors):
        if col not in grp.columns:
            continue
        vals = grp[col].fillna(0).values
        ax.bar(x, vals, width, bottom=bottom, label=label, color=color)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(robots)
    ax.legend()


def _conflict_resolution(ax, df, mapf_method, robot_counts):
    """Line plot: conflicts encountered vs resolved vs robots."""
    mask = df['mapf_method'] == mapf_method
    if robot_counts is not None:
        mask &= df['robots'].isin(robot_counts)
    sub = df[mask].copy()

    ax.set_title(MAPF_LABELS[mapf_method])
    ax.set_xlabel('Robots')
    ax.set_ylabel('Mean count')
    ax.grid(True, ls='--', lw=0.5)

    for col, label, color in [
        ('total_conflicts_encountered', 'Encountered', '#E53935'),
        ('total_conflicts_resolved',    'Resolved',    '#43A047'),
    ]:
        if col not in sub.columns or sub[col].isna().all():
            continue
        grp = sub.groupby('robots')[col].mean().reset_index()
        ax.plot(grp['robots'], grp[col], marker='o', label=label, color=color)
    ax.legend()


def _strategy_usage(ax, df, mapf_method, robot_counts):
    """Grouped bar: attempts vs successes for each conflict-resolution strategy."""
    mask = df['mapf_method'] == mapf_method
    if robot_counts is not None:
        mask &= df['robots'].isin(robot_counts)
    sub = df[mask].copy()

    ax.set_title(MAPF_LABELS[mapf_method])
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Mean count per run')
    ax.grid(True, axis='y', ls='--', lw=0.5)

    labels, attempts, successes = [], [], []
    for att_col, suc_col, label in STRATEGY_PAIRS:
        if att_col not in sub.columns:
            continue
        labels.append(label)
        attempts.append(sub[att_col].mean() if not sub[att_col].isna().all() else 0)
        successes.append(sub[suc_col].mean() if not sub[suc_col].isna().all() else 0)

    if not labels:
        return

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, attempts,  w, label='Attempts',  color='#5C6BC0')
    ax.bar(x + w / 2, successes, w, label='Successes', color='#26A69A')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend()


# ── page builders ─────────────────────────────────────────────────────────────

def _make_param_sweep_page(df, param, fixed_val, robot_counts, draw_fn, suptitle, fixed_label):
    """Two-column figure (CBS | A*) for a given draw function."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    desc = f"fixed {fixed_label}={fixed_val}" if fixed_val is not None else "all values"
    fig.suptitle(f"{suptitle}  ({desc})", fontsize=13)
    for ax, method in zip(axes, ['cbs', 'astar']):
        draw_fn(ax, df, param, fixed_val, method, robot_counts)
    fig.tight_layout()
    return fig


def _make_single_param_sweep(df, param, fixed_val, robot_counts, draw_fn, method):
    """Single-panel figure for one MAPF method."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    draw_fn(ax, df, param, fixed_val, method, robot_counts)
    ax.set_title('')
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.tick_params(labelsize=18)
    legend = ax.get_legend()
    if legend is not None:
        legend.get_title().set_fontsize(18)
        for text in legend.get_texts():
            text.set_fontsize(16)
    fig.tight_layout()
    return fig


def _make_cbs_vs_astar_page(df, default_rs, default_ratio, robot_counts):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"CBS vs A*  (region_size={default_rs}, ratio={default_ratio})", fontsize=13
    )
    _cbs_vs_astar(axes, df, default_rs, default_ratio, robot_counts)
    fig.tight_layout()
    return fig


def _make_heatmap_page(df, metric, robot_counts, suptitle, fmt, cmap):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rc_label = "all robots" if robot_counts is None else f"robots={robot_counts}"
    fig.suptitle(f"{suptitle}  ({rc_label})", fontsize=13)
    for ax, method in zip(axes, ['cbs', 'astar']):
        _heatmap(ax, df, metric, method, robot_counts, fmt=fmt, cmap=cmap)
    fig.tight_layout()
    return fig


def _make_two_col_page(df, robot_counts, draw_fn, suptitle):
    """Generic two-column page (CBS | A*) for stats draw functions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(suptitle, fontsize=13)
    for ax, method in zip(axes, ['cbs', 'astar']):
        draw_fn(ax, df, method, robot_counts)
    fig.tight_layout()
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    default_csv = str(project_root / 'experiments' / 'results' / 'kino_ablation.csv')
    default_out = str(project_root / 'experiments' / 'plots' / 'kino_ablation')

    parser = argparse.ArgumentParser(
        description="Plot ablation results for kinodynamic_cipher.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', default=default_csv, help="Path to ablation CSV")
    parser.add_argument('--output-dir', default=default_out, help="Directory for output plots")
    parser.add_argument('--default-region-size', type=float, default=1.0,
                        help="Fixed region_size for ratio-sweep and CBS-vs-A* plots")
    parser.add_argument('--default-ratio', type=float, default=1.0,
                        help="Fixed robot_cell_size_ratio for region-size-sweep plots")
    parser.add_argument('--robots', nargs='+', type=int, default=None,
                        help="Filter to these robot counts (default: all)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: CSV not found at {args.input}")
        return

    df = pd.read_csv(args.input)
    df['solved'] = df['solved'].astype(str).str.lower().map(
        lambda x: True if x == 'true' else (False if x == 'false' else bool(x))
    )

    os.makedirs(args.output_dir, exist_ok=True)
    pdf_path = os.path.join(args.output_dir, 'kino_ablation.pdf')

    robot_counts = args.robots
    drs = args.default_region_size
    drat = args.default_ratio

    pages = [
        ("p1_region_size_success",
         _make_param_sweep_page(df, 'region_size', drat, robot_counts,
                                _success_by_param, 'Region size → Success Rate', 'ratio')),
        ("p2_region_size_time_cbs",
         _make_single_param_sweep(df, 'region_size', drat, robot_counts, _time_by_param, 'cbs')),
        ("p2_region_size_time_astar",
         _make_single_param_sweep(df, 'region_size', drat, robot_counts, _time_by_param, 'astar')),
        ("p3_ratio_success",
         _make_param_sweep_page(df, 'robot_cell_size_ratio', drs, robot_counts,
                                _success_by_param, 'Robot cell ratio → Success Rate', 'region_size')),
        ("p4_ratio_time",
         _make_param_sweep_page(df, 'robot_cell_size_ratio', drs, robot_counts,
                                _time_by_param, 'Robot cell ratio → Planning Time', 'region_size')),
        ("p5_cbs_vs_astar",
         _make_cbs_vs_astar_page(df, drs, drat, robot_counts)),
        ("p6_heatmap_success",
         _make_heatmap_page(df, 'solved', robot_counts,
                            'Heatmap: Mean Success Rate', '.2f', 'YlGn')),
        ("p7_heatmap_time",
         _make_heatmap_page(df, 'planning_time', robot_counts,
                            'Heatmap: Mean Planning Time (solved)', '.1f', 'YlOrRd_r')),
        ("p8_makespan",
         _make_param_sweep_page(df, 'region_size', drat, robot_counts,
                                _makespan_by_param, 'Makespan', 'ratio')),
        ("p9_time_breakdown",  _make_two_col_page(
            df, robot_counts, _time_breakdown, 'Time Breakdown by Component')),
        ("p10_conflict_resolution", _make_two_col_page(
            df, robot_counts, _conflict_resolution, 'Conflict Resolution')),
        ("p11_strategy_usage", _make_two_col_page(
            df, robot_counts, _strategy_usage, 'Conflict-Resolution Strategy Usage')),
    ]

    with pdf_backend.PdfPages(pdf_path) as pdf:
        for name, fig in pages:
            png_path = os.path.join(args.output_dir, f'{name}.png')
            fig.savefig(png_path, dpi=150)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  saved {png_path}")

    print(f"\nPDF: {pdf_path}")


if __name__ == '__main__':
    main()
