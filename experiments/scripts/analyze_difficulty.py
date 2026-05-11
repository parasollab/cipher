#!/usr/bin/env python3
"""
Analyze which start/goal configuration properties correlate with CIPHER success vs failure.

Usage:
    python analyze_difficulty.py [--scenarios open rooms ...]
                                  [--robots 8 10 12 ...]
                                  [--output-dir experiments/plots/difficulty]
                                  [--no-save-csv]
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _point_to_box_dist(px, py, cx, cy, sx, sy):
    dx = max(abs(px - cx) - sx / 2, 0.0)
    dy = max(abs(py - cy) - sy / 2, 0.0)
    return np.sqrt(dx * dx + dy * dy)


def _min_obs_dist(px, py, obstacles):
    if not obstacles:
        return np.inf
    return min(
        _point_to_box_dist(px, py, o['center'][0], o['center'][1], o['size'][0], o['size'][1])
        for o in obstacles
    )


def _path_min_clearance(sx, sy, gx, gy, obstacles, resolution=0.5):
    if not obstacles:
        return np.inf
    n = max(2, int(np.hypot(gx - sx, gy - sy) / resolution) + 1)
    ts = np.linspace(0, 1, n)
    return min(
        _min_obs_dist(sx + t * (gx - sx), sy + t * (gy - sy), obstacles)
        for t in ts
    )


def _ccw(ax, ay, bx, by, cx, cy):
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)


def _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    return (
        _ccw(ax, ay, cx, cy, dx, dy) != _ccw(bx, by, cx, cy, dx, dy) and
        _ccw(ax, ay, bx, by, cx, cy) != _ccw(ax, ay, bx, by, dx, dy)
    )


def _cell(x, y, region_size=5.0):
    return (int(x / region_size), int(y / region_size))


# ── Per-seed metrics ──────────────────────────────────────────────────────────

def compute_metrics(robots, obstacles, region_size=5.0):
    n = len(robots)
    starts = np.array([[r['start'][0], r['start'][1]] for r in robots])
    goals = np.array([[r['goal'][0], r['goal'][1]] for r in robots])
    m = {}

    # Pairwise start / goal distances
    if n > 1:
        sd = pdist(starts)
        gd = pdist(goals)
        m['min_start_dist'] = sd.min()
        m['mean_start_dist'] = sd.mean()
        m['min_goal_dist'] = gd.min()
        m['mean_goal_dist'] = gd.mean()
    else:
        m.update(min_start_dist=np.nan, mean_start_dist=np.nan,
                 min_goal_dist=np.nan, mean_goal_dist=np.nan)

    # Individual displacements
    disp = np.linalg.norm(goals - starts, axis=1)
    m['min_displacement'] = disp.min()
    m['mean_displacement'] = disp.mean()
    m['max_displacement'] = disp.max()

    # Cross-distances: dist(start_i, goal_j) for i != j
    cross = [np.linalg.norm(starts[i] - goals[j])
             for i in range(n) for j in range(n) if i != j] if n > 1 else []
    m['min_cross_dist'] = min(cross) if cross else np.nan
    m['mean_cross_dist'] = float(np.mean(cross)) if cross else np.nan

    # Path crossings
    n_pairs = n * (n - 1) // 2
    crossings = sum(
        _segments_intersect(
            starts[i, 0], starts[i, 1], goals[i, 0], goals[i, 1],
            starts[j, 0], starts[j, 1], goals[j, 0], goals[j, 1],
        )
        for i in range(n) for j in range(i + 1, n)
    )
    m['num_crossings'] = crossings
    m['crossing_fraction'] = crossings / n_pairs if n_pairs > 0 else 0.0

    # Robots sharing the same CIPHER grid cell
    sc = Counter(_cell(s[0], s[1], region_size) for s in starts)
    gc = Counter(_cell(g[0], g[1], region_size) for g in goals)
    m['shared_start_cells'] = sum(v - 1 for v in sc.values() if v > 1)
    m['shared_goal_cells'] = sum(v - 1 for v in gc.values() if v > 1)

    # Obstacle distances
    s_dists = [_min_obs_dist(s[0], s[1], obstacles) for s in starts]
    g_dists = [_min_obs_dist(g[0], g[1], obstacles) for g in goals]
    m['min_start_obs_dist'] = min(s_dists)
    m['mean_start_obs_dist'] = float(np.mean(s_dists))
    m['min_goal_obs_dist'] = min(g_dists)
    m['mean_goal_obs_dist'] = float(np.mean(g_dists))

    # Minimum clearance along each robot's straight-line path
    clearances = [
        _path_min_clearance(starts[i, 0], starts[i, 1], goals[i, 0], goals[i, 1], obstacles)
        for i in range(n)
    ]
    m['min_path_obs_clearance'] = min(clearances)

    return m


METRIC_COLS = [
    'min_start_dist', 'mean_start_dist',
    'min_goal_dist', 'mean_goal_dist',
    'min_displacement', 'mean_displacement', 'max_displacement',
    'num_crossings', 'crossing_fraction',
    'min_cross_dist', 'mean_cross_dist',
    'shared_start_cells', 'shared_goal_cells',
    'min_start_obs_dist', 'mean_start_obs_dist',
    'min_goal_obs_dist', 'mean_goal_obs_dist',
    'min_path_obs_clearance',
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_seed_metrics(df_cipher, experiments_root):
    rows = []
    missing = 0
    for _, row in df_cipher.iterrows():
        seed_path = (experiments_root / row['scenario'] / 'seeds'
                     / str(row['robots']) / f"{row['seed']}.yaml")
        if not seed_path.exists():
            missing += 1
            continue
        cfg = yaml.safe_load(open(seed_path))
        m = compute_metrics(cfg['robots'], cfg['environment'].get('obstacles', []))
        m.update(scenario=row['scenario'], robots=row['robots'],
                 seed=row['seed'], solved=bool(row['solved']),
                 planning_time=row['planning_time'])
        rows.append(m)
    if missing:
        print(f'  Warning: {missing} seed files not found')
    return pd.DataFrame(rows)


# ── Statistical analysis ──────────────────────────────────────────────────────

def _rank_biserial(u, n1, n2):
    return 1.0 - 2.0 * u / (n1 * n2)


def analyze(df, metric_cols):
    solved = df[df['solved']]
    failed = df[~df['solved']]
    rows = []
    for col in metric_cols:
        sv = solved[col].replace([np.inf, -np.inf], np.nan).dropna()
        fv = failed[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sv) < 2 or len(fv) < 2:
            continue
        if pd.concat([sv, fv]).nunique() <= 1:
            continue
        u, p = stats.mannwhitneyu(sv, fv, alternative='two-sided')
        rows.append({
            'metric': col,
            'solved_median': sv.median(),
            'failed_median': fv.median(),
            'effect_size': _rank_biserial(u, len(sv), len(fv)),
            'p_value': p,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result['abs_effect'] = result['effect_size'].abs()
    return result.sort_values('abs_effect', ascending=False).reset_index(drop=True)


# ── Plots ─────────────────────────────────────────────────────────────────────

_PALETTE = {'Solved': '#2196F3', 'Failed': '#F44336'}


def plot_violin(df, metric_cols, output_dir, scenarios):
    for scenario in scenarios:
        sub = df[df['scenario'] == scenario].copy()
        if sub.empty or sub['solved'].nunique() < 2:
            continue
        sub['Status'] = sub['solved'].map({True: 'Solved', False: 'Failed'})
        present = [c for c in metric_cols
                   if c in sub.columns
                   and sub[c].replace([np.inf, -np.inf], np.nan).notna().any()]
        if not present:
            continue
        ncols = 3
        nrows = (len(present) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()
        for ax, col in zip(axes, present):
            plot_data = sub.copy()
            plot_data[col] = plot_data[col].replace([np.inf, -np.inf], np.nan)
            plot_data = plot_data.dropna(subset=[col])
            if plot_data.empty:
                ax.set_visible(False)
                continue
            sns.violinplot(x='Status', y=col, hue='Status', data=plot_data,
                           palette=_PALETTE, order=['Solved', 'Failed'],
                           inner='box', legend=False, ax=ax)
            ax.set_title(col.replace('_', ' '), fontsize=9)
            ax.set_xlabel('')
        for ax in axes[len(present):]:
            ax.set_visible(False)
        fig.suptitle(f'Metric distributions — {scenario}', fontsize=13)
        fig.tight_layout()
        out = output_dir / f'violin_{scenario}.png'
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f'  Saved {out}')


def plot_effect_sizes(ranking, output_dir, title_suffix=''):
    if ranking.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, len(ranking) * 0.45)))
    colors = ['#2196F3' if e >= 0 else '#F44336' for e in ranking['effect_size']]
    ax.barh(ranking['metric'], ranking['effect_size'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Rank-biserial correlation  (positive = higher in solved runs)')
    ax.set_title(f'Effect size: solved vs failed{title_suffix}')
    ax.invert_yaxis()
    fig.tight_layout()
    slug = title_suffix.strip().replace(' ', '_').replace('—', '').replace('/', '_')
    out = output_dir / f'effect_sizes{slug}.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'  Saved {out}')


def plot_corr_heatmap(df, metric_cols, output_dir):
    cols = [c for c in metric_cols if c in df.columns]
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty or sub.shape[1] < 2:
        return
    corr = sub.corr(method='spearman')
    size = max(6, len(corr) * 0.75)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    labels = [c.replace('_', '\n') for c in corr.columns]
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.3, vmin=-1, vmax=1)
    ax.set_title('Spearman correlation between metrics')
    fig.tight_layout()
    out = output_dir / 'metric_correlation.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'  Saved {out}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    results_dir = project_root / 'experiments' / 'results'
    experiments_root = project_root / 'experiments'

    parser = argparse.ArgumentParser(description='Analyze CIPHER difficulty by seed metrics')
    parser.add_argument('--scenarios', nargs='+', default=None,
                        help='Scenarios to include (default: all)')
    parser.add_argument('--robots', nargs='+', type=int, default=None,
                        help='Robot counts to include (default: all)')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for plots (default: experiments/plots/difficulty)')
    parser.add_argument('--no-save-csv', dest='save_csv', action='store_false',
                        help='Skip writing seed_metrics.csv')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (
        project_root / 'experiments' / 'plots' / 'difficulty'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df_summary = pd.read_csv(results_dir / 'summary.csv')
    df_cipher = df_summary[df_summary['method'] == 'geometric_cipher'].copy()
    df_cipher['solved'] = df_cipher['solved'].astype(str).str.lower() == 'true'

    if args.scenarios:
        df_cipher = df_cipher[df_cipher['scenario'].isin(args.scenarios)]
    if args.robots:
        df_cipher = df_cipher[df_cipher['robots'].isin(args.robots)]

    print(f'Loading metrics for {len(df_cipher)} CIPHER runs...')
    df = load_seed_metrics(df_cipher, experiments_root)
    print(f'  {len(df)} rows loaded  '
          f'({df["solved"].sum()} solved, {(~df["solved"]).sum()} failed)')

    if args.save_csv:
        csv_out = results_dir / 'seed_metrics.csv'
        df.to_csv(csv_out, index=False)
        print(f'  Saved {csv_out}')

    # Global ranking
    ranking = analyze(df, METRIC_COLS)
    print('\n── Ranked metrics (all scenarios combined) ─────────────────────────────')
    if not ranking.empty:
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(ranking[['metric', 'solved_median', 'failed_median',
                        'effect_size', 'p_value']].to_string(index=False))

    # Per-scenario ranking
    scenarios = sorted(df['scenario'].unique())
    for scenario in scenarios:
        sub = df[df['scenario'] == scenario]
        if sub['solved'].nunique() < 2:
            continue
        rank = analyze(sub, METRIC_COLS)
        print(f'\n── {scenario} ───────────────────────────────────────────────────────')
        if not rank.empty:
            print(rank[['metric', 'solved_median', 'failed_median',
                         'effect_size', 'p_value']].head(10).to_string(index=False))

    # Plots
    print('\nGenerating plots...')
    plot_violin(df, METRIC_COLS, output_dir, scenarios)
    plot_effect_sizes(ranking, output_dir, ' — all scenarios')
    for scenario in scenarios:
        sub = df[df['scenario'] == scenario]
        if sub['solved'].nunique() < 2:
            continue
        rank = analyze(sub, METRIC_COLS)
        plot_effect_sizes(rank, output_dir, f' — {scenario}')
    plot_corr_heatmap(df, METRIC_COLS, output_dir)
    print('Done.')


if __name__ == '__main__':
    main()
