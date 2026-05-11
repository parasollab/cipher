"""
Plot CIPHER conflict-resolution strategy metrics from per-seed output YAMLs.

For each scenario / num_robots combination this script reads every
results/{scenario}/geometric_cipher/{robots}/{seed}.yaml file, extracts the
resolution_stats block, and produces line-plots (vs. number of robots) of:

  - total_conflicts_encountered / total_conflicts_resolved
  - decomposition_refinement_attempts / decomposition_refinement_successes
  - expansion_attempts / expansion_successes
  - composite_planner_attempts / composite_planner_successes

Usage:
    python plot_resolution_stats.py [--results RESULTS_DIR] [--out OUT_DIR]
                                    [--scenarios S1 S2 ...] [--pdf PATH]
"""

import argparse
import os

import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


STAT_KEYS = [
    "total_conflicts_encountered",
    "total_conflicts_resolved",
    "decomposition_refinement_attempts",
    "decomposition_refinement_successes",
    "expansion_attempts",
    "expansion_successes",
    "composite_planner_attempts",
    "composite_planner_successes",
]

SCENARIOS = [
    "open",
    "rooms",
    "low_clutter",
    "medium_clutter",
    "high_clutter",
    "open_70x70",
]


def collect_data(results_dir: str, scenarios: list[str]) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        cipher_dir = os.path.join(results_dir, scenario, "geometric_cipher")
        if not os.path.isdir(cipher_dir):
            continue
        for robots_str in sorted(os.listdir(cipher_dir), key=lambda x: int(x) if x.isdigit() else 0):
            robots_dir = os.path.join(cipher_dir, robots_str)
            if not os.path.isdir(robots_dir):
                continue
            try:
                robots = int(robots_str)
            except ValueError:
                continue
            for fname in os.listdir(robots_dir):
                if not fname.endswith(".yaml"):
                    continue
                path = os.path.join(robots_dir, fname)
                try:
                    with open(path) as f:
                        data = yaml.safe_load(f)
                except Exception:
                    continue
                if data is None:
                    continue
                row = {
                    "scenario": scenario,
                    "robots": robots,
                    "seed": fname.replace(".yaml", ""),
                    "solved": bool(data.get("solved", False)),
                    "planning_time": data.get("planning_time", float("nan")),
                }
                stats = data.get("resolution_stats", {}) or {}
                for key in STAT_KEYS:
                    row[key] = stats.get(key, 0)
                rows.append(row)
    return pd.DataFrame(rows)


def _panel_attempts(df_scenario: pd.DataFrame, ax: plt.Axes, title_suffix: str = "") -> None:
    palette = sns.color_palette("tab10", 3)
    colors = {
        "decomposition_refinement": palette[0],
        "expansion": palette[1],
        "composite_planner": palette[2],
    }
    labels = {
        "decomposition_refinement": "Refinement/decomp",
        "expansion": "Expansion",
        "composite_planner": "Composite (fallback)",
    }
    for key, label in labels.items():
        col = f"{key}_attempts"
        if col not in df_scenario.columns:
            continue
        sns.lineplot(
            x="robots", y=col,
            data=df_scenario,
            label=label,
            color=colors[key],
            marker="o",
            err_style="bars",
            ax=ax,
        )
    ax.set_title(f"Strategy attempts{title_suffix}")
    ax.set_xlabel("Number of robots")
    ax.set_ylabel("Attempts (mean ± CI)")
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend()


def _panel_successes(df_scenario: pd.DataFrame, ax: plt.Axes, title_suffix: str = "") -> None:
    palette = sns.color_palette("tab10", 3)
    colors = {
        "decomposition_refinement": palette[0],
        "expansion": palette[1],
        "composite_planner": palette[2],
    }
    labels = {
        "decomposition_refinement": "Refinement/decomp",
        "expansion": "Expansion",
        "composite_planner": "Composite (fallback)",
    }
    for key, label in labels.items():
        col = f"{key}_successes"
        if col not in df_scenario.columns:
            continue
        sns.lineplot(
            x="robots", y=col,
            data=df_scenario,
            label=label,
            color=colors[key],
            marker="o",
            err_style="bars",
            ax=ax,
        )
    ax.set_title(f"Strategy successes{title_suffix}")
    ax.set_xlabel("Number of robots")
    ax.set_ylabel("Successes (mean ± CI)")
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend()


def _panel_conflicts(df_scenario: pd.DataFrame, ax: plt.Axes, title_suffix: str = "") -> None:
    if "total_conflicts_encountered" not in df_scenario.columns:
        return
    palette = sns.color_palette("tab10", 2)
    for col, label, color in [
        ("total_conflicts_encountered", "Encountered", palette[0]),
        ("total_conflicts_resolved",    "Resolved",    palette[1]),
    ]:
        sns.lineplot(
            x="robots", y=col,
            data=df_scenario,
            label=label,
            color=color,
            marker="o",
            err_style="bars",
            ax=ax,
        )
    ax.set_title(f"Conflicts{title_suffix}")
    ax.set_xlabel("Number of robots")
    ax.set_ylabel("Count (mean ± CI)")
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend()


def plot_scenario(df: pd.DataFrame, scenario: str, out_dir: str) -> None:
    df_s = df[df["scenario"] == scenario]
    if df_s.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f"CIPHER resolution stats — {scenario}", fontsize=13)
    _panel_conflicts(df_s, axes[0])
    _panel_attempts(df_s, axes[1])
    _panel_successes(df_s, axes[2])
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{scenario}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


def generate_pdf(df: pd.DataFrame, scenarios: list[str], pdf_path: str) -> None:
    with pdf_backend.PdfPages(pdf_path) as pdf:
        for scenario in scenarios:
            df_s = df[df["scenario"] == scenario]
            if df_s.empty:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(20, 5))
            fig.suptitle(f"CIPHER resolution stats — {scenario}", fontsize=13)
            _panel_conflicts(df_s, axes[0])
            _panel_attempts(df_s, axes[1])
            _panel_successes(df_s, axes[2])
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved PDF to {pdf_path}")


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_results = os.path.join(project_root, "experiments", "results")
    default_out = os.path.join(project_root, "experiments", "plots", "resolution_stats")
    default_pdf = os.path.join(project_root, "experiments", "plots", "resolution_stats.pdf")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=default_results, help="Path to results directory")
    parser.add_argument("--out",     default=default_out,     help="Directory for per-scenario PNG files")
    parser.add_argument("--pdf",     default=default_pdf,     help="Path for combined PDF output")
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS, help="Scenarios to include")
    args = parser.parse_args()

    df = collect_data(args.results, args.scenarios)
    if df.empty:
        print("No resolution_stats found in any output file. "
              "Re-run experiments with the updated planner to populate this data.")
        return

    present = [s for s in args.scenarios if s in df["scenario"].values]
    print(f"Found data for scenarios: {present}")
    print(f"Total rows: {len(df)}, rows with resolution_stats: {(df['total_conflicts_encountered'] > 0).sum()}")

    os.makedirs(args.out, exist_ok=True)
    for scenario in present:
        plot_scenario(df, scenario, args.out)

    generate_pdf(df, present, args.pdf)


if __name__ == "__main__":
    main()
