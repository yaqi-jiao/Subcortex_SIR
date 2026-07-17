#!/usr/bin/env python3

"""
Plot Fig2-style results from an experiment output folder.

This script reads experiment summary CSV files produced by `run_summary_one_conn.py` run and
renders a combined diagnostic figure in the folder's `plots/` subdirectory.

Required input files under the experiment folder:
- Fig2_R_across_time.csv
- Fig2_Best_predictions.csv
- Fig2_Metrics.csv

The output is a PNG image named like:
- plots/Fig2_combined_YYYYMMDD_HHMMSS.png

Usage:
    python plot_scatter_from_folder.py --input_dir /path/to/experiment/folder
"""

import argparse
from pathlib import Path
import sys
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


from datetime import datetime


def check_required_files(input_dir: Path) -> dict[str, Path]:
    """Check whether all required files exist."""
    required = {
        "r_across_time": input_dir / "Fig2_R_across_time.csv",
        "best_predictions": input_dir / "Fig2_Best_predictions.csv",
        "metrics": input_dir / "Fig2_Metrics.csv",
    }

    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The following required files are missing:\n" + "\n".join(missing)
        )

    return required


def load_data(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required csv files."""
    files = check_required_files(input_dir)

    dat_r = pd.read_csv(files["r_across_time"], index_col=0)
    dat_pred = pd.read_csv(files["best_predictions"], index_col=0)
    dat_metrics = pd.read_csv(files["metrics"], index_col=0)

    return dat_r, dat_pred, dat_metrics


def infer_prediction_columns(dat_pred: pd.DataFrame):
    """
    Infer observed and simulated prediction columns from the best-predictions table.

    The input file is expected to contain at least three columns.
    By convention this script selects:
        - observed values from the first column
        - simulated values from the third column

    Args:
        dat_pred: DataFrame loaded from Fig2_Best_predictions.csv.

    Returns:
        A tuple of (observed, simulated) pandas Series.
    """
    observed = dat_pred.iloc[:, 0]
    simulated = dat_pred.iloc[:, 2]
    return observed, simulated


def infer_metrics(dat_metrics: pd.DataFrame):
    """
    Read evaluation metrics from the metrics CSV.

    Args:
        dat_metrics: DataFrame loaded from Fig2_Metrics.csv.

    Returns:
        A tuple of (R, MSE) values as floats.
    """
    r_val = float(dat_metrics["R"].iloc[0])
    mse_val = float(dat_metrics["MSE"].iloc[0])
    return r_val, mse_val


def infer_r_col(dat_r: pd.DataFrame) -> str:
    """
    Infer the R-value column name from the time-series CSV.

    The input file is expected to contain a single data column after
    the index column. This function returns that first data column name.

    Args:
        dat_r: DataFrame loaded from Fig2_R_across_time.csv.

    Returns:
        The name of the R column.
    """
    r_col = dat_r.columns[0]
    return r_col


def compute_spearman(observed, simulated):
    rho, pval = spearmanr(observed, simulated)
    return float(rho), float(pval)


def infer_title_from_input_dir(input_dir: Path) -> str:
    """
    Infer a plot title key from the experiment folder path.

    The function expects the folder path to follow the convention:
        .../results/<simulated_protein>/<protein_type>/...

    Args:
        input_dir: Experiment result directory path.

    Returns:
        A short title key such as 'tau_presence' or 'tau_load'.
    """
    parts = input_dir.parts

    if "results" not in parts:
        raise ValueError(f"'results' not found in path: {input_dir}")

    lower_parts = [p.lower() for p in parts]

    results_idx = lower_parts.index("results")

    simulated_protein = parts[results_idx + 1]   # tau
    protein_type = parts[results_idx + 2]        # load

    exp_name = input_dir.name.lower()
    if "baseline" in exp_name:
        condition = "full_model"

        lower_parts = [p.lower() for p in input_dir.parts]

    results_idx = lower_parts.index("results")
    protein_type = lower_parts[results_idx + 2]  # gives "load", not "Load"

    exp_name = input_dir.name.lower()

    if "baseline" in exp_name:
        condition = "full_model"
    elif "spread" in exp_name and "mapt" in exp_name:
        condition = "spreadMAPT"
    elif "spread" in exp_name and "apoe" in exp_name:
        condition = "spreadAPOE"
    elif "synthesis" in exp_name and "mapt" in exp_name:
        condition = "synMAPT"
    elif "synthesis" in exp_name and "apoe" in exp_name:
        condition = "synAPOE"
    elif "clearance" in exp_name and "mapt" in exp_name:
        condition = "clearMAPT"
    elif "clearance" in exp_name and "apoe" in exp_name:
        condition = "clearAPOE"
    elif "misfold" in exp_name and "mapt" in exp_name:
        condition = "misMAPT"
    elif "misfold" in exp_name and "apoe" in exp_name:
        condition = "misAPOE"
    else:
        condition = exp_name

    return condition, protein_type

    # Expect:
    # results / simulated_protein / protein_type / ...
    # simulated_protein = lower_parts[results_idx + 1]
    # protein_type = lower_parts[results_idx + 2]
    # condition = lower_parts[results_idx + 5]
    # parts = condition.split("_")
    # idx_76 = parts.index("68")  # need to modify based on the result path details
    # idx_protein_type = parts.index(protein_type)
    # condition_formatted = "_".join(parts[idx_76 + 1 : idx_protein_type])

    # if condition_formatted == "":
    #     condition_formatted = "full_model"
    # elif condition_formatted == "drop_all":
    #     condition_formatted = "cortex_only"

    # # title_name = f"{simulated_protein}_{protein_type.lower()}"
    # title_name = condition_formatted
    # return title_name, protein_type


def format_title_name(title_name: str) -> str:
    return title_name.replace("_", " ").title()


def get_plot_colors(protein_type: str) -> dict:
    if protein_type == "presence":
        return {
            "color": "#36A39D",
            "line_kws": {"color": "#2C8480", "linewidth": 1},
        }
    elif protein_type == "load":
        return {
            "color": "#969C9C",
            "line_kws": {"color": "#4B4D4D", "linewidth": 1},
        }
    else:
        raise ValueError(f"Unsupported protein type: {protein_type}")


def plot_combined(
    dat_r,
    r_col,
    observed,
    simulated,
    r_val,
    mse_val,
    rho_val,
    title_name,
    protein_type,
    output_path):
    """
    Create the combined Fig2-style figure and save it to disk.

    The figure contains:
      - a scatter/regression plot of observed vs simulated values
      - a time series plot of R across time

    Args:
        dat_r: DataFrame with R values across time.
        r_col: Column name or index for the R series in dat_r.
        observed: Observed data series for the scatter plot.
        simulated: Simulated data series for the scatter plot.
        r_val: Pearson R value to display in the plot.
        mse_val: Mean squared error value to display in the plot.
        title_name: Title key indicating tau presence/load mode.
        output_path: Path prefix for the saved PNG file.
    """
    # Global style for presentation
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.titlesize": 20,
        "axes.labelsize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    colors = get_plot_colors(protein_type)
    display_title = format_title_name(title_name)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(11.5, 4.8),
        gridspec_kw={"wspace": 0.28}
    )

    # Helper: cleaner axis styling
    def style_axis(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.tick_params(axis="both", which="major", length=4, width=1)
        ax.grid(False)

    # Left panel: scatter + regression
    ax = axes[0]

    sns.regplot(
        x=observed,
        y=simulated,
        ax=ax,
        color=colors["color"],
        ci=95,
        scatter_kws={
            "alpha": 0.85,
            "s": 28,
            "edgecolor": "none",
        },
        line_kws={
            "linewidth": 1.8,
        },
    )

    # Set comparable x/y limits
    observed_arr = np.asarray(observed, dtype=float)
    simulated_arr = np.asarray(simulated, dtype=float)
    all_vals = np.concatenate([observed_arr, simulated_arr])

    vmin = max(0, np.nanmin(all_vals) - 0.03)
    vmax = np.nanmax(all_vals) + 0.03

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)

    # # Identity line (y=x)
    # ax.plot(
    #     [vmin, vmax],
    #     [vmin, vmax],
    #     linestyle="--",
    #     linewidth=1.2,
    #     color="0.7",
    #     zorder=0,
    # )

    ax.set_title(display_title, pad=10, fontweight="normal")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Simulated")

    # Metrics box
    # ax.text(
    #     0.04,
    #     0.96,
    #     f"R = {r_val:.2f}\nMSE* = {mse_val:.2f}",
    #     transform=ax.transAxes,
    #     ha="left",
    #     va="top",
    #     fontsize=15,
    #     bbox=dict(
    #         boxstyle="round,pad=0.28",
    #         facecolor="white",
    #         edgecolor="0.85",
    #         linewidth=0.8,
    #         alpha=0.95,
    #     ),
    # )

# bbox version
    ax.text(
    0.04, 0.96,
    f"Pearson r = {r_val:.4f}\nSpearman ρ = {rho_val:.4f}\nMSE* = {mse_val:.6f}",
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=10,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor="0.8",
        linewidth=0.8,
        alpha=0.95,
    ),
)


    style_axis(ax)

    # Right panel: R across time
    ax = axes[1]

    xvals = np.asarray(dat_r.index)
    yvals = np.asarray(dat_r[r_col] if isinstance(r_col, str) else r_col, dtype=float)

    sns.lineplot(
        x=xvals,
        y=yvals,
        ax=ax,
        color=colors["color"],
        linewidth=2.0,
    )

    # highlight peak R
    peak_idx = int(np.nanargmax(yvals))
    peak_x = xvals[peak_idx]
    peak_y = yvals[peak_idx]

    ax.scatter(
        peak_x,
        peak_y,
        s=36,
        color=colors["color"],
        zorder=3,
    )

    ax.annotate(
        f"Peak R = {peak_y:.2f}",
        xy=(peak_x, peak_y),
        xytext=(10, 8),
        textcoords="offset points",
        fontsize=11,
        ha="left",
        va="bottom",
    )

    # optional: thin reference line at y=0
    # ax.axhline(0, color="0.8", linewidth=1.0, linestyle="--", zorder=0)

    ax.set_title("R across time", pad=10, fontweight="normal")
    ax.set_xlabel("Time")
    ax.set_ylabel("R value")

    # add a bit of vertical breathing room
    y_min = np.nanmin(yvals)
    y_max = np.nanmax(yvals)
    y_pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    style_axis(ax)

    # Save
    plt.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot Fig2-style figures from an experiment folder."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Absolute path to the experiment result folder",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir = input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    dat_r, dat_pred, dat_metrics = load_data(input_dir)

    title_name, protein_type = infer_title_from_input_dir(input_dir)
    r_col = infer_r_col(dat_r)

    observed, simulated = infer_prediction_columns(dat_pred)
    r_val, mse_val = infer_metrics(dat_metrics)
    rho_val, rho_p = compute_spearman(observed, simulated)

    out_data_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_out = output_dir / f"Fig2_combined_{out_data_time}"

    plot_combined(
        dat_r=dat_r,
        r_col=r_col,
        observed=observed,
        simulated=simulated,
        r_val=r_val,
        mse_val=mse_val,
        rho_val = rho_val,
        title_name=title_name,
        protein_type=protein_type,
        output_path=combined_out,
    )

    print(f"[INFO] title_name: {title_name}")
    print(f"[INFO] r_col: {r_col}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Figure saved: {combined_out.with_suffix('.png')}")

if __name__ == "__main__":
    main()