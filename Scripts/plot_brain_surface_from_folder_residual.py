#!/usr/bin/env python3
"""
Derived from Xiao Yu's original notebook: plot_surface_brain.ipynb

Plot static 4-view brain surface maps from an experiment result folder.

This is the from-folder version of `plot_brain_surface.py`, following the
same input style as `plot_fig2_from_folder.py`.

The script reads the prediction CSV produced under one experiment result folder
and saves a 4-view cortical surface figure into that folder's `plots/`
subdirectory.

Default required input file under the experiment folder:
- Fig2_Best_predictions.csv

The output is a PNG image named like:
- plots/Brain_surface_YYYYMMDD_HHMMSS.png

Usage:
    python plot_brain_surface_from_folder.py \
        --input_dir /path/to/experiment/folder \
        --atlas-path "E:/01_jiaoyq/02_Project_management/005_SIR_sim/Brain_surface_plot"

Optional:
    python plot_brain_surface_from_folder.py \
        --input_dir /path/to/experiment/folder \
        --atlas-path "E:/.../Brain_surface_plot" \
        --csv-name Fig2_Best_predictions.csv \
        --value-col exp18_ReplacedInput_mean_scaled
"""

import argparse
import math
import os
from datetime import datetime
from pathlib import Path
import warnings
from matplotlib.colors import LinearSegmentedColormap
import abagen
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from nilearn import image, plotting, regions


def check_required_files(input_dir: Path, csv_name: str) -> dict[str, Path]:
    """Check whether the prediction CSV exists in the experiment folder."""
    required = {
        "best_predictions": input_dir / csv_name,
    }

    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The following required files are missing:\n" + "\n".join(missing)
        )

    return required


def load_prediction_data(input_dir: Path, csv_name: str, index_col: int) -> pd.DataFrame:
    """Load the prediction CSV from the experiment folder."""
    files = check_required_files(input_dir, csv_name)
    return pd.read_csv(files["best_predictions"], index_col=index_col)


def get_min_max_for_plot(data: np.ndarray) -> tuple[float, float]:
    """Round min/max to a readable plotting range."""
    data = np.asarray(data, dtype=float)
    epsilon = 1e-20

    min_val = max(float(np.nanmin(data)), epsilon)
    max_val = max(float(np.nanmax(data)), epsilon)

    order_of_magnitude = math.floor(math.log10(min_val))
    factor = 10 ** (-order_of_magnitude)
    range_min = math.floor(min_val * factor) / factor

    order_of_magnitude = math.floor(math.log10(max_val))
    factor = 10 ** (-order_of_magnitude)
    range_max = math.ceil(max_val * factor) / factor

    print(f"Auto range from data: {min_val} --> {range_min}     {max_val} --> {range_max}")
    return range_min, range_max


def infer_title_from_input_dir(input_dir: Path) -> str:
    """
    Infer display title from experiment folder name.

    Examples:
    exp1_merged_connectomes_76_presence_20260424_200344
        -> full_presence

    exp12_merged_connectomes_76_drop_Thalamusproper_load_20260427_005546
        -> No_Thalamusproper_load
    """
    folder_name = input_dir.name
    parts = folder_name.split("_")
    parts_lower = [p.lower() for p in parts]
    
    if "load" in parts_lower:
        protein_type = "load"
    elif "presence" in parts_lower:
        protein_type = "presence"
    else:
        print("[WARN] Cannot detect load/presence from folder name; fallback to folder name.")
        return folder_name

    if "drop" in parts_lower:
        drop_idx = parts_lower.index("drop")
        type_idx = parts_lower.index(protein_type)
        dropped_region = "_".join(parts[drop_idx + 1:type_idx])

        if dropped_region:
            return f"No_{dropped_region}"
        else:
            return f"No_unknown"    

    return "full model"


def format_title_name(title_name: str) -> str:
    """Make title key display-friendly."""
    return title_name.replace("_", " ").title()


def infer_tau_range_from_input_dir(input_dir: Path) -> tuple[float | None, float | None]:
    """Infer plotting range from the folder path"""

    path_lower = str(input_dir).lower()

    if "/presence/" in path_lower or "\\presence\\" in path_lower:
        print("Detected Presence from input_dir -> use vmin=0.1, vmax=0.4")
        return 0.1, 0.4

    if "/load/" in path_lower or "\\load\\" in path_lower:
        print("Detected Load from input_dir -> use vmin=0.4, vmax=1.0")
        return 0.4, 1.0

    print("No Presence/Load keyword detected in input_dir; fallback to data-driven range.")
    return None, None


def infer_value_col(dat_pred: pd.DataFrame, value_col: str | None = None) -> str:
    """
    Infer which column to plot.

    Priority:
    1. Use --value-col if provided.
    2. Otherwise use the 4th data column, matching the previous script behavior.
    """
    if value_col is not None:
        if value_col not in dat_pred.columns:
            raise ValueError(
                f"Requested value column not found: {value_col}\n"
                f"Available columns: {list(dat_pred.columns)}"
            )
        return value_col

    if len(dat_pred.columns) < 1:
        raise ValueError("No data columns found in CSV.")

    return dat_pred.columns[-1]




def infer_observation_col(dat_pred: pd.DataFrame, observation_col: str | None = None) -> str:
    """
    Infer which observation column to use for residual plotting.

    Priority:
    1. Use --observation-col if provided.
    2. Otherwise try common observation column names.
    3. If no common name is found, use the first data column.
    """
    if observation_col is not None:
        if observation_col not in dat_pred.columns:
            raise ValueError(
                f"Requested observation column not found: {observation_col}\n"
                f"Available columns: {list(dat_pred.columns)}"
            )
        return observation_col

    common_obs_cols = [
        "Presence", "Load",
    ]
    for col in common_obs_cols:
        if col in dat_pred.columns:
            return col

    if len(dat_pred.columns) < 2:
        raise ValueError(
            "Residual plotting needs at least observation and prediction columns. "
            "Please provide --observation-col explicitly."
        )

    print(
        f"[WARN] Cannot identify observation column by name; fallback to first column: {dat_pred.columns[0]}"
    )
    return dat_pred.columns[0]


def match_desikan_label(df_label: pd.DataFrame | str) -> pd.DataFrame:
    """Match atlas labels to SC-style ROI labels."""
    warnings.filterwarnings("ignore")

    if isinstance(df_label, pd.DataFrame):
        atlas_info = df_label.copy()
    else:
        atlas_info = pd.read_csv(df_label)

    atlas_info.loc[atlas_info["structure"] != "cortex", "label"] = (
        atlas_info.loc[atlas_info["structure"] != "cortex", "label"].str.capitalize()
    )

    atlas_info["label_hemisphere"] = atlas_info["hemisphere"] + "_" + atlas_info["label"]

    atlas_info.loc[atlas_info["structure"] == "cortex", "label_hemisphere"] = (
        atlas_info.loc[atlas_info["structure"] == "cortex", "label_hemisphere"]
        .str.replace("L_", "ctx_lh_", regex=False)
        .str.replace("R_", "ctx_rh_", regex=False)
    )

    atlas_info.loc[atlas_info["structure"] != "cortex", "label_hemisphere"] = (
        atlas_info.loc[atlas_info["structure"] != "cortex", "label_hemisphere"]
        .str.replace("L_", "Left_", regex=False)
        .str.replace("R_", "Right_", regex=False)
    )

    return atlas_info


def get_atlas(atlas_path: str, mod_cols_name: str = "mod_cols.npy"):
    """Load atlas image and map ROI labels to atlas indices."""
    mod_cols_path = os.path.join(atlas_path, mod_cols_name)
    if not os.path.exists(mod_cols_path):
        raise FileNotFoundError(f"mod-cols file does not exist: {mod_cols_path}")

    cols = np.load(mod_cols_path, allow_pickle=True)
    cols_m = [str(item).replace("-", "_") for item in cols]

    atlas = abagen.fetch_desikan_killiany()
    atlas_info = match_desikan_label(atlas["info"])

    missing_labels = [label for label in cols_m if label not in set(atlas_info["label_hemisphere"])]
    if missing_labels:
        raise ValueError(
            "Some labels in mod_cols could not be matched to the Desikan atlas:\n"
            + "\n".join(missing_labels[:20])
            + (f"\n... and {len(missing_labels) - 20} more" if len(missing_labels) > 20 else "")
        )

    region_index = [
        atlas_info.index[atlas_info["label_hemisphere"] == label][0]
        for label in cols_m
    ]
    atlas_img = image.load_img(atlas["image"])
    roi_num = len(atlas_info)

    return atlas_img, region_index, roi_num, atlas_info


def plot_brain_surf_4views(
    data_regions,
    atlas_img,
    region_index,
    roi_num,
    vmax=None,
    vmin=None,
    cmap="hot",
    find=True,
    output_path=None,
    symmetric_cbar=False,
    title=None,
    show=False,
):
    """
    Plot static 4-view cortical surface maps:
    left lateral, left medial, right lateral, right medial.
    """
    data_regions = np.asarray(data_regions, dtype=float)

    if len(data_regions) != len(region_index):
        raise ValueError(
            f"Length mismatch: data has {len(data_regions)} values, "
            f"but region_index has {len(region_index)} entries.\n"
            f"This usually means the plotted CSV and mod_cols.npy are not using the same ROI set."
        )

    region_signals = np.zeros((roi_num,), dtype=float)
    region_signals[region_index] = data_regions

    img_maps = regions.signals_to_img_labels(
        region_signals.reshape(1, -1),
        atlas_img,
    )

    if vmax is None and vmin is None and find:
        vmin, vmax = get_min_max_for_plot(data_regions)

    fig, axes = plotting.plot_img_on_surf(
        stat_map=img_maps,
        hemispheres=["left", "right"],
        views=["lateral", "medial"],
        cmap=cmap,
        colorbar=True,
        vmin=vmin,
        vmax=vmax,
        symmetric_cbar=symmetric_cbar,
    )

    cbar_ax = axes[-1]
    if vmin is not None and vmax is not None:
        cbar_ax.set_xlim(vmin, vmax)
        ticks = np.linspace(vmin, vmax, 2)
        cbar_ax.set_xticks(ticks)
        cbar_ax.set_xticklabels([f"{t:.2f}" for t in ticks])
    cbar_ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=16)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate static 4-view surface brain plots from an experiment folder."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Absolute path to the experiment result folder.",
    )
    parser.add_argument(
        "--atlas-path",
        required=True,
        help="Directory containing mod_cols.npy or another ROI label .npy file.",
    )
    parser.add_argument(
        "--csv-name",
        default="Fig2_Best_predictions.csv",
        help="Prediction CSV filename under --input_dir. Default: Fig2_Best_predictions.csv.",
    )
    parser.add_argument(
        "--value-col",
        default=None,
        help="Column to plot. If omitted, the 4th data column is used.",
    )
    parser.add_argument(
        "--mod-cols",
        default="mod_cols.npy",
        help="ROI label .npy filename under --atlas-path. Default: mod_cols.npy.",
    )
    parser.add_argument("--cmap", default="hot", help="Matplotlib/nilearn colormap.")
    parser.add_argument(
        "--symmetric-cbar",
        action="store_true",
        help="Use symmetric colorbar around zero.",
    )
    parser.add_argument(
        "--draw-residual",
        action="store_true",
        help=(
            "Plot residuals instead of predictions. Residual = prediction - observation. "
            "When enabled, the plot uses a blue-white-red colormap and fixed range -0.3 to 0.3."
        ),
    )
    parser.add_argument(
        "--observation-col",
        default=None,
        help=(
            "Observation column used only when --draw-residual is enabled. "
            "If omitted, the script tries common observation column names, then falls back to the first column."
        ),
    )
    parser.add_argument(
        "--index-col",
        default=0,
        type=int,
        help="CSV index column. Default: 0.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively after saving. Default: save and close.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir = input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    dat_pred = load_prediction_data(
        input_dir=input_dir,
        csv_name=args.csv_name,
        index_col=args.index_col,
    )

    print("Columns:", list(dat_pred.columns))
    print(dat_pred.head())

    title_name = infer_title_from_input_dir(input_dir)
    display_title = format_title_name(title_name)
    value_col = infer_value_col(dat_pred, args.value_col)

    if args.draw_residual:
        observation_col = infer_observation_col(dat_pred, args.observation_col)
        plot_mode = "residual"
        vmin, vmax = -0.3, 0.3
        # cmap = "bwr"
        cmap = LinearSegmentedColormap.from_list(
            "residual_map",
                [
                    (0.00, "#00d5ff"),
                    (0.12, "#33ccff"),
                    (0.24, "#0066ff"),
                    (0.36, "#0033aa"),
                    (0.46, "#111111"),
                    (0.50, "#000000"),
                    (0.54, "#220000"),
                    (0.64, "#770000"),
                    (0.76, "#ff0000"),
                    (0.88, "#ff8800"),
                    (1.00, "#ffff00"),
                ]
        )
        symmetric_cbar = True
        display_title = f"{display_title} Residual"
    else:
        observation_col = None
        plot_mode = "prediction"
        vmin, vmax = infer_tau_range_from_input_dir(input_dir)
        cmap = args.cmap
        symmetric_cbar = args.symmetric_cbar

    out_date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"Brain_surface_{plot_mode}_{out_date_time}.png"

    print(f"[INFO] title_name: {title_name}")
    print(f"[INFO] plot_mode: {plot_mode}")
    print(f"[INFO] value_col: {value_col}")
    if observation_col is not None:
        print(f"[INFO] observation_col: {observation_col}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] output_path: {output_path}")

    atlas_img, region_index, roi_num, _ = get_atlas(
        atlas_path=args.atlas_path,
        mod_cols_name=args.mod_cols,
    )

    if args.draw_residual:
        data_regions = (dat_pred[value_col] - dat_pred[observation_col]).to_numpy()
    else:
        data_regions = dat_pred[value_col].to_numpy()

    plot_brain_surf_4views(
        data_regions=data_regions,
        atlas_img=atlas_img,
        region_index=region_index,
        roi_num=roi_num,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        symmetric_cbar=symmetric_cbar,
        title=display_title,
        output_path=output_path,
        show=args.show,
    )

    print(f"[INFO] Figure saved: {output_path}")


if __name__ == "__main__":
    main()
