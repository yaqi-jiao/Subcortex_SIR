#!/usr/bin/env python3
"""
Derived from Xiao Yu's original notebook: plot_surface_brain.ipynb
Generate a static 4-view brain surface figure
(left lateral / left medial / right lateral / right medial)
from the 4th column in a CSV file.

Behavior
--------
1. Automatically detects tau type from CSV path / filename:
   - contains 'tau_present'  -> vmin=0.1, vmax=0.4
   - contains 'tau_load'     -> vmin=0.4, vmax=1.0
2. Automatically uses the 4th column in the CSV as plotting values.
3. Automatically creates output filename with current datetime.
4. Automatically creates figure title from the last field of the CSV filename.

Example
-------
python plot_surface_brain_4views.py \
    --csv exp19_tau_load_full.csv \
    --atlas-path "E:/01_jiaoyq/02_Project_management/005_SIR_sim/Brain_surface_plot" \
    --mod-cols mod_cols.npy \
    --output-dir ./plots
"""

import argparse
import math
import os
from datetime import datetime
from pathlib import Path
import warnings

import abagen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import image, plotting, regions


def get_min_max_for_plot(data: np.ndarray) -> tuple[float, float]:
    """Round min/max to a readable plotting range."""
    data = np.asarray(data, dtype=float)
    epsilon = 1e-20

    min_val = max(float(np.min(data)), epsilon)
    max_val = max(float(np.max(data)), epsilon)

    order_of_magnitude = math.floor(math.log10(min_val))
    factor = 10 ** (-order_of_magnitude)
    range_min = math.floor(min_val * factor) / factor

    order_of_magnitude = math.floor(math.log10(max_val))
    factor = 10 ** (-order_of_magnitude)
    range_max = math.ceil(max_val * factor) / factor

    print(f"Auto range from data: {min_val} --> {range_min}     {max_val} --> {range_max}")
    return range_min, range_max


def infer_tau_range_from_csv_path(csv_path: str) -> tuple[float | None, float | None]:
    """Infer plotting range from CSV path / filename keywords."""
    csv_path_lower = str(csv_path).lower()

    if "tau_present" in csv_path_lower:
        print("Detected tau_present from CSV path -> use vmin=0.1, vmax=0.4")
        return 0.1, 0.4
    if "tau_load" in csv_path_lower:
        print("Detected tau_load from CSV path -> use vmin=0.4, vmax=1.0")
        return 0.4, 1.0

    print("No tau_present / tau_load keyword detected in CSV path; fallback to data-driven range.")
    return None, None


def get_auto_title(csv_path: str) -> str:
    """Use the first field of the CSV filename (before first underscore) as title."""
    stem = Path(csv_path).stem
    return stem.split("_")[-1]


def get_auto_output_path(csv_path: str, output_dir: str) -> Path:
    """Create output filename from CSV stem + datetime."""
    stem = Path(csv_path).stem
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{stem}_{dt}.png"
    return Path(output_dir) / filename


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
    cols = np.load(os.path.join(atlas_path, mod_cols_name), allow_pickle=True)
    cols_m = [str(item).replace("-", "_") for item in cols]

    atlas = abagen.fetch_desikan_killiany()
    atlas_info = match_desikan_label(atlas["info"])

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
):
    """
    Plot static 4-view cortical surface maps:
    left lateral, left medial, right lateral, right medial.
    """
    data_regions = np.asarray(data_regions, dtype=float)

    if len(data_regions) != len(region_index):
        raise ValueError(
            f"Length mismatch: data has {len(data_regions)} values, "
            f"but region_index has {len(region_index)} entries."
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

    plt.show()
    return fig, axes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate static 4-view surface brain plots from the 4th column in a CSV file."
    )
    parser.add_argument("--csv", required=True, help="Input CSV file.")
    parser.add_argument(
        "--atlas-path",
        required=True,
        help="Directory containing mod_cols.npy (or another ROI label .npy file).",
    )
    parser.add_argument(
        "--mod-cols",
        default="mod_cols.npy",
        help="ROI label .npy filename under --atlas-path.",
    )
    parser.add_argument(
        "--output-dir",
        default="./plots",
        help="Directory for output PNG. Filename will be auto-generated with datetime.",
    )
    parser.add_argument("--cmap", default="hot", help="Matplotlib/nilearn colormap.")
    parser.add_argument(
        "--symmetric-cbar",
        action="store_true",
        help="Use symmetric colorbar around zero.",
    )
    parser.add_argument(
        "--index-col",
        default=0,
        type=int,
        help="CSV index column (default: 0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv, index_col=args.index_col)
    print("Columns:", list(df.columns))
    print(df.head())

    # if len(df.columns) < 4:
    #     raise ValueError(
    #         f"The CSV must contain at least 4 data columns after reading index_col={args.index_col}, "
    #         f"but only {len(df.columns)} column(s) were found."
    #     )

    value_col = df.columns[0]
    title = get_auto_title(args.csv)
    vmin, vmax = infer_tau_range_from_csv_path(args.csv)
    output_path = get_auto_output_path(args.csv, args.output_dir)

    print(f"Auto-selected value column (4th column): {value_col}")
    print(f"Auto title: {title}")
    print(f"Auto output path: {output_path}")

    atlas_img, region_index, roi_num, _ = get_atlas(
        atlas_path=args.atlas_path,
        mod_cols_name=args.mod_cols,
    )

    data_regions = df[value_col].to_numpy()

    plot_brain_surf_4views(
        data_regions=data_regions,
        atlas_img=atlas_img,
        region_index=region_index,
        roi_num=roi_num,
        vmin=vmin,
        vmax=vmax,
        cmap=args.cmap,
        symmetric_cbar=args.symmetric_cbar,
        title=title,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
