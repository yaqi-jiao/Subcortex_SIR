#!/usr/bin/env python3

"""
Reduce an SIR input pickle file by dropping specified brain regions.

This script loads a model input pickle file containing tau, connectivity,
region names, ROI sizes, and SC length matrices. It removes regions whose
names are listed in `--drop-names`, then saves a reduced pickle file with
all relevant data structures updated consistently.

Expected input structure:
    data["tau"]         -> pandas DataFrame indexed by region names
    data["conn"]["conn"] -> connectivity matrix (n x n)
    data["conn"]["name"] -> list of region names
    data["conn"]["SC_len"] -> SC length matrix (n x n)
    data["conn"]["ROI_size"] -> dict mapping region name -> size

Usage:
    python reduce_input_pkl_by_names.py \
        --input-pkl path/to/input.pkl \
        --output-pkl path/to/output.pkl \
        --drop-names region1 region2 ...

Optional flags:
    --show-all-names           print all region names before reduction
    --allow-missing-drop-names  ignore drop names that are not present
"""

import argparse
import pickle
from pathlib import Path
import random

import numpy as np
import pandas as pd

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Reduce SIR input pkl by dropping regions using names."
    )
    parser.add_argument("--input-pkl", required=True, type=str)
    # parser.add_argument("--output-pkl", required=True, type=str)
    parser.add_argument("--drop-names", nargs="+", required=True)
    parser.add_argument("--drop-names-txt", type=str, default=None)
    parser.add_argument("--show-all-names", action="store_true")
    parser.add_argument("--allow-missing-drop-names", action="store_true")
    return parser.parse_args()


def load_input_pkl(input_pkl_path):
    """Load input pickle file."""
    input_pkl_path = Path(input_pkl_path)

    with open(input_pkl_path, "rb") as f:
        data = pickle.load(f)

    print("Top-level keys:", list(data.keys()))
    print("conn subkeys:", list(data["conn"].keys()))
    return data


def load_drop_names_from_txt(txt_path):
    """
    Load drop names from a text file (one name per line).
    and expand each into Left_ and Right_ versions
    """
    txt_path = Path(txt_path)
    if not txt_path.is_file():
        raise ValueError(f"Drop names txt file not found: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        base_names = [line.strip() for line in f if line.strip()]
    
    drop_names = []
    for name in base_names:
        drop_names.append(f"Left_{name}")
        drop_names.append(f"Right_{name}")

    print("Base names:", base_names)
    print("Expanded drop names:", drop_names)
    return drop_names


def extract_core_data(data):
    """Extract core data structures from loaded pickle."""
    tau_df = data["tau"]
    conn = np.asarray(data["conn"]["conn"])
    names = list(data["conn"]["name"])
    roi_size = data["conn"]["ROI_size"]
    sc_len = np.asarray(data["conn"]["SC_len"])

    return {
        "tau_df": tau_df,
        "conn": conn,
        "names": names,
        "roi_size": roi_size,
        "sc_len": sc_len,
    }


def basic_information_check(extracted, show_all_names=False):
    """Print basic information of original data."""
    tau_df = extracted["tau_df"]
    conn = extracted["conn"]
    names = extracted["names"]
    roi_size = extracted["roi_size"]
    sc_len = extracted["sc_len"]

    print("\n*** Original Data Info ***")
    print("tau shape:", tau_df.shape)
    print("conn shape:", conn.shape)
    print("SC_len shape:", sc_len.shape)
    print("len(names):", len(names))
    print("ROI_size len:", len(roi_size))

    if show_all_names:
        print("\n*** tau index ***")
        for i, name in enumerate(tau_df.index):
            print(i, name)

        print("\n*** conn names ***")
        for i, name in enumerate(names):
            print(i, name)


def alignment_check(data, stage="Current"):
    """Check alignment among conn names, tau index, and ROI_size keys."""
    names = list(data["conn"]["name"])
    tau_index = list(data["tau"].index)
    roi_keys = list(data["conn"]["ROI_size"].keys())

    print(f"\n*** {stage} Alignment Check ***")
    print("name == tau.index:", names == tau_index)
    print("name == ROI_size.keys:", names == roi_keys)
    print("conn shape == SC_len shape:",
          np.asarray(data["conn"]["conn"]).shape == np.asarray(data["conn"]["SC_len"]).shape)

    return {
        "name_eq_tau": names == tau_index,
        "name_eq_roi": names == roi_keys,
        "conn_eq_sc_len": np.asarray(data["conn"]["conn"]).shape == np.asarray(data["conn"]["SC_len"]).shape,
    }


def find_drop_indices(names, drop_names, allow_missing_drop_names=False):
    """Find indices to drop based on region names."""
    drop_idx = [i for i, n in enumerate(names) if n in drop_names]

    if not allow_missing_drop_names:
        missing = [n for n in drop_names if n not in names]
        if missing:
            raise ValueError(f"These drop_names not found: {missing}")

    print("\n*** Drop Info ***")
    print("drop_idx:", drop_idx)
    print("drop_names found:", [names[i] for i in drop_idx])

    keep_idx = [i for i in range(len(names)) if i not in drop_idx]

    return drop_idx, keep_idx


def apply_modifications(data, extracted, keep_idx):
    """Apply region dropping to tau, conn, SC_len, name, ROI_size."""
    conn = extracted["conn"]
    sc_len = extracted["sc_len"]
    names = extracted["names"]
    old_roi = extracted["roi_size"]

    # tau
    data["tau"] = data["tau"].iloc[keep_idx]
    # conn
    data["conn"]["conn"] = conn[np.ix_(keep_idx, keep_idx)]
    # SC_len
    data["conn"]["SC_len"] = sc_len[np.ix_(keep_idx, keep_idx)]
    # names
    new_names = [names[i] for i in keep_idx]
    data["conn"]["name"] = new_names
    # ROI_size
    data["conn"]["ROI_size"] = {name: old_roi[name] for name in new_names}

    print("\n*** After Modification ***")
    print("tau:", data["tau"].shape)
    print("conn:", data["conn"]["conn"].shape)
    print("SC_len:", data["conn"]["SC_len"].shape)
    print("name:", len(data["conn"]["name"]))
    print("ROI_size:", len(data["conn"]["ROI_size"]))

    return data


def random_spot_check(data, sample_size=5, seed=None):
    """Randomly inspect several positions for name/tau/ROI_size alignment."""
    print("\n*** Random Spot Check ***")

    if seed is not None:
        random.seed(seed)

    names = list(data["conn"]["name"])
    tau_index = list(data["tau"].index)
    roi_keys = list(data["conn"]["ROI_size"].keys())

    n = len(names)
    sample_size = min(sample_size, n)
    check_ids = random.sample(range(n), sample_size)

    print("Random indices:", check_ids)

    for i in check_ids:
        print(
            i,
            "| name:", names[i],
            "| tau index:", tau_index[i],
            "| ROI key:", roi_keys[i],
        )


def main():
    args = parse_args()

    input_pkl_path = Path(args.input_pkl)

    if args.drop_names and args.drop_names_txt:
        raise ValueError("Cannot specify both --drop-names and --drop-names-txt")

    if args.drop_names_txt:
        base_name, drop_names = load_drop_names_from_txt(args.drop_names_txt)
    elif args.drop_names:
        drop_names = args.drop_names

        base_name =[]
        for i in drop_names:
            base = "_".join(i.split("_")[1:])
            if base not in base_name:
                base_name.append(base)


    data = load_input_pkl(args.input_pkl)
    extracted = extract_core_data(data)

    basic_information_check(extracted, show_all_names=args.show_all_names)
    alignment_check(data, stage="Before Drop")

    _, keep_idx = find_drop_indices(
        names=extracted["names"],
        drop_names=drop_names,
        allow_missing_drop_names=args.allow_missing_drop_names,
    )

    data = apply_modifications(data, extracted, keep_idx)

    final_alignment = alignment_check(data, stage="Final")

    assert final_alignment["name_eq_tau"], "Mismatch between conn names and tau index"
    assert final_alignment["name_eq_roi"], "Mismatch between conn names and ROI_size keys"
    assert final_alignment["conn_eq_sc_len"], "Mismatch between conn and SC_len shapes"

    random_spot_check(data, sample_size=5, seed=None)

    # output_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(base_name)
    output_name = f"{input_pkl_path.stem}_drop_{tag}.pkl"

    with open(output_name, "wb") as f:
        pickle.dump(data, f)

    print("\nSaved to:", output_name)


if __name__ == "__main__":
    main()