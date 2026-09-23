#!/usr/bin/env python3
"""
Export already-assembled alternative-connectome PKL files into CSV files.

Designed for PKL files with the structure used in Assemble_FC_pkl.ipynb:

{
    "FC": np.ndarray,
    "labels": list[str],
    "ROI_size": dict[str, number],
    "SC_len": None or np.ndarray
}

The script writes one set of CSV files per PKL, plus a small metadata CSV so
the Bianca-side script can reconstruct the dictionary without guessing.

Example
-------
python export_alternative_connectomes_to_csv.py \
    --input_dir /path/to/alternative_connectivity \
    --output_dir /path/to/alternative_connectivity_csv
"""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


RESERVED_KEYS = {"labels", "ROI_size", "SC_len"}


def write_key_value_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerows(rows)


def write_labels_csv(path, labels):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label"])
        for label in labels:
            writer.writerow([str(label)])


def write_roi_size_csv(path, labels, roi_size):
    # Export in exactly the same order as labels.
    missing = [label for label in labels if label not in roi_size]
    extra = [name for name in roi_size if name not in set(labels)]

    if missing:
        raise ValueError(
            f"ROI_size is missing {len(missing)} labels. "
            f"First missing labels: {missing[:5]}"
        )
    if extra:
        raise ValueError(
            f"ROI_size contains {len(extra)} names not present in labels. "
            f"First extras: {extra[:5]}"
        )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "ROI_size"])
        for label in labels:
            value = roi_size[label]
            if np.ndim(value) != 0:
                raise ValueError(
                    f"ROI_size[{label!r}] is not scalar: {type(value)}"
                )
            writer.writerow([str(label), repr(float(value))])


def export_one(pkl_path, output_dir):
    print("\n" + "=" * 80)
    print("Reading:", pkl_path)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"{pkl_path}: expected dict, got {type(data)}")

    connectivity_keys = [k for k in data.keys() if k not in RESERVED_KEYS]
    if len(connectivity_keys) != 1:
        raise ValueError(
            f"{pkl_path}: expected exactly one connectivity key besides "
            f"{sorted(RESERVED_KEYS)}, found {connectivity_keys}"
        )

    connectivity_key = connectivity_keys[0]

    for required in ["labels", "ROI_size", "SC_len"]:
        if required not in data:
            raise KeyError(f"{pkl_path}: missing required key {required!r}")

    matrix = np.asarray(data[connectivity_key])
    labels = [str(x) for x in data["labels"]]
    roi_size = data["ROI_size"]
    sc_len = data["SC_len"]

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"{pkl_path}: {connectivity_key} must be a square 2D matrix, "
            f"got {matrix.shape}"
        )

    if len(labels) != matrix.shape[0]:
        raise ValueError(
            f"{pkl_path}: matrix has {matrix.shape[0]} ROIs but labels has "
            f"{len(labels)}"
        )

    if not isinstance(roi_size, dict):
        raise TypeError(
            f"{pkl_path}: ROI_size must be dict, got {type(roi_size)}"
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{pkl_path}: {connectivity_key} contains NaN/Inf")

    stem = pkl_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_csv = output_dir / f"{stem}__connectivity.csv"
    labels_csv = output_dir / f"{stem}__labels.csv"
    roi_size_csv = output_dir / f"{stem}__ROI_size.csv"
    metadata_csv = output_dir / f"{stem}__metadata.csv"

    # High precision to make the CSV round-trip as faithful as practical.
    np.savetxt(matrix_csv, matrix, delimiter=",", fmt="%.18g")
    write_labels_csv(labels_csv, labels)
    write_roi_size_csv(roi_size_csv, labels, roi_size)

    metadata_rows = [
        ["original_filename", pkl_path.name],
        ["connectivity_key", connectivity_key],
        ["connectivity_dtype", str(matrix.dtype)],
        ["n_roi", str(matrix.shape[0])],
        ["SC_len_is_none", "1" if sc_len is None else "0"],
    ]

    if sc_len is not None:
        sc_len = np.asarray(sc_len)

        if sc_len.shape != matrix.shape:
            raise ValueError(
                f"{pkl_path}: SC_len shape {sc_len.shape} does not match "
                f"{connectivity_key} shape {matrix.shape}"
            )

        if not np.all(np.isfinite(sc_len)):
            raise ValueError(f"{pkl_path}: SC_len contains NaN/Inf")

        sc_len_csv = output_dir / f"{stem}__SC_len.csv"
        np.savetxt(sc_len_csv, sc_len, delimiter=",", fmt="%.18g")
        metadata_rows.append(["SC_len_dtype", str(sc_len.dtype)])

    write_key_value_csv(metadata_csv, metadata_rows)

    print(f"  key       : {connectivity_key}")
    print(f"  matrix    : {matrix.shape}, {matrix.dtype}")
    print(f"  labels    : {len(labels)}")
    print(f"  ROI_size  : {len(roi_size)}")
    print(f"  SC_len    : {'None' if sc_len is None else sc_len.shape}")
    print("  exported  :", stem)


def main():
    parser = argparse.ArgumentParser(
        description="Export alternative-connectome PKL dictionaries to CSV."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing the already-assembled .pkl files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="CSV output directory. Default: <input_dir>/csv_export",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = (
        args.output_dir if args.output_dir is not None
        else input_dir / "csv_export"
    )

    pkl_files = sorted(input_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {input_dir}")

    print(f"Found {len(pkl_files)} PKL files.")

    for pkl_path in pkl_files:
        export_one(pkl_path, output_dir)

    print("\n" + "=" * 80)
    print(f"Done. CSV files saved to: {output_dir}")


if __name__ == "__main__":
    main()
