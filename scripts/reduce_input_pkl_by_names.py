#!/usr/bin/env python3

"""
Reduce brain-region input pickle files by dropping specified regions.

This script removes selected brain regions from an input pickle file and
updates all related data structures consistently. It supports two input
formats:

1. Standard SIR simulation input
   A dictionary containing observed tau data together with structural
   connectivity information.

2. Connectivity-only alternative input
   A dictionary containing only connectivity-related matrices and labels,
   without any tau DataFrame.

After dropping the requested regions, the script writes a new reduced
pickle file with updated matrices, labels, and ROI size mappings.

----------------------------------------------------------------------
Supported input formats
----------------------------------------------------------------------

Format A: Standard SIR input
----------------------------
Expected structure:

data = {
    "tau": pandas.DataFrame,
    "conn": {
        "conn": numpy.ndarray,      # shape (n_regions, n_regions)
        "SC_len": numpy.ndarray,    # shape (n_regions, n_regions)
        "name": list[str],          # length n_regions
        "ROI_size": dict[str, Any]  # one entry per region
    }
}

Description:
- "tau":
    Observed protein measurements indexed by brain region names.
- "conn"]["conn"]:
    Structural connectivity matrix.
- "conn"]["SC_len"]:
    Structural connectivity length matrix.
- "conn"]["name"]:
    Region names corresponding to matrix rows/columns.
- "conn"]["ROI_size"]:
    ROI size dictionary aligned with region names.

Format B: Connectivity-only alternative input
---------------------------------------------
Expected structure:

data = {
    "alt_conn": numpy.ndarray,        # shape (n_regions, n_regions)
    "alt_sc_len": numpy.ndarray,      # shape (n_regions, n_regions)
    "labels": list[str],              # length n_regions
    "alt_ROI_size": dict[str, Any]    # one entry per region
}

Description:
- "alt_conn":
    Alternative connectivity matrix.
- "alt_sc_len":
    Alternative structural connectivity length matrix.
- "labels":
    Region names corresponding to matrix rows/columns.
- "alt_ROI_size":
    ROI size dictionary aligned with labels.

----------------------------------------------------------------------
What this script does
----------------------------------------------------------------------

Given a list of region names to remove, the script:

- finds the matching region indices
- removes those rows from tau data if tau is present
- removes corresponding rows/columns from connectivity matrices
- updates the region-name list
- updates the ROI_size dictionary
- performs alignment checks before and after reduction
- saves the reduced result as a new pickle file

If tau is not present, the script skips tau-related processing and only
updates connectivity-related structures.

----------------------------------------------------------------------
Usage
----------------------------------------------------------------------

Drop regions by explicitly listing names:

    python reduce_input_pkl_by_names.py \
        --input-pkl path/to/input.pkl \
        --drop-names Left_Hippocampus Right_Hippocampus

Drop regions using a text file (one base region name per line):

    python reduce_input_pkl_by_names.py \
        --input-pkl path/to/input.pkl \
        --drop-names-txt path/to/drop_regions.txt

Example contents of drop_regions.txt:

    Hippocampus
    Amygdala

These names can be expanded internally into:
    Left_Hippocampus, Right_Hippocampus
    Left_Amygdala, Right_Amygdala

----------------------------------------------------------------------
Optional flags
----------------------------------------------------------------------

--show-all-names
    Print all available region names before reduction.

--allow-missing-drop-names
    Ignore requested drop names that are not found in the input.

----------------------------------------------------------------------
Output
----------------------------------------------------------------------

The script saves a new pickle file in the current working directory.
The output filename is automatically generated from the input filename
plus a suffix indicating the dropped region name(s).

Example:
    Input_SIR_example_drop_Hippocampus_Amygdala.pkl

----------------------------------------------------------------------
Notes
----------------------------------------------------------------------

- The script assumes region ordering must remain consistent across all
  matrices, labels, and ROI_size mappings.
- For standard SIR input, tau index and connectivity names should refer
  to the same regions in the same order.
- For connectivity-only input, labels and ROI_size keys should remain
  aligned after reduction.
- Incorrect or inconsistent input structure may cause alignment checks
  to fail or lead to invalid output.
"""

import argparse
import pickle
from pathlib import Path
import random

import numpy as np
import pandas as pd

def parse_args():
    """
    Parse command-line arguments.

    Supports two mutually exclusive ways to specify regions to drop:
    - --drop-names: directly provide full region names
    - --drop-names-txt: provide a text file containing base region names

    Returns
    -------
    argparse.Namespace
        Parsed arguments including:
        - input_pkl: path to input pickle file
        - drop_names: list of region names (or None)
        - drop_names_txt: path to txt file (or None)
        - show_all_names: whether to print all region names
        - allow_missing_drop_names: whether to ignore missing names
    """
    parser = argparse.ArgumentParser(
        description="Reduce SIR input pkl by dropping regions using names."
    )
    parser.add_argument("--input-pkl", required=True, type=str)
    # parser.add_argument("--output-pkl", required=True, type=str)
    parser.add_argument("--drop-names", nargs="+", default=None)
    parser.add_argument("--drop-names-txt", type=str, default=None)
    parser.add_argument("--show-all-names", action="store_true")
    parser.add_argument("--allow-missing-drop-names", action="store_true")
    return parser.parse_args()


def load_input_pkl(input_pkl_path):
    """
    Load a pickle file containing brain-region data.

    Parameters
    ----------
    input_pkl_path : str or Path
        Path to the input pickle file.

    Returns
    -------
    dict
        Loaded data dictionary. Expected to match one of:
        - Standard SIR format (with 'tau' and 'conn')
        - Connectivity-only format (with 'alt_conn', 'labels', etc.)
    """
    input_pkl_path = Path(input_pkl_path)

    with open(input_pkl_path, "rb") as f:
        data = pickle.load(f)

    if 'tau' in data:
        print("Top-level keys:", list(data.keys()))
        print("conn subkeys:", list(data["conn"].keys()))
    else:
        print("connectome data only")
        print("Top-level keys:", list(data.keys()))

    return data


def load_drop_names_from_txt(txt_path):
    """
    Load region names from a text file and expand to full hemisphere names.

    Each line in the file is treated as a base region name (e.g., "Hippocampus"),
    which will be expanded to:
        Left_<name>, Right_<name>

    Parameters
    ----------
    txt_path : str or Path
        Path to text file containing region base names.

    Returns
    -------
    base_names : list of str
        Original base names from file.
    drop_names : list of str
        Expanded region names including hemisphere prefixes.
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
    return base_names, drop_names


def extract_core_data(data):
    """
    Extract core data structures from input dictionary.

    This function normalizes access to key components regardless of input type.

    Supported input formats:
    - Standard SIR input (contains 'tau' and 'conn')
    - Connectivity-only input (contains 'alt_conn', 'labels', etc.)

    Parameters
    ----------
    data : dict
        Input data dictionary.

    Returns
    -------
    dict
        Dictionary containing:
        - tau_df : pandas.DataFrame or None
        - conn : numpy.ndarray
        - names : list of str
        - roi_size : dict
        - sc_len : numpy.ndarray

    Notes
    -----
    - tau_df will be None if tau is not present.
    - names always define the reference ordering for all matrices.
    """
    tau_df = data.get("tau", None)

    if tau_df is None:
        conn = np.asanyarray(data['alt_conn'])
        names = list(data['labels'])
        roi_size = data["alt_ROI_size"]
        sc_len = np.asanyarray(data['alt_sc_len'])
    else:
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
    """
    Print basic structural information of the input data.

    Parameters
    ----------
    extracted : dict
        Output from extract_core_data().
    show_all_names : bool
        If True, print full list of region names (and tau index if available).

    Notes
    -----
    - Handles both tau-present and tau-absent cases.
    - Used for sanity checking before modification.
    """
    tau_df = extracted["tau_df"]

    print("\n*** Original Data Info ***")
    if tau_df is not None:
        print("tau shape:", tau_df.shape)
    else:
        print("tau: NOT PRESENT")
    
    conn = extracted["conn"]
    names = extracted["names"]
    roi_size = extracted["roi_size"]
    sc_len = extracted["sc_len"]
   
    print("conn shape:", conn.shape)
    print("SC_len shape:", sc_len.shape)
    print("len(names):", len(names))
    print("ROI_size len:", len(roi_size))

    if show_all_names:
        if tau_df is not None:
            print("*** observed tau index ***")
            for i, name in enumerate(tau_df.index):
                print(i, name)
        print("*** connectome ROI index***")
        for i, name in enumerate(names):
            print(i, name)


def alignment_check(data, stage="Current"):
    """
    Check alignment consistency across data structures.

    Parameters
    ----------
    data : dict
        Input data dictionary.
    stage : str
        Label for logging (e.g., "Before Drop", "Final").

    Notes
    -----
    For standard SIR input:
        - checks name == tau.index
        - checks conn shape == SC_len shape
        - checks name == ROI_size keys

    For connectivity-only input:
        - checks alt_conn shape == alt_sc_len shape
        - checks labels == alt_ROI_size keys

    This function prints results but does not raise errors.
    """
    print(f"\n*** {stage} Alignment Check ***")
    
    if "tau" in data:
        tau_index = list(data["tau"].index)
        names = list(data["conn"]["name"])
        roi_keys = list(data["conn"]["ROI_size"].keys())
        print("name == tau.index:", names == tau_index)
        print("conn shape == SC_len shape:",
          np.asarray(data["conn"]["conn"]).shape == np.asarray(data["conn"]["SC_len"]).shape)
    else:
        names = list(data['labels'])
        roi_keys = list(data['alt_ROI_size'].keys())
        print("conn shape == SC_len shape:",
            np.asarray(data["alt_conn"]).shape == np.asarray(data["alt_sc_len"]).shape)

    print("name == ROI_size.keys:", names == roi_keys)


    # return {
    #     "name_eq_tau": names == tau_index,
    #     "name_eq_roi": names == roi_keys,
    #     "conn_eq_sc_len": np.asarray(data["conn"]["conn"]).shape == np.asarray(data["conn"]["SC_len"]).shape,
    # }


def find_drop_indices(names, drop_names, allow_missing_drop_names=False):
    """
    Identify indices of regions to remove.

    Parameters
    ----------
    names : list of str
        Full list of region names.
    drop_names : list of str
        Region names to remove.
    allow_missing_drop_names : bool
        If False, raise error when drop_names not found in names.

    Returns
    -------
    drop_idx : list of int
        Indices of regions to remove.
    keep_idx : list of int
        Indices of regions to keep.

    Notes
    -----
    - Matching is exact string match.
    - Order of regions is preserved via keep_idx.
    """
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
    """
    Apply region removal to all relevant data structures.

    Parameters
    ----------
    data : dict
        Original data dictionary (modified in place).
    extracted : dict
        Output from extract_core_data().
    keep_idx : list of int
        Indices of regions to retain.

    Returns
    -------
    dict
        Updated data dictionary with reduced regions.

    Behavior
    --------
    If tau is present:
        - Subset tau DataFrame rows
        - Subset conn and SC_len matrices
        - Update region names
        - Update ROI_size

    If tau is NOT present:
        - Subset alt_conn and alt_sc_len matrices
        - Update labels
        - Update alt_ROI_size

    Notes
    -----
    - All structures are assumed to share the same region ordering.
    - Modifications preserve index alignment across all components.
    """
    conn = extracted["conn"]
    sc_len = extracted["sc_len"]
    names = extracted["names"]
    old_roi = extracted["roi_size"]

    new_names = [names[i] for i in keep_idx]

    # tau (optional)
    if "tau" in data:
        data["tau"] = data["tau"].iloc[keep_idx]
        data["conn"]["conn"] = conn[np.ix_(keep_idx, keep_idx)]
        data["conn"]["SC_len"] = sc_len[np.ix_(keep_idx, keep_idx)]
        data["conn"]["name"] = new_names
        data["conn"]["ROI_size"] = {name: old_roi[name] for name in new_names}

        print("\n*** After Modification ***")
        print("tau:", data["tau"].shape)
        print("conn:", data["conn"]["conn"].shape)
        print("SC_len:", data["conn"]["SC_len"].shape)
        print("name:", len(data["conn"]["name"]))
        print("ROI_size:", len(data["conn"]["ROI_size"]))
    else:
        data['alt_conn'] = conn[np.ix_(keep_idx, keep_idx)]
        data['alt_sc_len'] = sc_len[np.ix_(keep_idx, keep_idx)]
        data['labels'] = new_names
        data['alt_ROI_size'] = {name: old_roi[name] for name in new_names}
        
        print("\n*** After Modification ***")
        print("conn:", data['alt_conn'].shape)
        print("SC_len:", data['alt_sc_len'].shape)
        print("name:", len(data['labels']))
        print("ROI_size:", len(data['alt_ROI_size']))


    return data


def random_spot_check(data, sample_size=5, seed=None):
    """
    Randomly inspect a subset of regions to verify alignment.

    Parameters
    ----------
    data : dict
        Updated data dictionary.
    sample_size : int
        Number of random indices to inspect.
    seed : int or None
        Random seed for reproducibility.

    Notes
    -----
    - For tau-present input: checks name vs tau index vs ROI keys.
    - For tau-absent input: checks labels vs ROI keys only.
    - Intended as a lightweight sanity check after modification.
    """
    print("*** Random Spot Check ***")

    if seed is not None:
        random.seed(seed)

    if 'tau' in data:
        tau_index = list(data["tau"].index)
        names = list(data["conn"]["name"])
        roi_keys = list(data["conn"]["ROI_size"].keys())
    else:
        tau_index = None
        names = list(data['labels'])
        roi_keys = list(data['alt_ROI_size'].keys())

    n = len(names)
    sample_size = min(sample_size, n)
    check_ids = random.sample(range(n), sample_size)

    print("Random indices:", check_ids)

    for i in check_ids:
        if tau_index is not None:
            print(
                i,
                "| name:", names[i],
                "| tau index:", tau_index[i],
                "| ROI key:", roi_keys[i],
            )
        else:
            print(
                i,
                "| name:", names[i],
                "| tau index: N/A",
                "| ROI key:", roi_keys[i],
            )


def main():
    """
    Main execution pipeline.

    Workflow
    --------
    1. Parse command-line arguments
    2. Load input pickle
    3. Determine drop_names (direct or from txt)
    4. Extract core data structures
    5. Perform initial information and alignment checks
    6. Compute indices to keep
    7. Apply modifications
    8. Perform final alignment and random spot checks
    9. Save reduced pickle file

    Notes
    -----
    - Supports both tau-present and connectivity-only inputs.
    - Output filename is auto-generated based on dropped region names.
    """
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

    alignment_check(data, stage="Final")

    random_spot_check(data, sample_size=5, seed=None)

    # output_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(base_name)
    output_name = f"{input_pkl_path.stem}_drop_{tag}.pkl"

    with open(output_name, "wb") as f:
        pickle.dump(data, f)

    print("\nSaved to:", output_name)


if __name__ == "__main__":
    main()