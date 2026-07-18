#!/usr/bin/env python3

"""
Reduce brain-region input pickle files by dropping selected bilateral regions.

This script removes specified brain regions from an SIR input pickle file and
updates all associated data structures while preserving their original region
ordering. It supports both standard SIR input files containing observed tau
data and connectivity-only input files.

Region names are supplied as base names, such as ``Hippocampus`` or
``Amygdala``. Each base name is automatically expanded into its corresponding
left- and right-hemisphere names:

    Hippocampus
        -> Left_Hippocampus
        -> Right_Hippocampus

The reduced data are saved as one or more new pickle files in the current
working directory.

----------------------------------------------------------------------
Supported input formats
----------------------------------------------------------------------

Format A: Standard SIR input
----------------------------
Expected structure:

    data = {
        "tau": pandas.DataFrame,
        "conn": {
            "conn": numpy.ndarray,
            "SC_len": numpy.ndarray,
            "name": list[str],
            "ROI_size": dict[str, Any],
        },
    }

The expected components are:

``tau``
    Observed protein measurements. Rows must correspond to brain regions.

``conn["conn"]``
    Square structural-connectivity matrix.

``conn["SC_len"]``
    Square structural-connection-length matrix.

``conn["name"]``
    Ordered region names corresponding to the rows and columns of the
    connectivity matrices.

``conn["ROI_size"]``
    Mapping from region names to ROI sizes.

Format B: Connectivity-only input
---------------------------------
Expected structure:

    data = {
        "conn": numpy.ndarray,
        "SC_len": numpy.ndarray,
        "labels": list[str],
        "ROI_size": dict[str, Any],
    }

The expected components are:

``conn``
    Square structural-connectivity matrix.

``SC_len``
    Square structural-connection-length matrix.

``labels``
    Ordered region names corresponding to matrix rows and columns.

``ROI_size``
    Mapping from region names to ROI sizes.

----------------------------------------------------------------------
Processing
----------------------------------------------------------------------

For each requested base region name, the script:

1. expands the base name into ``Left_<name>`` and ``Right_<name>``;
2. identifies the corresponding region indices;
3. removes the associated rows from the tau DataFrame, when tau is present;
4. removes the associated rows and columns from ``conn`` and ``SC_len``;
5. updates the ordered region-name list;
6. rebuilds the ``ROI_size`` dictionary using the retained regions;
7. prints alignment information before and after reduction;
8. performs a random spot check of retained region names;
9. saves the reduced dictionary as a new pickle file.

----------------------------------------------------------------------
Ways to specify regions
----------------------------------------------------------------------

Regions can be supplied directly using ``--drop-names``:

    python reduce_input_pkl_by_names.py \
        --input-pkl path/to/input.pkl \
        --drop-names Hippocampus Amygdala

They can also be read from a text file using ``--drop-names-txt``:

    python reduce_input_pkl_by_names.py \
        --input-pkl path/to/input.pkl \
        --drop-names-txt path/to/drop_regions.txt

The text file must contain one base region name per non-empty line:

    Hippocampus
    Amygdala
    Caudate

``--drop-names`` and ``--drop-names-txt`` are mutually exclusive.

----------------------------------------------------------------------
Combined and per-line processing
----------------------------------------------------------------------

By default, all base names are removed together and one output file is
generated:

    python reduce_input_pkl_by_names.py \
        --input-pkl path/to/input.pkl \
        --drop-names-txt drop_regions.txt

To process every line of the text file independently, use
``--process-each-line``:

    python reduce_input_pkl_by_names.py \
        --input-pkl path/to/input.pkl \
        --drop-names-txt drop_regions.txt \
        --process-each-line

In this mode, the original input pickle is reloaded for every base region.
Therefore, each output file represents removal of only one bilateral region
pair rather than cumulative removal across lines.

The ``--process-each-line`` option is only used with
``--drop-names-txt``.

----------------------------------------------------------------------
Optional arguments
----------------------------------------------------------------------

``--show-all-names``
    Print all available connectivity region names before reduction. For
    standard SIR inputs, the tau DataFrame index is also printed.

``--allow-missing-drop-names``
    Continue when one or more expanded region names are not present in the
    input. Without this option, missing names raise ``ValueError``.

``--process-each-line``
    Process every base name from ``--drop-names-txt`` independently and
    generate a separate output pickle for each line.

----------------------------------------------------------------------
Output naming
----------------------------------------------------------------------

Outputs are written to the current working directory.

When processing one base region independently, the output filename is:

    <input_stem>_drop_<base_name>.pkl

Example:

    Input_SIR_ROI83_drop_Hippocampus.pkl

When multiple regions are removed together and no more than five base names
are provided, the names are included in the output filename:

    <input_stem>_drop_<name1>_<name2>.pkl

Example:

    Input_SIR_ROI83_drop_Hippocampus_Amygdala.pkl

When more than five base names are removed together, the filename records only
the number of base regions:

    <input_stem>_drop_<number>_regions.pkl

Example:

    Input_SIR_ROI83_drop_7_regions.pkl

Existing files with the same output filename will be overwritten.

----------------------------------------------------------------------
Validation and diagnostic output
----------------------------------------------------------------------

Before and after modification, the script reports whether:

- connectivity region names match the tau DataFrame index, when tau is present;
- ``conn`` and ``SC_len`` have identical shapes;
- connectivity region names match the ordered keys of ``ROI_size``.

These checks are diagnostic only. They print Boolean results but do not raise
errors when an inconsistency is detected.

After reduction, up to five retained region indices are randomly selected and
printed to allow manual comparison of:

- connectivity names;
- tau index values, when tau is present;
- ROI size dictionary keys.

Because no random seed is specified by the main workflow, the selected indices
may differ between runs.

----------------------------------------------------------------------
Notes
----------------------------------------------------------------------

- Region-name matching is exact and case-sensitive.
- Input region names must follow the ``Left_<base_name>`` and
  ``Right_<base_name>`` naming convention for automatic bilateral expansion.
- All matrices, names, tau rows, and ROI size entries are assumed to use the
  same region order before reduction.
- The script modifies the loaded dictionary in memory but does not overwrite
  the original input pickle.
- Pickle files can execute arbitrary code during loading and should only be
  loaded from trusted sources.
"""



import argparse
import pickle
from pathlib import Path
import random

import numpy as np
import pandas as pd

def parse_args():
    """
    Parse command-line arguments for region removal.

    Regions can be provided either directly as base names through
    ``--drop-names`` or through a text file using ``--drop-names-txt``.
    The two methods cannot be used together.

    Base region names are later expanded into corresponding ``Left_`` and
    ``Right_`` region names.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing:

        input_pkl : str
            Path to the input pickle file.

        drop_names : list of str or None
            Base region names supplied directly on the command line.

        drop_names_txt : str or None
            Path to a text file containing one base region name per line.

        process_each_line : bool
            Whether each line in ``drop_names_txt`` should be processed
            independently to generate a separate output file.

        show_all_names : bool
            Whether to print all available region names.

        allow_missing_drop_names : bool
            Whether requested region names that are absent from the input
            should be ignored.
    """

    parser = argparse.ArgumentParser(
        description="Reduce SIR input pkl by dropping regions using names."
    )
    parser.add_argument("--input-pkl", required=True, type=str)
    parser.add_argument("--drop-names", nargs="+", default=None)
    parser.add_argument("--drop-names-txt", type=str, default=None)
    parser.add_argument("--process-each-line", action="store_true")
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
        - Connectivity-only format (with 'conn', 'labels', etc.)
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


def load_base_names_from_txt(txt_path):
    """
    Load base region names from a text file, one per line.

    Parameters
    ----------
    txt_path : str or Path
        Path to text file.

    Returns
    -------
    list of str
        Base region names read from file.
    """
    txt_path = Path(txt_path)
    if not txt_path.is_file():
        raise ValueError(f"Drop names txt file not found: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        base_names = [line.strip() for line in f if line.strip()]

    print("Base names:", base_names)
    return base_names


def expand_base_names(base_names):
    """
    Expand base region names into Left_ and Right_ full region names.

    Parameters
    ----------
    base_names : list of str
        Base region names such as ['Hippocampus', 'Amygdala'].

    Returns
    -------
    list of str
        Expanded full region names such as
        ['Left_Hippocampus', 'Right_Hippocampus',
         'Left_Amygdala', 'Right_Amygdala'].
    """
    if isinstance(base_names, str):
        base_names = [base_names]

    drop_names = []
    for name in base_names:
        drop_names.append(f"Left_{name}")
        drop_names.append(f"Right_{name}")
    return drop_names        


def extract_core_data(data):
    """
    Extract core data structures from input dictionary.

    This function normalizes access to key components regardless of input type.

    Supported input formats:
    - Standard SIR input (contains 'tau' and 'conn')
    - Connectivity-only input (contains 'conn', 'labels', etc.)

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
        conn = np.asarray(data['conn'])
        names = list(data['labels'])
        roi_size = data["ROI_size"]
        sc_len = np.asarray(data['SC_len'])
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
        roi_keys = list(data['ROI_size'].keys())
        print("conn shape == SC_len shape:",
            np.asarray(data["conn"]).shape == np.asarray(data["SC_len"]).shape)

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
    For standard SIR input:

    - subsets the tau DataFrame by row;
    - subsets ``conn["conn"]`` and ``conn["SC_len"]`` by row and column;
    - updates ``conn["name"]``;
    - rebuilds ``conn["ROI_size"]``.

    For connectivity-only input:

    - subsets the top-level ``conn`` and ``SC_len`` matrices;
    - updates the top-level ``labels`` list;
    - rebuilds the top-level ``ROI_size`` dictionary.
    
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
        data['conn'] = conn[np.ix_(keep_idx, keep_idx)]
        data['SC_len'] = sc_len[np.ix_(keep_idx, keep_idx)]
        data['labels'] = new_names
        data['ROI_size'] = {name: old_roi[name] for name in new_names}
        
        print("\n*** After Modification ***")
        print("conn:", data['conn'].shape)
        print("SC_len:", data['SC_len'].shape)
        print("name:", len(data['labels']))
        print("ROI_size:", len(data['ROI_size']))


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
        roi_keys = list(data['ROI_size'].keys())

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
        base_names = load_base_names_from_txt(args.drop_names_txt)

        if args.process_each_line:
            for base_name in base_names:
                print(f"Processing one region set: {base_name}")

                drop_names = expand_base_names([base_name])
                data_single = load_input_pkl(args.input_pkl)
                extracted = extract_core_data(data_single)

                basic_information_check(extracted, show_all_names=args.show_all_names)
                alignment_check(data_single, stage=f"Before Drop ({base_name})")

                _, keep_idx = find_drop_indices(
                    names = extracted['names'],
                    drop_names=drop_names,
                    allow_missing_drop_names=args.allow_missing_drop_names,
                )
                data_single = apply_modifications(data_single, extracted, keep_idx)
                alignment_check(data_single, stage=f"Final ({base_name})")
                random_spot_check(data_single, sample_size=5, seed=None)

                output_name = f"{input_pkl_path.stem}_drop_{base_name}.pkl"
                with open(output_name, "wb") as f:
                        pickle.dump(data_single, f)
                
                print("Saved to:", output_name)
            return
        else:
            drop_names = expand_base_names(base_names)

    elif args.drop_names:
        base_names = args.drop_names
        drop_names = expand_base_names(base_names)

    else:
        raise ValueError("Either --drop-names or --drop-names-txt must be provided")

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
    if len(base_names) <=5:
        tag = "_".join(base_names)
    else:
        tag = f"{len(base_names)}_regions"
    output_name = f"{input_pkl_path.stem}_drop_{tag}.pkl"

    with open(output_name, "wb") as f:
        pickle.dump(data, f)

    print("\nSaved to:", output_name)


if __name__ == "__main__":
    main()