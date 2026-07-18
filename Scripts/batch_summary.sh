#!/usr/bin/env bash
"""
Batch summary script for SIR simulation results

This script reads a list of SIR simulation result directories from:
    ../Xiao_Tau_simulation_SIR/results/results_path.txt

Each line in the file is treated as the path to one completed simulation
result. The script processes these directories sequentially and calls
`run_summary_one_conn.py` to summarize the model results using:
    - structural connectivity type: SC;
    - epicenter: ctx_lh_entorhinal;
    - input data name: Input_SIR_ROI66_replace_load.pkl

For each result directory, the following command is executed:
    python ./scripts/summary/run_summary_one_conn.py \
    --result_path <result_directory> --connectivity SC --epicenter ctx_lh_entorhinal \
    --input_data_name <input_pickle_file>

Usage:
    1. Ensure that `results_path.txt` contains the paths to the result directories.
    2. Run this script in a bash environment.
        bash batch_summary.sh
"""


while IFS= read -r cond
do
    cond=${cond%$'\r'}
    echo "Processing: [$cond]"
    python ./scripts/summary/run_summary_one_conn.py --result_path "$cond" --connectivity SC --epicenter ctx_lh_entorhinal \
    --input_data_name /e/01_jiaoyq/02_Project_management/005_SIR_sim/Xiao_Tau_simulation_SIR/data/review_model/Input_SIR_ROI66_replace_load.pkl  
done < ../Xiao_Tau_simulation_SIR/results/results_path.txt