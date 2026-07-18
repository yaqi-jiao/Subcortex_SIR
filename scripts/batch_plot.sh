#!/usr/bin/env bash
"""
Batch figure generation script

This script reads a list of result-directory paths from:
    ./results/results_path.txt

Each non-empty line is treated as the input directory for one completed model 
result. The script processes the directories sequentially and calls `plot_scatter_from_folder.py` 
to generate scatter plots for each result folder.

The expected command for each directory is:
    python ../scripts/plot_scatter_from_folder.py --input_dir <result_directory>

An alternative brain-surface plotting command is retained as a commented line 
and can be enabled when surface plots are required.

Usage:
    1. Ensure that `results_path.txt` contains the paths to the result directories.
    2. Run this script in a bash environment.
        bash batch_plot.sh
    
"""

while IFS= read -r cond
do
    cond=${cond%$'\r'}
    echo "Processing: [$cond]"
    # python ../scripts/plot_brain_surface_from_folder.py --input_dir "$cond" --atlas-path /e/01_jiaoyq/02_Project_management/005_SIR_sim/Brain_surface_plot/
    python ../scripts/plot_scatter_from_folder.py --input_dir "$cond"
done < ./results/results_path.txt