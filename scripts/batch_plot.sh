#!/usr/bin/env bash

while IFS= read -r cond
do
    cond=${cond%$'\r'}
    echo "Processing: [$cond]"
    python plot_fig2_from_folder.py --input_dir "$cond"
done < ../Xiao_Tau_simulation_SIR/results/results_path.txt