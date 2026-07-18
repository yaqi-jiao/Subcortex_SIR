## Tau Propagation Simulation in subcortiacl regions with SIR Model
This research project aims to apply SIR model on different brain structures, to see if including 

This repository contains python code and mock data for batch run SIR hypertune model locally. It's a complementation for SIR model on SLURM (Bianca), allowing people running multiple jobs in one bash script.

#### Directory structure

- Script Directory

    It contains the python and bash code for batch running the SIR model, and generate scatter plot

- Data Directory

    It contains 

- Model Directory:
   
   It contains the python code for SIR model.

#### Installation (mention the SIR model repo)
1. clone the repository
2. clone SIR repositry following the guidance
Note: 

#### Usage: batch run the model

1. Generate different connectomes
   `reduce_input_pkl_by_names.py`
    To generate alternative input connectome with different brain regions with brain regions you want to drop. 

    there are 2 ways to specify dropped regions:
    --drop-names
    ```python
        python reduce_input_pkl_by_names.py \
        --input-pkl path/to/input.pkl \
        --drop-names <regions you want to drop>  # for example: Hippocampus Amygdala
    ```
    --drop-names-txt
    ```python
    python reduce_input_pkl_by_names.py \
    --input-pkl path/to/input.pkl \
    --drop-names-txt path/to/drop_regions.txt
    --process-each-line  # note: without this, all regions will be removed once
    ```
    
2. Batch Running the SIR model
3. Batch Run the summary
4. Bacth Generate scatter plot and brain surface projection



