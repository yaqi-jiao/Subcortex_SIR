# Tau Propagation Simulation in Subcortical Regions Using an SIR Model

This research project investigates tau propagation across different brain network structures using a network-based susceptible–infected–removed (SIR) model.

The main goal is to evaluate how the inclusion or exclusion of specific subcortical regions affects simulated tau propagation and model performance. The repository also supports experiments using region-specific biological factors that modify model parameters such as spreading, synthesis, clearance, and misfolding.


## Repository Relationship

This repository contains the **project-specific workflow** built around a modified SIR tau-propagation model.

The core SIR simulation code is maintained separately in my fork of the original SIR model repository:

**SIR model repository:** [SIR model](<https://github.com/yaqi-jiao/Xiao_Tau_simulation_SIR>)

The two repositories serve different purposes:

- This repository contains Python and Bash scripts, together with example or mock input data, for the **project-specific workflow** built around a modified SIR tau-propagation model, which runs batches of SIR hyperparameter-tuning experiments locally. It complements the original SLURM-based SIR workflow used on the Bianca computing cluster by allowing multiple experiments to be configured and executed sequentially on a local machine.

- **The SIR model repository** contains the core simulation framework, hyperparameter tuning, model evaluation, and the ROI-mapping / alternative-connectivity extensions used by this project.

The workflow in this repository therefore depends on the SIR model repository, but the model source code is kept separate to avoid mixing project-specific analysis scripts with the core simulation implementation.

Conceptually:

```text
Project / workflow repository
│
├── input data preparation
├── alternative-connectome generation
├── batch experiment configuration
├── batch result summarization
└── plotting / visualization
        │
        ▼
Modified SIR model repository
│
├── SIR simulation
├── hyperparameter tuning
├── ROI-space matching
├── alternative-connectivity support
└── model evaluation        
```
The modified SIR model repository is itself a fork of the original lab SIR repository. Changed specific to this project, including support for mismatched ROI space and higher resolution-to-coarse ROI mapping, are documented in the model repository.

## Directory Structure

- Script Directory:
    
    All scripts used are in `scripts`

- Input Directory:

    All input files for creating simulation data must be stored in the `Data/` directory.

    All input files for generating scatter plot and brain surface projection must be stored in the `Result/` directory in **SIR model repository**

- Output Directory:

    Created simulation data will be saved to the `Results/` directory.

    The generated plots will be saved to the `results/` directory in **SIR model repository**

```text
project/
├── Scripts/
│   ├── build_input.py  # build main simulation data
│   ├── reduce_input_pkl_by_names.py  # alternative-connectome generation scripts
│   ├── batch_run_hypertune_modified.y  # batch hyperparameter-tuning scripts
│   ├── plot_scatter_from_folder.py  # plotting and visualization scripts
│   ├── batch_plot.sh  # batch ploy scripts
│   └── batch_summary.sh  # result summarization scripts
│
├── Data/
│   ├── input and mock data
│   ├── ROI information
│   ├── connectivity matrices
│   ├── regional-factor data
│   └── generated SIR-compatible pickle files
│
├── SIR model repository/  # the core SIR model is maintained separately and used as an external repository
│   └── results/
│       └── results_path.txt  # store specific result path for plotting
│
│
└── README.md  # This file
```

## Installation

### Requirements

This project depends on the modified SIR model repository and its Python environment. The software requirements for the SIR model are described in the model repository:

[SIR model requirements](<https://github.com/yaqi-jiao/Xiao_Tau_simulation_SIR/blob/main/requirements.txt>)

Additional packages required by the project-specific preprocessing and visualization scripts should be installed as needed.


### 1. Clone this repository

```bash
git clone <this-repository-url>
cd <repository-name>
```

### Clone the original SIR model repository

Clone the original SIR simulation repository and follow its installation instructions:

```bash
git clone <SIR-model-repository-url>
```

The SIR model repository should be placed in the expected project location, or the paths in the scripts should be updated accordingly.

### 3. Create the Python environment

Install the dependencies required by both this repository and the original SIR model.

## Usage

### 1. Build an SIR-compatible input pickle file

Use the input-building script to combine CSV files containing tau measurements, ROI names, structural connectivity, connection lengths, and ROI sizes into the pickle format expected by the SIR model.

```bash
python Scripts/local_test_build_input_from_dict.py
```

Before running the script, confirm that the input file paths and output filename are configured correctly.

### 2. Generate alternative connectomes

Use `reduce_input_pkl_by_names.py` to remove selected bilateral brain regions from an existing SIR input pickle file.

Region names should be provided as base names. For example, Hippocampus is automatically expanded to:

```txt
Left_Hippocampus
Right_Hippocampus
```

#### Specify regions directly

```bash
python Scripts/reduce_input_pkl_by_names.py \
    --input-pkl path/to/input.pkl \
    --drop-names Hippocampus Amygdala
```

This command removes both the left and right hippocampus and amygdala from the input data.

#### Read region names from a text file

```bash
python Scripts/reduce_input_pkl_by_names.py \
    --input-pkl path/to/input.pkl \
    --drop-names-txt path/to/drop_regions.txt
```

The text file should contain one base region name per line:

```txt
Hippocampus
Amygdala
Caudate
```

Without additional options, all listed regions are removed together and one output pickle file is generated.

#### Process each region independently

```bash
python Scripts/reduce_input_pkl_by_names.py \
    --input-pkl path/to/input.pkl \
    --drop-names-txt path/to/drop_regions.txt \
    --process-each-line
```

With --process-each-line, the original input pickle is reloaded for every line in the text file. A separate output file is therefore generated for each removed bilateral region pair.

### 3. Run a batch of SIR hyperparameter-tuning experiments

Use `batch_run_hypertune_modified.py` to run multiple experiments sequentially.

```bash
python Scripts/batch_run_hypertune_modified.py
```

The script currently supports two experiment modes:

#### Alternative-connectome mode

Runs the SIR model using multiple connectivity files generated by removing or modifying selected brain regions.

#### Regional-factor mode

Runs one baseline model followed by combinations of region-specific factors and parameter-variation options.

### 4. Summarize simulation results

Add one simulation result directory per line to:

```bash
results/results_path.txt
```

Then run:

```bash
bash Scripts/batch_summary.sh
```

The script processes each result directory sequentially and calls the summary pipeline with the configured connectivity type, epicenter, and input dataset.

### 5. Generate scatter plots and brain-surface visualizations

After summary files have been generated, run:

```bash
bash Scripts/batch_plot.sh
```

The script reads result directories from `results_path.txt` and generates plots for each directory.

## Notes

- Most scripts currently contain project-specific paths that must be updated before use on another computer.
- Relative paths are resolved from the current working directory.
- Brain-region names are matched exactly and are case-sensitive.
- Large datasets, sensitive data, and full simulation outputs are not included in this repository.
