# Toy Data
This folder contains small toy datasets for testing the data-preparation and plotting utilities associated with the SIR model pipeline.

The toy datasets are intentionally small and are intended for checking data structure, file handling, ROI alignment, and plotting workflows rather than for scientific analysis.

## 1. toy_for_prepare_input

This folder contains raw CSV files used to construct a standard input file for the SIR model.

The files include:

- `tau.csv`: observed regional tau values
- `roi_names.csv`: ordered ROI names
- `conn.csv`: structural connectivity matrix
- `SC_len.csv`: structural connection-length matrix
- `ROI_size.csv`: ROI sizes

These files represent the main information required by the SIR model input structure. During input preparation, they are combined into a single pickle file containing the observed tau data together with the corresponding connectivity, connection length, ROI names, and ROI sizes.

The purpose of this toy dataset is to provide a small example of how raw regional data are organized before being converted into the input format used by the SIR simulation.

## 2. toy_for_prepare_altSC

This folder contains toy data for preparing alternative structural connectivity inputs.

It includes:

- `toy_input_with_tau.pkl`: a standard SIR-style input containing both observed tau data and connectivity information
- `toy_input_conn_only.pkl`: a connectivity-only input containing connectivity, connection length, ROI labels, and ROI sizes
- `toy_drop_regions.txt`: a small list of regions to be removed

The alternative-connectivity preparation step is used when the connectivity used for simulation should contain a different set of ROIs from the original input.

Selected regions can be removed from the connectivity matrix, `SC_len`, ROI labels, and ROI-size information while keeping all remaining data structures aligned. When tau data are present, the corresponding tau rows are also updated.

This makes it possible to construct reduced or otherwise modified connectivity inputs that can later be used as alternative structural connectivity (`altSC`) in the SIR model.

## 3. toy_for_batch_plot

This folder contains small example SIR result files for testing the batch plotting workflow.

If the SIR model toy data successfully generate simulation results, those results can be used directly for testing the plotting scripts.

If suitable SIR toy results are not available or were not successfully generated, the example result folders included here can be used instead.

The directory structure of these toy results should be preserved because the plotting workflow relies on information encoded in the SIR result path, for example:

```tree
results/
 └── tau/ 
    └── presence/ 
        └── hypertune/ 
            └── entorhinal/ 
                └── <result_folder>/ 
                    ├── Fig2_Best_predictions.csv 
                    ├── Fig2_Metrics.csv 
                    └── Fig2_R_across_time.csv
```

The location of `result_path.txt` is also important:

`result_path.txt` MUST be placed inside the corresponding SIR result folder.

The batch plotting workflow uses the SIR result-folder structure to locate and interpret the result files, so moving the CSV files or result_path.txt outside this structure may prevent the plotting scripts from identifying the results correctly.