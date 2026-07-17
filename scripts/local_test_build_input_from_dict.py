#!/usr/bin/env python3
"""
Build and save an input dataset for the SIR tau-propagation model.

This script reconstructs the serialized input file required by the SIR simulation pipeline from several CSV files:
    - tau.csv
    - roi_names.csv
    - conn.csv
    - SC_len.csv
    - ROI_size.csv

The saved object has the following structure: 
{ "conn": { "conn": numpy.ndarray, 
            "SC_len": numpy.ndarray, 
            "name": list[str], 
            "ROI_size": dict[str, float], }, 
    "tau": pandas.DataFrame, }
"""


import pickle
import pandas as pd

# tau.csv
# Table containing regional tau measurements. 
# The first column is used as the DataFrame index. The remaining columns may contain measurements such as tau Load and Presence.
tau = pd.read_csv("tau.csv", index_col=0)

# roi_names.csv
# File containing the ROI names used by the model. The first column is read as the ordered ROI list.
roi_names = pd.read_csv("roi_names.csv").iloc[:, 0].tolist()

# conn.csv
# Structural-connectivity matrix. Both rows and columns must be labelled with ROI names.
conn = (
    pd.read_csv("conn.csv", index_col=0)
    .loc[roi_names, roi_names]
    .to_numpy()
)

# SC_len.csv
# Matrix containing structural-connection lengths. Rows and columns must use the same ROI labels as `conn.csv`
SC_len = (
    pd.read_csv("SC_len.csv", index_col=0)
    .loc[roi_names, roi_names]
    .to_numpy()
)

# ROI_size.csv
# Table containing ROI sizes. It must include the following columns:
# - ROI: ROI name;
# - size: corresponding ROI size.
roi_size_df = pd.read_csv("ROI_size.csv")

ROI_size = dict(
    zip(
        roi_size_df["ROI"],
        roi_size_df["size"]
    )
)

# pack
data = {
    "conn": {
        "conn": conn,
        "SC_len": SC_len,
        "name": roi_names,
        "ROI_size": ROI_size,
    },
    "tau": tau,
}

# save
with open("rebuilt_input_SIR_ROI66.pkl", "wb") as f:
    pickle.dump(data, f, protocol=4)

print("saved")