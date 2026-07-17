#!/usr/bin/env python3
"""
Batch launcher for SIR model hyperparameter-tuning experiments.

This script automates the sequential execution of multiple experiments using
`run_hypertune.py`. Before each experiment, it updates the shared
`User_input_settings.txt` file with experiment-specific settings and then
launches the corresponding hyperparameter-tuning command in a subprocess.

Two batch modes are supported:
    1. `regional_factor`: Runs one baseline experiment followed by all combinations of:
        4 parameter-specific regional variation and regional factor maps listed in `region_factors`
        With four parameter-variation options and four regional factors, this mode
        generates 17 jobs in total (1 baseline job + 16 parameter-variation jobs)

    2. `alternative_sc`:
        Runs experiments using multiple alternative structural-connectivity files.
        Each connectivity file is evaluated separately for both Presence and Load.

For every job, the script:
    1. generates a unique experiment name;
    2. skips the job if its name is already recorded in `completed_jobs.txt`;
    3. updates the following entries in `User_input_settings.txt`
    4. constructs the command for `run_hypertune.py`;
    5. runs the command from the project scripts directory;
    6. records the experiment name in completed_jobs.txt after successful completion.

If a subprocess exits with a non-zero return code, the current job is marked as
failed and the remaining jobs are not started. Completed jobs remain recorded,
allowing the batch to be resumed without rerunning successful experiments.

Usage: python batch_run_hypertune_modified.py

"""


from pathlib import Path
import subprocess
import re


# Set file paths and directories according to your project structure.
project_dir = Path(r"E:\01_jiaoyq\02_Project_management\005_SIR_sim\Xiao_Tau_simulation_SIR")
data_dir = project_dir / "data" / "ROI_83_SC"
script_dir = project_dir / "scripts"
usr_input_file = project_dir / "scripts" / "User_input_settings.txt"
run_script = project_dir / "scripts" / "run_hypertune.py"
python_exe = project_dir / "simulation_env" / "Scripts" / "python.exe"

# This file records completed jobs, so rerunning this script will skip finished jobs.
done_file = data_dir / "completed_jobs.txt"

if done_file.exists():
    with open(done_file, "r", encoding="utf-8") as f:
        done_jobs = set(line.strip() for line in f if line.strip())
else:
    done_jobs = set()


"""
Data settings for two batch modes. Modify before running the script!
"""
# select script mode: "region_factor" or "alternative_sc"
RUN_MODE = "region_factor"

# Data setting for function 1: alternative SC batch
input_alt_sc = [
    "merged_connectomes_68_add_Accumbensarea.pkl",
    "merged_connectomes_68_add_Caudate.pkl",
    "merged_connectomes_68_add_Pallidum.pkl",
    "merged_connectomes_68_add_Putamen.pkl",
    "merged_connectomes_68_add_Thalamusproper.pkl",
    # "merged_connectomes_76_drop_Putamen.pkl",
    # "merged_connectomes_76_drop_Thalamusproper.pkl",
]

# Keep both options here, but the default mode for function 2 below uses Load only through new_protein_type.
protein_types = ["Presence", "Load"]
new_protein_type = "Load"

# Data setting for function 2: regional factor batch
# Default: baseline + 4 parameter-var options × 4 region factors = 17 jobs
# If your actual command-line argument names are different,
# only change the left side here.
parameter_var_args = [
    "spread_var",       # python run_hypertune.py --spread_var MAPT
    "synthesis_var",    # python run_hypertune.py --synthesis_var MAPT
    "clearance_var",    # python run_hypertune.py --clearance_var MAPT
    "misfold_var",      # python run_hypertune.py --misfold_var MAPT
]

# Replace these four names with actual region-factor names in the regional factor list.
region_factors = [
    "MAPT",
    "REGION_FACTOR_2",
    "REGION_FACTOR_3",
    "REGION_FACTOR_4",
]


# For the function2, use the default connectivity from User_input_settings.txt,
# unless you set this to a real file path.
# Example:
# default_connectivity_file = data_dir / "merged_connectomes_xxx.pkl"
default_connectivity_file = None



def build_alternative_sc_jobs(input_pkls):
    """
    Build experiment configurations for the alternative-sc batch mode.

    For each structural-connectivity file in `input_pkls`, this function creates one experiment for every protein 
    measurement type defined in the global `protein_types` list. The current configuration therefore produces separate 
    Presence and Load experiments for each connectivity file.
    """
    jobs = []
    exp_i = 1

    for input_pkl_name in input_pkls:
        p = Path(input_pkl_name)
        for protein_type in protein_types:
            jobs.append(
                {
                    "model_name": f"exp{exp_i}_{p.stem}_{protein_type.lower()}",
                    "connectivity_file": str(data_dir / input_pkl_name),
                    "protein_type": protein_type,
                    "extra_args": [],
                }
            )
            exp_i += 1

    return jobs


def build_region_factor_jobs():
    """
    Build experiment configurations for the regional-factor batch mode.

    The generated batch contains one baseline experiment followed by every combination of:
    - parameter-variation arguments defined in `parameter_var_args`;
    - region-factor arguments defined in `region_factors`.
    The baseline job is created without any parameter-variation arguments.

    All jobs use the global `new_protein_type` setting.
    """
    
    jobs = []

    # 1 baseline job: no parameter-var / region-factor argument
    jobs.append(
        {
            "model_name": "exp0_baseline_load",
            "protein_type": new_protein_type,
            "connectivity_file": default_connectivity_file,
            "extra_args": [],
        }
    )

    exp_i = 1
    for var_arg in parameter_var_args:
        for factor in region_factors:
            jobs.append(
                {
                    "model_name": f"exp{exp_i}_{var_arg}_{factor}_load",
                    "protein_type": new_protein_type,
                    "connectivity_file": default_connectivity_file,
                    "extra_args": [f"--{var_arg}", factor],
                }
            )
            exp_i += 1

    return jobs


def update_user_input(file_path, updates):
    """
    Update selected configuration entries in a user-input settings file.
    The function reads the complete text file and replaces lines matching each key in `updates`.
    """
    text = file_path.read_text(encoding="utf-8")

    for key, value in updates.items():
        pattern = rf"^{key}\s*=\s*.*$"
        replacement = f'{key} = "{value}"'
        text, n = re.subn(pattern, lambda m: replacement, text, flags=re.MULTILINE)

        if n == 0:
            text += f"\n{replacement}"

    file_path.write_text(text, encoding="utf-8")


def run_one_job(job):
    """
    Construct and execute the hyperparameter-tuning command for one experiment.

    The command is built using the global Python executable and the global `run_hypertune.py` path.
    When the job explicitly provides a connectivity file, the function adds the following command-line arguments:
    --connectivity_file <file_path> --SC conn
    """
    cmd = [
        str(python_exe),
        str(run_script),
    ]

    # Only add --connectivity_file when this job explicitly provides one.
    # This allows the new baseline/region-factor mode to use the default setting.
    if job.get("connectivity_file"):
        cmd.extend(["--connectivity_file", str(job["connectivity_file"]), "--SC", "conn"])

    cmd.extend(job.get("extra_args", []))

    print("Command:")
    print(" ".join(cmd))

    return subprocess.run(
        cmd,
        cwd=str(script_dir),
        check=False,
    )



def main():
    if RUN_MODE == "region_factor":
        jobs = build_region_factor_jobs()
    elif RUN_MODE == "alternative_sc":
        jobs = build_alternative_sc_jobs(input_alt_sc)
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")

    for i, job in enumerate(jobs, 1):
        name = job["model_name"]

        if name in done_jobs:
            print(f"[{i}] Skipping {name} (already done)")
            continue

        print(f"\n[{i}/{len(jobs)}] Running:")
        for k, v in job.items():
            print(f"{k}: {v}")

        update_user_input(
            file_path=usr_input_file,
            updates={
                "model_name": name,
                "input_data_name": str(project_dir / "data" / "pilot_analysis" / "Input_SIR_ROI66.pkl"),
                "protein_type": job["protein_type"],
                "epicenter_list": "ctx_lh_entorhinal",
                "simulated_protein": "tau",
            },
        )

        result = run_one_job(job)

        if result.returncode != 0:
            print(f"FAILED: {name}")
            break

        with open(done_file, "a", encoding="utf-8") as f:
            f.write(name + "\n")

    print("\nAll jobs finished.")


if __name__ == "__main__":
    main()
