#!/usr/bin/env python3

from pathlib import Path
import subprocess
import re
import sys

# Set the path to your project directory
project_dir = Path(r"E:\01_jiaoyq\02_Project_management\005_SIR_sim\Xiao_Tau_simulation_SIR")
data_dir = project_dir / "data" / "review_model"
# data_dir = project_dir / "data" / "toy_for_batch"
script_dir = project_dir / "scripts"
usr_input_file = project_dir / "scripts" / "User_input_settings.txt"
run_script = project_dir / "scripts" / "run_hypertune.py"
python_exe = project_dir / "simulation_env" / "Scripts" / "python.exe"

# This file records completed jobs, so rerunning this script will skip finished jobs.
done_file = data_dir / "completed_jobs.txt"

if done_file.exists():
    with open(done_file, "r") as f:
        done_jobs = set(line.strip() for line in f)
else:
    done_jobs = set()

# set data
# im pretty sure it doesn't need to be hard-coded
input_alt_sc = [
    # "merged_connectomes_68_add_Accumbensarea.pkl",
    # "merged_connectomes_68_add_Caudate.pkl",
    # "merged_connectomes_68_add_Pallidum.pkl",
    # "merged_connectomes_68_add_Putamen.pkl",
    # "merged_connectomes_68_add_Thalamusproper.pkl",
    # "merged_connectomes_76_drop_Putamen.pkl",
    # "merged_connectomes_76_drop_Thalamusproper.pkl"
]

protein_types = ["Presence", "Load"]

# new function
RUN_MODE = "region_factor"
new_protein_type = "Load"
default_connectivity_file = None

parameter_var_args = [
    "spread_var",       # python run_hypertune.py --spread_var MAPT
    "synthesis_var",    # python run_hypertune.py --synthesis_var MAPT
    "clearance_var",    # python run_hypertune.py --clearance_var MAPT
    "misfold_var",      # python run_hypertune.py --misfold_var MAPT
]

region_factors = [
    # "MAPT",
    # "APOE_inv",
    "positiveSCD-MCI-AD",
    # "REGION_FACTOR_4",
]

# change parameters

def build_alternative_sc_jobs(input_pkls):
    """Original batch mode: multiple alternative SC files × Presence/Load."""
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
                }
            )
            exp_i += 1   
    
    return jobs


def build_region_factor_jobs():
    """New batch mode: baseline + all parameter-var × region-factor combinations."""
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
    text = file_path.read_text(encoding="utf-8")

    for key, value in updates.items():
        pattern = rf'^{key}\s*=\s*.*$'
        replacement = f'{key} = "{value}"'
        text, n = re.subn(pattern, lambda m: replacement, text, flags=re.MULTILINE)

        if n == 0:
            text += f"\n{replacement}"

    file_path.write_text(text, encoding="utf-8")


def run_one_job(job):
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
                "input_data_name": str(project_dir / "data" / "review_model" / "Input_SIR_ROI66_replace_load.pkl"),
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