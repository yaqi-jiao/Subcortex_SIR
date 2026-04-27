#!/usr/bin/env python3

from pathlib import Path
import subprocess
import re
import sys

# Set the path to your project directory
project_dir = Path(r"E:\01_jiaoyq\02_Project_management\005_SIR_sim\Xiao_Tau_simulation_SIR")
data_dir = project_dir / "data" / "ROI_83_SC"
# data_dir = project_dir / "data" / "toy_for_batch"
script_dir = project_dir / "scripts"
usr_input_file = project_dir / "scripts" / "User_input_settings.txt"
run_script = project_dir / "scripts" / "run_hypertune.py"
done_file = data_dir / "completed_jobs.txt"
python_exe = project_dir / "simulation_env" / "Scripts" / "python.exe"

if done_file.exists():
    with open(done_file, "r") as f:
        done_jobs = set(line.strip() for line in f)
else:
    done_jobs = set()

# set data
# im pretty sure it doesn't need to be hard-coded
input_alt_sc = [
    # "merged_connectomes_76.pkl",
    # "merged_connectomes_76_drop_Accumbensarea.pkl",
    # "merged_connectomes_76_drop_Caudate.pkl",
    "merged_connectomes_76_drop_Pallidum.pkl",
    "merged_connectomes_76_drop_Putamen.pkl",
    "merged_connectomes_76_drop_Thalamusproper.pkl"
]

protein_types = ["Presence", "Load"]

# fixed parameters
# fixed_params = {
#      "input_data_name": "Input_ROI_66.pkl",
#     "epicenter_list": "ctx_lh_entorhinal",
#     "simulated_protein": "tau",
# }

# change parameters

def build_jobs(input_pkls):
    jobs = []
    exp_i = 7

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


def update_user_input(file_path, updates):
    text = file_path.read_text(encoding="utf-8")

    for key, value in updates.items():
        pattern = rf'^{key}\s*=\s*.*$'
        replacement = f'{key} = "{value}"'
        text, n = re.subn(pattern, lambda m: replacement, text, flags=re.MULTILINE)

        if n == 0:
            text += f"\n{replacement}"

    file_path.write_text(text, encoding="utf-8")


def main():
    jobs = build_jobs(input_alt_sc)
    for i, job in enumerate(jobs, 1):
        name = job["model_name"]

        if name in done_jobs:
            print(f"[{i}] Skipping {name} (already done)")
            continue

        print(f"\n[{i}/{len(jobs)}] Running:")
        for k, v in job.items():
            print(f"{k}: {v}")

        update_user_input(
            file_path=usr_input_file, updates=
            {
                "model_name": name,
                "input_data_name": str(project_dir / "data" / "pilot_analysis" / "Input_SIR_ROI66.pkl" ),
                # "input_data_name": str(data_dir / "Input_ROI_10.pkl" ),
                "protein_type": job["protein_type"],
                "epicenter_list": "ctx_lh_entorhinal",
                "simulated_protein": "tau",
            }
        )

        result = subprocess.run(
        [
        str(python_exe),
        str(run_script),
        "--connectivity_file", job['connectivity_file'],
        "--SC", "conn",
    ],
        cwd=str(script_dir),
        check=False,
    )
        
        if result.returncode !=0:
            print(f"FAILED: {name}")
            break

        with open(done_file, "a") as f:
            f.write(name + "\n")


    print("\nAll jobs finished.")


if __name__ == "__main__":
    main()