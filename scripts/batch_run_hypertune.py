#!/usr/bin/env python3

# 要是可以的话希望和drop region.py结合起来，就可以让drop region生成新pkl，然后直接运行
# 或者这个脚本可以直接调用drop region的输出的名字，来修改user input

from pathlib import Path
import subprocess
import re

# Set the path to your project directory
project_dir = Path(r"E:\01_jiaoyq\02_Project_management\005_SIR_sim\Xiao_Tau_simulation_SIR")
usr_input_file = project_dir / "scripts" / "User_input_settings.txt"
run_script = project_dir / "scripts" / "run_hypertune.py"

# fixed parameters
fixed_params = {
    "epicenter_list": "ctx_lh_entorhinal",
    "simulated_protein": "tau",
}


# change parameters




def update_user_input(file_path, updates):
    text = file_path.read_text(encoding="utf-8")

    for key, value in updates.items():
        pattern = rf'^{key}\s*=\s*.*$'
        replacement = f'{key} = "{value}"'
        text, n = re.subn(pattern, replacement, text, flags=re.MULTILINE)

        if n == 0:
            text += f"\n{replacement}"

    file_path.write_text(text, encoding="utf-8")


def main():
    for i, job in enumerate(jobs, start=1):
        params = {}
        params.update(fixed_params)
        params.update(job)

        print(f"\nRunning job {i}/{len(jobs)}")
        for k, v in params.items():
            print(f"{k}: {v}")

        # 1. update User_input_settings.txt
        update_user_input(usr_input_file, params)

        # 2. run hypertune
        result = subprocess.run(
            ["python", str(run_script)],
            cwd=str(project_dir / "scripts"),
            text=True
        )

        if result.returncode != 0:
            print(f"Job failed: {job['model_name']}")
            break

    print("\nAll jobs finished.")