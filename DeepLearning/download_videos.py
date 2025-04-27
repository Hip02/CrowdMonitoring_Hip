import subprocess
import os

local_path = os.path.expanduser("~/Downloads")  # <-- corrige ici

for i in range(75, 81):
    print(f"Copying file for exp{i}...")
    remote_path = f"hhilgers@betelgeuse:/linux/hhilgers/Code/CrowdMonitoring_Hip/DeepLearning/visualizations/fold9_exp{i}_visu.mp4"
    command = ["scp", "-r", remote_path, local_path]
    subprocess.run(command)
