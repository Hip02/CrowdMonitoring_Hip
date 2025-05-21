import argparse
import os
import yaml
from utils.utils import DataLoader
from networks.model import Network_Class

# Argument parser
parser = argparse.ArgumentParser(description="Run Grad-CAM analysis on one or more experiments")
parser.add_argument("-exp", "--exp_indices", type=int, nargs='+', required=True,
                    help="Indices of experiments to run (in l_exp list)")
parser.add_argument("-fold", "--fold", type=int, default=1,
                    help="Fold number to run (default=1)")
parser.add_argument("-out", "--output_dir", type=str, default="gradcam_output",
                    help="Directory to save Grad-CAM results")
args = parser.parse_args()

# Load Data
base_path = "/home/hhilgers/Dataset"
if args.fold == 9:
    all_experiments = [f"NewExp{i}" for i in range(1, 81)]
else:
    all_experiments = [f"NewExp{i}" for i in range(1, 51)]

l_exp = [f"Regression/AFinalExp{i}" for i in range(0, 40)]

fold = args.fold

for exp_index in args.exp_indices:
    try:
        exp = l_exp[exp_index]
    except IndexError:
        print(f"❌ Invalid index {exp_index}. Please choose a value between 0 and {len(l_exp) - 1}.")
        continue

    print(f"🔍 Starting Grad-CAM for {exp}, fold {fold}")

    yaml_path = os.path.join("exp_list", f"{exp}.yaml")
    if not os.path.isfile(yaml_path):
        print(f"❌ YAML file not found at {yaml_path}")
        continue

    with open(yaml_path, 'r') as stream:
        param = yaml.safe_load(stream)

    param["DATASET"]["FOLD_NUMBER"] = fold
    results_path = os.path.join("results", exp, f"fold{fold}")
    os.makedirs(results_path, exist_ok=True)

    data_loader = DataLoader(base_path, param, exp_list=all_experiments)
    myNetwork = Network_Class(data_loader, param, results_path, sub_sample_factor=1)

    myNetwork.loadWeight()  # modèle chargé automatiquement depuis le chemin par défaut

    print(f"📸 Running Grad-CAM analysis... (Channel Ablation)")
    myNetwork.save_input_channel_ablation()
    #myNetwork.analyze_gradcam(save_path=os.path.join(args.output_dir, f"{exp.replace('/', '_')}_fold{fold}"))

print("✅ Grad-CAM analysis completed for all selected experiments.")
