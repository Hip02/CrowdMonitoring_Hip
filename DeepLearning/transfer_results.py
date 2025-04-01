import os
import subprocess

# --- Configuration ---
remote_user = "hhilgers"  # <-- À remplacer
remote_host = "betelgeuse"  # <-- À remplacer (ex: 192.168.1.25)
remote_base_path = "/linux/hhilgers/Code/CrowdMonitoring_Hip/DeepLearning/results/"
local_destinations = os.path.expanduser("/Users/hippolytehilgers/Desktop/UCL_Hip/Mémoire/Code/DeepLearning/results/Regression")
# ---------------------

# --- Entrée utilisateur : nom du sous-dossier à transférer ---
local_destinations = os.path.join(local_destinations, "ResNet_Folds_debug22")
local_destinations = [f"{local_destinations}" for i in [2,3,5,8]]
subfolders = [f"Regression/ResNet_Folds_debug22/fold{i}" for i in [2,3,5,8]]

for local_dest, subfolder in zip(local_destinations, subfolders):
    # --- Construction des chemins distants et locaux ---
    os.makedirs(local_dest, exist_ok=True)
    remote_path = os.path.join(remote_base_path, subfolder)

    # --- Construction de la commande SCP ---
    scp_command = [
        "scp",
        "-r",  # option récursive
        f"{remote_user}@{remote_host}:{remote_path}",
        local_dest
    ]

    # --- Affichage et exécution de la commande ---
    print(f"\nTransfert de {remote_path} vers {local_dest}...\n")
    try:
        subprocess.run(scp_command, check=True)
        print("\n✅ Transfert terminé avec succès.")
    except subprocess.CalledProcessError as e:
        print("\n❌ Une erreur est survenue pendant le transfert.")
        print("Détail de l'erreur :", e)
