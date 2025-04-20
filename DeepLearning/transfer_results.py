import os
import subprocess

# --- Configuration ---
remote_user = "hhilgers"  # <-- Modifier si besoin
remote_host = "betelgeuse"  # <-- Modifier si besoin
remote_base_path = "/linux/hhilgers/Code/CrowdMonitoring_Hip/DeepLearning/results/"
local_base_path = os.path.expanduser(
    "~/Desktop/UCL_Hip/Mémoire/Code/DeepLearning/results/Regression"
)
folder_name = "AFinalExp"  # Nom de la sous-expérience

# --- Plages à transférer ---
exp_range = range(1, 3)     # Expériences 0 à 2
fold_range = range(1, 9)    # Folds 1 à 8

# --- Transfert ---
for exp_id in exp_range:
    for fold_id in fold_range:
        # Construction des chemins
        remote_path = os.path.join(
            remote_base_path, f"Regression/{folder_name}{exp_id}/fold{fold_id}"
        )
        local_path = os.path.join(
            local_base_path, f"{folder_name}{exp_id}", f"fold{fold_id}"
        )

        # Création du dossier local si nécessaire
        os.makedirs(local_path, exist_ok=True)

        # Commande SCP
        scp_command = [
            "scp", "-r",
            f"{remote_user}@{remote_host}:{remote_path}",
            local_path
        ]

        # Affichage et exécution
        print(f"\n🔄 Transfert de {remote_path} vers {local_path}...\n")
        try:
            subprocess.run(scp_command, check=True)
            print("✅ Transfert terminé avec succès.\n")
        except subprocess.CalledProcessError as e:
            print("❌ Erreur lors du transfert.")
            print("Détail :", e)
