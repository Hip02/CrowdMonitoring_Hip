import subprocess

# Dossier source local
LOCAL_BASE_DIR = "/Volumes/HIP_BACKUP/MEMOIRE/Code/MyDB2"

# Dossier destination distant
REMOTE_USER = "betelgeuse"
REMOTE_BASE_DIR = "/linux/hhilgers/Dataset/"

# Commande rsync unique pour tout le dataset
rsync_command = f'rsync -av --progress {LOCAL_BASE_DIR}/ {REMOTE_USER}:{REMOTE_BASE_DIR}'
subprocess.run(rsync_command, shell=True)

print("\n✅ Transfert complet terminé avec structure préservée !")