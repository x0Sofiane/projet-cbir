"""
run.py — Script de lancement principal de l'application CBIR

Ce script vérifie que l'interface graphique principale est bien présente,
et la lance avec l'interpréteur Python courant.
"""
#----------------------------------------------------------------------------------------------------

import subprocess
import sys
import os

#----------------------------------------------------------------------------------------------------

if not os.path.exists("cbir_gui.py"):
    print("cbir_gui.py est introuvable dans le dossier courant.")
    sys.exit(1)

try:
    subprocess.run([sys.executable, "cbir_gui.py"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Erreur lors de l'exécution de l'application : {e}")
