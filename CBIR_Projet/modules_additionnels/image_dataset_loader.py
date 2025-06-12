"""
image_dataset_loader.py — Chargement d’une base d’images et extraction des descripteurs 

Ce module contient la fonction permettant de parcourir un dossier d’images,
d’extraire les descripteurs visuels (RGB, HSV, etc.) pour chaque image, et
de retourner une base structurée prête pour l’indexation et la recherche.

Il est utilisé pour importer une base via l’interface graphique
"""
#----------------------------------------------------------------------------------------------------

import os
import cv2
import numpy as np
import shutil
from backend.extraction_caracteristiques import extract_all_features
from backend.database import sauvegarder_base_descripteurs
from modules_additionnels.pca_transform import transformer_via_pca

#----------------------------------------------------------------------------------------------------

def generer_base_depuis_dossier(dossier, pca_model_path="pca_model.pkl"):
    """
    Génère une base de descripteurs à partir d'un dossier d'images et copie les images valides dans base_images/
    pour éviterles erreurss concernant les chemins 
    
    Entrée :
        dossier (str): Chemin du dossier contenant les images
        pca_model_path (str): Chemin vers le modèle PCA (optionnel)

    Sortie :
        dict: Base d'images avec descripteurs associés
    """
    base = {}
    
    if not os.path.exists(dossier):
        print(f"[ERREUR] Dossier non trouvé : {dossier}")
        return base
    
    fichiers = [f for f in os.listdir(dossier) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not fichiers:
        print("[INFO] Aucun fichier image trouvé dans le dossier.")
        return base

    dossier_cible = "base_images"
    os.makedirs(dossier_cible, exist_ok=True)

    for nom_fichier in fichiers:
        chemin_image = os.path.join(dossier, nom_fichier)
        img = cv2.imread(chemin_image)
        
        if img is None:
            print(f"[IGNORÉ] Image non lisible : {nom_fichier}")
            continue

        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            features = extract_all_features(img_rgb)

            features_trans = {}
            for k, v in features.items():
                v_array = np.array(v)
                try:
                    if os.path.exists(pca_model_path):
                        v_array = transformer_via_pca(v_array, pca_model_path)
                    features_trans[k] = v_array.tolist()
                except Exception as e:
                    print(f"[WARN] PCA échoué pour '{k}' ({nom_fichier}) : {e}")
                    features_trans[k] = v_array.tolist()

            base[nom_fichier] = features_trans

            chemin_cible = os.path.join(dossier_cible, nom_fichier)
            if not os.path.exists(chemin_cible):
                shutil.copy2(chemin_image, chemin_cible)

        except Exception as e:
            print(f"[ERREUR] Échec sur {nom_fichier} : {e}")
            continue

    if base:
        sauvegarder_base_descripteurs(base, "database.json")
        print(f"[SUCCÈS] Base enregistrée avec {len(base)} images.")
    else:
        print("[INFO] Aucune image valide extraite.")

    return base