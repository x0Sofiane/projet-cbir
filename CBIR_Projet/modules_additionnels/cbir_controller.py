"""
cbir_controller.py — Contrôleur principal du système CBIR

Ce module contient la logique de traitement qui orchestre l’extraction de descripteurs,
la transformation PCA, et la comparaison avec la base de données pour renvoyer
les résultats d’images les plus similaires.
"""
#----------------------------------------------------------------------------------------------------

import numpy as np
from backend.extraction_caracteristiques import extract_all_features
from backend.calcul_distances import (
    weighted_distance,
    euclidean_distance,
    cosine_distance,
    chi2_distance,
    manhattan_distance,
    histogram_intersection,
    bhattacharyya_distance,
    hellinger_distance,
    emd_distance
)
from backend.database import charger_base_descripteurs
from modules_additionnels.pca_transform import transformer_via_pca
from sklearn.preprocessing import StandardScaler
from modules_additionnels.pca_transform import charger_scaler_et_pca
from modules_additionnels.pca_transform import charger_descripteurs_utilises

#----------------------------------------------------------------------------------------------------

# Dictionnaire des distances
MESURES = {
    "euclidean": euclidean_distance,
    "cosine": cosine_distance,
    "chi2": chi2_distance,
    "manhattan": manhattan_distance,
    "intersection": histogram_intersection,
    "bhattacharyya": bhattacharyya_distance,
    "hellinger": hellinger_distance,
    "emd": emd_distance
}

#----------------------------------------------------------------------------------------------------

def rechercher(image_rgb, base, model_pca_path, mesures_par_descripteur, top_k=5):
    """
    Recherche les images les plus similaires à une image requête.

    Entrée :
        image_rgb (np.ndarray) : Image requête RGB
        base (dict) : Base d’images indexée avec leurs descripteurs
        model_pca_path (str) : Chemin du modèle PCA
        mesures_par_descripteur (dict) : {descripteur: nom_métrique}
        top_k (int) : Nombre d’images à renvoyer

    Sortie :
        list : Liste de tuples (nom_image, score) triée par similarité
    """
    import os

    # Vérifie la présence du fichier PCA
    descripteurs_utilisés = ['rgb', 'hsv', 'lbp', 'glcm', 'gfd', 'hough']
    config_path = "pca_used_descriptors.txt"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            descripteurs_utilisés = [line.strip() for line in f if line.strip()]
        print(f"[INFO] Descripteurs utilisés pour PCA : {descripteurs_utilisés}")
    else:
        print("[AVERTISSEMENT] Aucun fichier de config PCA, utilisation des descripteurs par défaut.")

    # Extraction de la requête
    requete_vecteurs = extract_all_features(image_rgb)

    # Chargement du PCA et scaler
    try:
        pca, scaler = charger_scaler_et_pca("pca_model.pkl", "pca_scaler.pkl")
    except Exception as e:
        print(f"[Erreur chargement PCA/scaler] : {e}")
        return []

    try:
        vecteur_requete_concat = np.hstack([
            np.ravel(requete_vecteurs[d]) for d in descripteurs_utilisés if d in requete_vecteurs
        ])
        vecteur_requete_final = pca.transform(scaler.transform([vecteur_requete_concat]))[0]
    except Exception as e:
        print(f"[Erreur PCA requête] : {e}")
        return []

    scores = []
    for nom_img, descripteurs in base.items():
        try:
            vecteur_base = np.hstack([
                np.ravel(descripteurs[d]) for d in descripteurs_utilisés if d in descripteurs
            ])
            vecteur_base_final = pca.transform(scaler.transform([vecteur_base]))[0]

            # Distance : euclidean par défaut
            distance = euclidean_distance(vecteur_requete_final, vecteur_base_final)
            scores.append((nom_img, distance))
        except Exception as e:
            print(f"[Erreur image {nom_img}] : {e}")
            continue

    scores = sorted(scores, key=lambda x: x[1])
    print("[INFO] Résultats triés (Top {}) :".format(top_k))
    for nom, s in scores[:top_k]:
        print(f" - {nom} : {s:.4f}")
    return scores[:top_k]
