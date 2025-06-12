"""
Date : 13/04/2025

Membre : Lyna Bennani

Ce script permet d'effectuer les calculs d'extraction des caractéristiques 
à partir de 2 images et d'afficher la distance pondérée entre les 2 images

Distance	                 Avantages	                                                                Inconvénients	                                                                          Type de retour

Euclidienne (L2)	         Simple, rapide, standard (np.linalg.norm)	                                 Sensible aux gros écarts, ne tient pas compte de la proximité entre bins	               Distance (0 = identique)
Manhattan (L1)	             Moins sensible que L2, robuste aux petits bruits	                         Même limitation que L2	                                                                  Distance (0 = identique)
Chi²	                    Très efficace pour histogrammes de fréquences (ex. couleur, texture)	     Instable si h1 + h2 ≈ 0, nécessite petit eps                                             Distance (0 = identique)
Intersection	            Mesure de similarité (plus proche de 1 = plus similaire)	                 Pas une vraie "distance" mathématique	                                                  Similarité (0–1)
Bhattacharyya	            Bonne pour mesurer l’overlap entre deux distributions	                     Sensible aux valeurs très faibles, peut produire NaN	                                  Distance (0 = identique)
Hellinger	                Variante robuste de Bhattacharyya (bornée entre 0 et 1)	                 Moins interprétable dans certains cas	                                                  Distance (0–1)
Cosine	                    Invariante à la magnitude, compare l’orientation du vecteur	             Ne voit pas les différences d’intensité	                                              Distance (0 = identique)
EMD      	               Approximation de la Earth Mover's Distance (Wasserstein)	                 Plus lente, ne capture pas la géométrie exacte                                          Distance (0 = identique)



    Calcule la distance pondérée entre deux images à partir de leurs caractéristiques.

    Paramètres :
    img1_features : dict            Caractéristiques de l'image 1 : 'rgb', 'hsv', 'lbp', 'glcm', 'gfd', 
    img2_features : dict            Caractéristiques de l'image 2
    weights : dict                  Poids attribués à chaque descripteur

    Retour :
    float                           Distance pondérée entre les deux images
"""
#import cv2

import numpy as np
from scipy.spatial import distance

# ----------------------------------------------------------------------------------------------------
# Dictionnaire des fonctions de distance
def histogram_intersection(h1, h2):
    """
    Intersection d'histogrammes (similitude).
    Plus c'est proche de 1, plus c'est similaire.
    """
    return np.sum(np.minimum(h1, h2))

def chi2_distance(h1, h2, eps=1e-10):
    """
    Distance du chi² (χ²), robuste pour des histogrammes normalisés.
    """
    return 0.5 * np.sum(((h1 - h2) ** 2) / (h1 + h2 + eps))

def euclidean_distance(h1, h2):
    """
    Distance euclidienne (L2).
    """
    return np.linalg.norm(h1 - h2)

def manhattan_distance(h1, h2):
    """
    Distance de Manhattan (L1).
    """
    return np.sum(np.abs(h1 - h2))

def bhattacharyya_distance(h1, h2):
    """
    Distance de Bhattacharyya (0 = identique, 1 = très différent).
    """
    h1 = h1 / (np.sum(h1) + 1e-10)
    h2 = h2 / (np.sum(h2) + 1e-10)
    return -np.log(np.sum(np.sqrt(h1 * h2)) + 1e-10)

def hellinger_distance(h1, h2):
    """
    Variante normalisée de Bhattacharyya : entre 0 et 1.
    """
    h1 = h1 / (np.sum(h1) + 1e-10)
    h2 = h2 / (np.sum(h2) + 1e-10)
    return np.sqrt(1 - np.sum(np.sqrt(h1 * h2)))

def cosine_distance(h1, h2):
    """
    Distance cosinus : 0 = identique directionnellement, 1 = orthogonal.
    """
    h1 = np.ravel(h1)
    h2 = np.ravel(h2)
    return distance.cosine(h1, h2)

def emd_distance(h1, h2):
    """
    Approximation de la Earth Mover's Distance (EMD) ou Wasserstein
    Attention : h1 et h2 doivent être des histogrammes cumulés normalisés.
    """
    cdf1 = np.cumsum(h1) / (np.sum(h1) + 1e-10)
    cdf2 = np.cumsum(h2) / (np.sum(h2) + 1e-10)
    return np.sum(np.abs(cdf1 - cdf2))

# Dictionnaire global des fonctions disponibles
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

# ----------------------------------------------------------------------------------------------------
def weighted_distance(image1_desc, image2_desc, mesure, weights=None):
    """
    Calcule une distance globale pondérée entre deux images.

    Entrée:
        image1_desc (dict): Descripteurs de l'image requête
        image2_desc (dict): Descripteurs d'une image de la base
        mesure (function): Fonction de distance à utiliser
        weights (dict): Pondérations optionnelles par descripteur

    Sortie:
        float: Distance globale pondérée
    """
    total = 0.0

    for desc in image1_desc:
        h1 = np.ravel(image1_desc.get(desc))
        h2 = np.ravel(image2_desc.get(desc))

        if h1 is None or h2 is None:
            print(f"[!] Descripteur manquant : {desc} — ignoré.")
            continue

        try:
            poids = weights.get(desc, 1.0) if weights else 1.0
            total += poids * mesure(h1, h2)
        except Exception as e:
            print(f"[!] Erreur sur {desc} : {e}")
            continue

    return total
