"""
pca_transform.py — Transformation PCA pour les descripteurs d’images

Ce module applique un modèle PCA à un vecteur de descripteurs
pour réduire sa dimensionnalité, en utilisant un modèle déjà entraîné.

Utilisé principalement pour transformer l'image requête
afin de la comparer à une base déjà projetée.
"""
#----------------------------------------------------------------------------------------------------

import joblib
import numpy as np
import os

#----------------------------------------------------------------------------------------------------

def transformer_via_pca(descripteur: np.ndarray, fichier_modele_pca: str, fichier_scaler: str = "pca_scaler.pkl") -> np.ndarray:
    """
    Applique le modèle PCA et la normalisation au vecteur de descripteurs d'une image

    Entrée :
        descripteur (np.ndarray) : Vecteur 1D (features concaténées) de l’image
        fichier_modele_pca (str) : Chemin vers le modèle PCA sauvegardé (en .pkl)
        fichier_scaler (str)     : Chemin vers le scaler (.pkl) utilisé avant PCA (StandardScaler)

    Sortie :
        np.ndarray : Vecteur transformé réduit

    Exceptions :
        FileNotFoundError : Si le modèle PCA ou le scaler est absent
        ValueError        : Si les formats sont invalides
    """
    if not os.path.exists(fichier_modele_pca):
        raise FileNotFoundError(f"Modèle PCA non trouvé : {fichier_modele_pca}")
    
    if not os.path.exists(fichier_scaler):
        raise FileNotFoundError(f"Fichier de normalisation non trouvé : {fichier_scaler}")

    try:
        scaler = joblib.load(fichier_scaler)
        pca = joblib.load(fichier_modele_pca)

        descripteur = np.asarray(descripteur).flatten().reshape(1, -1)
        descripteur_norm = scaler.transform(descripteur)
        descripteur_pca = pca.transform(descripteur_norm)

        return descripteur_pca[0]
    except Exception as e:
        raise ValueError(f"Erreur lors de la transformation PCA : {str(e)}")

#----------------------------------------------------------------------------------------------------

def charger_scaler_et_pca(pca_path: str, scaler_path: str):
    """
    Charge un modèle PCA et un scaler (StandardScaler) à partir de fichiers sauvegardés

    Entrée:
        pca_path (str) :  Chemin vers le fichier contenant le modèle PCA entraîné (fichier .pkl).
        scaler_path (str) : Chemin vers le fichier contenant l'instance de StandardScaler sauvegardée (fichier .pkl)

    Sortie:
    tuple (PCA, StandardScaler)
        Le modèle PCA et le scaler chargés, prêts à être utilisés pour transformer des vecteurs.

    Exceptions:
        FileNotFoundError si l'un des fichiers n'existe pas
    """
    pca = joblib.load(pca_path)
    scaler = joblib.load(scaler_path)
    return pca, scaler

#----------------------------------------------------------------------------------------------------

def charger_descripteurs_utilises(fichier="pca_used_descriptors.txt"):
    """
    Charge la liste des descripteurs utilisés pour l'entraînement PCA.

    Entrée :
        fichier (str) : chemin vers le fichier texte listant les descripteurs utilisés

    Sortie :
        list[str] : Liste des noms de descripteurs
    """
    try:
        with open(fichier, "r") as f:
            return [ligne.strip() for ligne in f if ligne.strip()]
    except FileNotFoundError:
        raise FileNotFoundError("Fichier descripteurs PCA manquan")
