
""" 
pca_training.py — Entraînement du modèle PCA pour le projet CBIR

Ce script permet de construire un modèle PCA (réduction de dimension de vecteur) à partir 
des vecteurs d’images extraits dans une base. Ce modèle sera ensuite utilisé
lors de la recherche pour transformer les vecteurs de requête et ceux de la base

Le fichier généré est enregistré sous 'pca_model.pkl'

Entrée attendue :
    'database.json' : base d’images contenant les descripteurs

Sortie :
    'pca_model.pkl' : modèle PCA entraîné avec scikit-learn
"""

import json
import numpy as np
from sklearn.decomposition import PCA
import joblib

def charger_base(chemin):
    """Charge les vecteurs depuis le fichier JSON"""
    with open(chemin, "r") as f:
        data = json.load(f)
    return data

def vectoriser_base(base):
    """Transforme les descripteurs de la base en vecteur réduits"""
    vecteurs = []
    for nom, descs in base.items():
        try:
            vecteur = []
            for val in descs.values():
                arr = np.array(val).flatten()
                vecteur.extend(arr)
            vecteurs.append(vecteur)
        except Exception as e:
            print(f"Erreur de vectorisation sur {nom}: {e}")
    return np.array(vecteurs)

def entrainer_pca(vecteurs, n_components=100):
    """Entraîne un PCA sur la base vectorisée"""
    pca = PCA(n_components=min(n_components, vecteurs.shape[1]))
    pca.fit(vecteurs)
    return pca

if __name__ == "__main__":
    try:
        print("Chargement de la base")
        base = charger_base("database.json")

        print(f"{len(base)} images chargées.")

        vecteurs = vectoriser_base(base)
        print(f"Dimensions du jeu de données : {vecteurs.shape}")

        print("Entraînement du modèle PCA")
        pca = entrainer_pca(vecteurs)

        joblib.dump(pca, "pca_model.pkl")
        print("Modèle PCA sauvegardé dans pca_model.pkl")

    except Exception as e:
        print(f"Erreur pendant l'entraînement : {e}")
