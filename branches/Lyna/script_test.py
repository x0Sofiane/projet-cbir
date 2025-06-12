"""
Date : 12/03/2025

Membre : Lyna Bennani

Ce script permet d'effectuer les calculs d'extraction des caractéristiques 
à partir d'une image et d'afficher les résultats obtenus 
leur bon fonctionnement


"""
#----------------------------------------------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from extraction_caracteristiques import compute_histograms,concatenate_histograms,normalize_histograms,compute_lbp,compute_glcm, compute_gfd
from affichage import display_original_image,display_rgb_histograms,display_hsv_histograms,display_concatenate_histograms,display_normalized_histograms,display_histograms_3d,display_image_3d_monocole,display_lbp,display_glcm,display_gfd
                       


image_path = '/home/bennani/Téléchargements/image-test.jpg'  # Remplacez par le chemin de votre image

#----------------------------------------------------------------------------------------------------
# Chargement de l'image et vérification
def load_image(image_path):
    """
    Charge une image à partir du chemin spécifié et vérifie si elle a été correctement chargée.

    Args:
        image_path (str): Chemin de l'image.

    Returns:
        numpy.ndarray: L'image chargée.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Erreur : Impossible de charger l'image. Vérifiez le chemin.")
    return image


#----------------------------------------------------------------------------------------------------
# Fonction principale pour exécuter le pipeline
def main(image_path):
    """
    Fonction principale pour exécuter toutes les étapes du pipeline.

    Args:
        image_path (str): Chemin de l'image.
    """
    # Chargement de l'image et vérification
    image = load_image(image_path)

    # Affichage de l'image originale
    display_original_image(image)

    # Convertir l'image en RGB et HSV
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calcul des histogrammes RGB et HSV
    rgb_histograms, hsv_histograms = compute_histograms(image_rgb, image_hsv)

    # Affichage des histogrammes RGB
    display_rgb_histograms(rgb_histograms)

    # Affichage des histogrammes HSV
    display_hsv_histograms(hsv_histograms)

    # Concaténation et affichage des 6 courbes (RGB + HSV)
    histograms = concatenate_histograms(rgb_histograms, hsv_histograms)

    # affichage des 6 courbes (RGB + HSV)
    affiche_concatenate_histograms(histograms)
    
    # Normalisation des histogrammes
    normalized_histograms = normalize_histograms(histograms)

    # Affichage des histogrammes concaténés et normalisés
    display_normalized_histograms(normalized_histograms)

    # Affichage des histogrammes RGB et HSV en 3D
    display_histograms_3d(rgb_histograms, hsv_histograms)

    # Affichage de l'image en 3D avec chaque canal comme une image distincte
    display_image_3d_monocole(image_rgb)

    # Calcul et affichage des LBP
    display_lbp(image)

    # Calcul et affichage du GLCM
    display_glcm(image)
    
    # Calcul et affichage des GFD
    display_gfd(image)

#----------------------------------------------------------------------------------------------------
# Point d'entrée du script
main(image_path)
#----------------------------------------------------------------------------------------------------






