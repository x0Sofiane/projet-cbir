"""
Date : 12/03/2025

Membre : Lyna Bennani

- Ce package contient les fonctions suivantes :
        - compute_histograms : fonction qui calcule les histogrammes RGB et HSV d'une image
        - concatenate_histograms : fonction permettant de concaténer des histogrammes,
        - normalize_histograms : fonction permettant de normaliser des histogrammes,
        - compute_lbp : fonction pour calculer les caractéristiques LBP (Local Binary Patterns),
        - compute_glm : fonction pour calculer la GLCM d'une image (Gray-Level Co-occurrence Matrix),
        - compute_gfd : fonction pour calculer les GFD d'une image (Generalized Fourier Descriptors)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#----------------------------------------------------------------------------------------------------
# Calcul des histogrammes RGB et HSV
def compute_histograms(image_rgb, image_hsv):
    """
    Calcule les histogrammes des canaux RGB et HSV.

    Args:
        image_rgb (numpy.ndarray): L'image en RGB.
        image_hsv (numpy.ndarray): L'image en HSV.

    Returns:
        tuple: Deux listes contenant les histogrammes RGB et HSV.
    """
    rgb_histograms = []
    hsv_histograms = []

    # Calcul des histogrammes RGB
    for i in range(3):  # Pour chaque canal (R, G, B)
        hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
        rgb_histograms.append(hist)

    # Calcul des histogrammes HSV
    for i in range(3):  # Pour chaque canal (H, S, V)
        hist = cv2.calcHist([image_hsv], [i], None, [256], [0, 256])
        hsv_histograms.append(hist)

    return rgb_histograms, hsv_histograms


#----------------------------------------------------------------------------------------------------
# Concaténation des 6 histogrammes (RGB + HSV)
def concatenate_histograms(rgb_histograms, hsv_histograms):
    """
    Concatène les histogrammes RGB et HSV et les affiche.

    Args:
        rgb_histograms (list): Liste des histogrammes RGB.
        hsv_histograms (list): Liste des histogrammes HSV.

    Returns:
        list: Liste des histogrammes concaténés.
    """
    
    histograms = rgb_histograms + hsv_histograms

    return histograms

#----------------------------------------------------------------------------------------------------
# Normalisation des histogrammes
def normalize_histograms(histograms):
    """
    Normalise les histogrammes pour qu'ils aient une plage de valeurs entre 0 et 1.

    Args:
        histograms (list): Liste des histogrammes à normaliser.

    Returns:
        list: Liste des histogrammes normalisés.
    """
    normalized_histograms = []
    for hist in histograms:
        normalized_hist = cv2.normalize(hist, None, 0, 1, cv2.NORM_MINMAX)
        normalized_histograms.append(normalized_hist)
    return normalized_histograms

#----------------------------------------------------------------------------------------------------    
def compute_lbp(image, P=8, R=1):
    """
    Calcule les Local Binary Patterns (LBP) d'une image.

    Args:
        image (numpy.ndarray): L'image d'entrée en niveaux de gris.
        P (int): Nombre de points voisins.
        R (int): Rayon du cercle.

    Returns:
        numpy.ndarray: Image LBP.
    """
    lbp = np.zeros_like(image, dtype=np.uint8)
    for i in range(R, image.shape[0] - R):
        for j in range(R, image.shape[1] - R):
            center_pixel = image[i, j]
            binary_string = ''
            for p in range(P):
                x = int(i + R * np.cos(2 * np.pi * p / P))
                y = int(j - R * np.sin(2 * np.pi * p / P))
                binary_string += '1' if image[x, y] >= center_pixel else '0'
            lbp[i, j] = int(binary_string, 2)
    return lbp

#----------------------------------------------------------------------------------------------------
def compute_glcm(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    """
    Calcule la Gray-Level Co-occurrence Matrix (GLCM) d'une image.

    Args:
        image (numpy.ndarray): L'image d'entrée en niveaux de gris.
        distances (list): Liste des distances pour le calcul du GLCM.
        angles (list): Liste des angles pour le calcul du GLCM.
        levels (int): Nombre de niveaux de gris.

    Returns:
        numpy.ndarray: Matrice GLCM.
    """
    glcm = np.zeros((levels, levels), dtype=np.uint8)
    for distance in distances:
        for angle in angles:
            for i in range(image.shape[0] - distance):
                for j in range(image.shape[1] - distance):
                    delta_x = int(distance * np.cos(angle))
                    delta_y = int(distance * np.sin(angle))
                    glcm[image[i, j], image[i + delta_x, j + delta_y]] += 1
    return glcm
#----------------------------------------------------------------------------------------------------
def compute_gfd(image, K=10):
    """
    Calcule les Generalized Fourier Descriptors (GFD) d'une image.
        - Conversion en niveaux de gris : L'image est convertie en niveaux de gris pour faciliter la détection des contours.
        - Détection des contours : Utilisation de cv2.findContours pour détecter les contours dans l'image.
        - Transformation de Fourier : La transformation de Fourier est appliquée aux points du contour pour obtenir les descripteurs de Fourier.
        - Normalisation : Les descripteurs sont normalisés par rapport à la composante continue (premier coefficient) pour obtenir l'invariance en échelle.

    Args:
        image (numpy.ndarray): L'image d'entrée en niveaux de gris.
        K (int): Nombre de descripteurs de Fourier à calculer.

    Returns:
        numpy.ndarray: Descripteurs de Fourier généralisés.
        
    """
    # Convertir l'image en niveaux de gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Détecter les contours dans l'image
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prendre le plus grand contour
    contour = max(contours, key=cv2.contourArea)

    # Extraire les points du contour
    contour = contour.squeeze()
    contour_complex = contour[:, 0] + 1j * contour[:, 1]

    # Calculer la transformation de Fourier
    fourier_result = np.fft.fft(contour_complex)

    # Normaliser les descripteurs de Fourier
    gfd = np.abs(fourier_result[:K]) / np.abs(fourier_result[0])

    return gfd