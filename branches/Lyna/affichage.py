"""
Date : 12/03/2025

Membre : Lyna Bennani

Ce package affiche le résultat des fonctions provenant de extraction_caracteristiques:
    - display_original_image : fonction permettant d'afficher l'image originale,
    - display_rgb_histograms : fonction permettant d'afficher les histogrammes RGB,
    - display_hsv_histograms : fonction permettant d'afficher les histogrammes HSV,
    - display_concatenate_histograms : fonction permettant d'afficher les histogrammes RGB et HSV concaténés,
    - display_normalized_histograms : fonction permettant d'afficher les histogrammes RGB et HSV normlisés et concaténés,
    - display_lbp : fonction permettant d'afficher les LBP d'une image (Local Binary Patterns),
    - display_image_3D_monocole : fonction permettant l'affichage de l'image en 3D avec chaque canal de couleur représenté comme une image distincte.
    - display_histogram_3D : fonction permettant d'afficher les histogrammes RGB et HSV en 3D.
    - display_glcm :fonction permettant d'afficher la GLCM d'une image (Gray-Level Co-occurrence Matrix),
    - display_gfd : fonction permettant d'afficher les GFD d'une image (Generalized Fourier Descriptor).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from extraction_caracteristiques import compute_histograms,concatenate_histograms,normalize_histograms,compute_lbp,compute_glcm, compute_gfd

#----------------------------------------------------------------------------------------------------
# Affichage de l'image originale
def display_original_image(image):
    """
    Affiche l'image originale.

    Args:
        image (numpy.ndarray): L'image à afficher.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
    plt.axis('off')
    plt.show()

#----------------------------------------------------------------------------------------------------
# Affichage des histogrammes RGB
def display_rgb_histograms(rgb_histograms):
    """
    Affiche les histogrammes des canaux RGB.

    Args:
        rgb_histograms (list): Liste des histogrammes RGB.
    """
    rgb_colors = ('r', 'g', 'b')
    rgb_names = ('Rouge', 'Vert', 'Bleu')
    plt.figure(figsize=(18, 6))
    plt.suptitle('Histogrammes RGB - Avant Normalisation')
    for i, (hist, color, name) in enumerate(zip(rgb_histograms, rgb_colors, rgb_names)):
        plt.subplot(1, 3, i + 1)
        plt.plot(hist, color=color)
        plt.title(f'{name} - Avant Normalisation')
        plt.xlim([0, 256])
        plt.xlabel('Intensité des pixels')
        plt.ylabel('Nombre de pixels')
    plt.tight_layout()
    plt.show()

#----------------------------------------------------------------------------------------------------
# Affichage des histogrammes HSV
def display_hsv_histograms(hsv_histograms):
    """
    Affiche les histogrammes des canaux HSV.

    Args:
        hsv_histograms (list): Liste des histogrammes HSV.
    """
    hsv_colors = ('m', 'c', 'y')
    hsv_names = ('Teinte', 'Saturation', 'Valeur')
    plt.figure(figsize=(18, 6))
    plt.suptitle('Histogrammes HSV - Avant Normalisation')
    for i, (hist, color, name) in enumerate(zip(hsv_histograms, hsv_colors, hsv_names)):
        plt.subplot(1, 3, i + 1)
        plt.plot(hist, color=color)
        plt.title(f'{name} - Avant Normalisation')
        plt.xlim([0, 256])
        plt.xlabel('Intensité des pixels')
        plt.ylabel('Nombre de pixels')
    plt.tight_layout()
    plt.show()

#----------------------------------------------------------------------------------------------------
# ffichage des 6 histogrammes (RGB + HSV)
def display_concatenate_histograms(histograms):
    """
    Concatène les histogrammes RGB et HSV et les affiche.

    Args:
        histograms (list): Liste des histogrammes RGB et HSV.
    """
    rgb_colors = ('r', 'g', 'b')
    hsv_colors = ('m', 'c', 'y')
    rgb_names = ('Rouge', 'Vert', 'Bleu')
    hsv_names = ('Teinte', 'Saturation', 'Valeur')

    plt.figure(figsize=(18, 6))
    plt.title('Concaténation des 6 courbes - Avant Normalisation')
    for i, (hist, color, name) in enumerate(zip(
        histograms,
        rgb_colors + hsv_colors,
        rgb_names + hsv_names
    )):
        plt.plot(hist, color=color, label=f'{name} - Avant Normalisation')
    plt.xlim([0, 256])
    plt.xlabel('Intensité des pixels')
    plt.ylabel('Nombre de pixels')
    plt.legend()
    plt.show()
    return histograms

#----------------------------------------------------------------------------------------------------
# Affichage des histogrammes concaténés et normalisés
def display_normalized_histograms(normalized_histograms):
    """
    Affiche les histogrammes concaténés et normalisés.

    Args:
        normalized_histograms (list): Liste des histogrammes normalisés.
    """
    rgb_colors = ('r', 'g', 'b')
    hsv_colors = ('m', 'c', 'y')
    rgb_names = ('Rouge', 'Vert', 'Bleu')
    hsv_names = ('Teinte', 'Saturation', 'Valeur')

    plt.figure(figsize=(18, 6))
    plt.title('Histogrammes Concaténés et Normalisés')
    for i, (hist, color, name) in enumerate(zip(
        normalized_histograms,
        rgb_colors + hsv_colors,
        rgb_names + hsv_names
    )):
        plt.plot(hist, color=color, label=f'{name} - Normalisé')
    plt.xlim([0, 256])
    plt.xlabel('Intensité des pixels')
    plt.ylabel('Nombre de pixels (normalisé)')
    plt.legend()
    plt.show()

#----------------------------------------------------------------------------------------------------
# Affichage des histogrammes RGB et HSV en 3D
def display_histograms_3d(rgb_histograms, hsv_histograms):
    """
    Affiche les histogrammes RGB et HSV en 3D.

    Args:
        rgb_histograms (list): Liste des histogrammes RGB.
        hsv_histograms (list): Liste des histogrammes HSV.
    """
    rgb_colors = ('r', 'g', 'b')
    hsv_colors = ('m', 'c', 'y')
    rgb_names = ('Rouge', 'Vert', 'Bleu')
    hsv_names = ('Teinte', 'Saturation', 'Valeur')
    histograms = rgb_histograms + hsv_histograms

    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Histogrammes RGB et HSV en 3D')

    # Création des données pour l'affichage 3D
    x = np.arange(256)
    y = np.arange(len(histograms))
    X, Y = np.meshgrid(x, y)
    Z = np.array(histograms).reshape(len(histograms), 256)

    # Affichage des histogrammes en 3D
    for i in range(len(histograms)):
        ax.bar(X[i], Z[i], zs=i, zdir='y', alpha=0.8, color=rgb_colors[i] if i < 3 else hsv_colors[i-3], label=rgb_names[i] if i < 3 else hsv_names[i-3])

    ax.set_xlabel('Intensité des pixels')
    ax.set_ylabel('Canaux de couleur')
    ax.set_zlabel('Nombre de pixels')
    ax.set_yticks(y)
    ax.set_yticklabels(rgb_names + hsv_names)
    ax.legend()
    plt.show()

#----------------------------------------------------------------------------------------------------
# Affichage de l'image en 3D avec chaque canal comme une image distincte
def display_image_3d_monocole(image):
    """
    Affiche l'image en 3D avec chaque canal de couleur représenté comme une image distincte.

    Args:
        image (numpy.ndarray): L'image à afficher en 3D.
    """
    # Convertir l'image en RGB si elle ne l'est pas déjà
    if image.shape[2] == 3:
        image_rgb = image
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extraire les dimensions de l'image
    h, w, d = image_rgb.shape

    # Créer une grille pour les axes X et Y
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Créer une figure 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Définir les positions Z pour chaque canal
    z_r = 0
    z_g = image_rgb[:, :, 0].max() + 10  # Espacement entre les canaux
    z_b = z_g + image_rgb[:, :, 1].max() + 10

    # Tracer chaque canal de couleur comme une image distincte
    ax.contourf(x, y, image_rgb[:, :, 0], zdir='z', offset=z_r, cmap='Reds', alpha=0.8)
    ax.contourf(x, y, image_rgb[:, :, 1], zdir='z', offset=z_g, cmap='Greens', alpha=0.8)
    ax.contourf(x, y, image_rgb[:, :, 2], zdir='z', offset=z_b, cmap='Blues', alpha=0.8)

    ax.set_xlabel('Largeur')
    ax.set_ylabel('Hauteur')
    ax.set_zlabel('Canaux')
    ax.set_zticks([z_r, z_g, z_b])
    ax.set_zticklabels(['Rouge', 'Vert', 'Bleu'])
    plt.title('Représentation 3D des canaux RGB de l\'image')
    plt.show()
    
#----------------------------------------------------------------------------------------------------
def display_lbp(image):
    """
    Affiche les Local Binary Patterns (LBP) d'une image.

    Args:
        image (numpy.ndarray): L'image d'entrée.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_image = compute_lbp(gray_image)

    plt.figure(figsize=(8, 6))
    plt.imshow(lbp_image, cmap='gray')
    plt.title('Local Binary Patterns (LBP)')
    plt.axis('off')
    plt.show()

#----------------------------------------------------------------------------------------------------
def display_glcm(image):
    """
    Affiche la Gray-Level Co-occurrence Matrix (GLCM) d'une image.

    Args:
        image (numpy.ndarray): L'image d'entrée.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = compute_glcm(gray_image)

    plt.figure(figsize=(8, 6))
    plt.imshow(glcm, cmap='gray')
    plt.title('Gray-Level Co-occurrence Matrix (GLCM)')
    plt.colorbar()
    plt.show()    
#----------------------------------------------------------------------------------------------------
def display_gfd(image, K=10):
    """
    Affiche les Generalized Fourier Descriptors (GFD) d'une image.

    Args:
        image (numpy.ndarray): L'image d'entrée.
        K (int): Nombre de descripteurs de Fourier à calculer.
    """
    gfd = compute_gfd(image, K)

    plt.figure(figsize=(10, 6))
    plt.stem(range(len(gfd)), gfd)
    plt.title('Generalized Fourier Descriptors (GFD)')
    plt.xlabel('Indice du descripteur')
    plt.ylabel('Amplitude normalisée')
    plt.show()