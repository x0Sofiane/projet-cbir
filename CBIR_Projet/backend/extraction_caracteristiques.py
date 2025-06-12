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
        - compute_hough : applique la transformée de Hough pour détecter des lignes dans une image de bords (Canny).
"""

import cv2
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import local_binary_pattern
from scipy.interpolate import splprep, splev

#----------------------------------------------------------------------------------------------------
# Calcul des histogrammes RGB et HSV
def compute_histograms(image_rgb, image_hsv):
    """
    Calcule les histogrammes des canaux RGB et HSV.

    Paramètres :
        image_rgb (numpy.ndarray): L'image en RGB.
        image_hsv (numpy.ndarray): L'image en HSV.

    Retour :
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

    Paramètres :
        rgb_histograms (list): Liste des histogrammes RGB.
        hsv_histograms (list): Liste des histogrammes HSV.

    Retour :
        list: Liste des histogrammes concaténés.
    """
    
    histograms = rgb_histograms + hsv_histograms

    return histograms

#----------------------------------------------------------------------------------------------------
# Normalisation des histogrammes
def normalize_histograms(histograms):
    """
    Normalise les histogrammes pour qu'ils aient une plage de valeurs entre 0 et 1.

    Paramètres :
        histograms (list): Liste des histogrammes à normaliser.

    Retour :
        list: Liste des histogrammes normalisés.
    """
    normalized_histograms = []
    for hist in histograms:
        normalized_hist = cv2.normalize(hist, None, 0, 1, cv2.NORM_MINMAX)
        normalized_histograms.append(normalized_hist)
    return normalized_histograms

#----------------------------------------------------------------------------------------------------    

def compute_lbp(image_gray, P=8, R=1, method='uniform'):
    """
    Calcule l'histogramme LBP (Local Binary Pattern) d'une image en niveaux de gris.
    L'histogramme est normalisé et a une taille fixe selon le mode 'uniform'.

    Paramètres :
    image_gray : ndarray                 Image en niveaux de gris.
    P : int                              Nombre de points échantillonnés.
    R : float                            Rayon du cercle de voisinage.
    method : str                         Méthode LBP (par défaut : 'uniform')

    Retour :
    numpy.ndarray                       Histogramme normalisé (vecteur 59D si P=8 et method='uniform').
    """
    lbp = local_binary_pattern(image_gray, P, R, method)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


#----------------------------------------------------------------------------------------------------

def compute_glcm(image_gray, levels=8):
    """
    GLCM simplifié : histogramme des co-occurrences horizontalement à 1 pixel de distance.
    On réduit l'image à 'levels' niveaux pour que ce soit compact.

    Retourne un vecteur normalisé de co-occurrence symétrique.
    """

def compute_glcm(image_gray, levels=8):

    image = np.uint8(image_gray // (256 // levels))
    glcm = np.zeros((levels, levels), dtype=np.uint32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1] - 1):  # Horizontal voisins
            row = image[i, j]
            col = image[i, j + 1]
            glcm[row, col] += 1
            glcm[col, row] += 1  # symétrique

    return glcm

#----------------------------------------------------------------------------------------------------
def compute_gfd(image, K=10):
    """
    Calcule les Generalized Fourier Descriptors (GFD) d'une image.
        - Conversion en niveaux de gris : L'image est convertie en niveaux de gris pour faciliter la détection des contours.
        - Détection des contours : Utilisation de cv2.findContours pour détecter les contours dans l'image.
        - Transformation de Fourier : La transformation de Fourier est appliquée aux points du contour pour obtenir les descripteurs de Fourier.
        - Normalisation : Les descripteurs sont normalisés par rapport à la composante continue (premier coefficient) pour obtenir l'invariance en échelle.

    Paramètres :
        image (numpy.ndarray): L'image d'entrée en niveaux de gris.
        K (int): Nombre de descripteurs de Fourier à calculer.

    Retour :
        numpy.ndarray: Descripteurs de Fourier généralisés.    
    """

    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(K)

    contour = max(contours, key=cv2.contourArea).squeeze()
    if contour.ndim != 2 or contour.shape[0] < 3:
        return np.zeros(K)

    # Interpolation du contour si trop court
    if contour.shape[0] < K:
        contour = interpolate_contour(contour, K + 5)

    contour_complex = contour[:, 0] + 1j * contour[:, 1]
    fourier_result = np.fft.fft(contour_complex)

    if np.abs(fourier_result[0]) < 1e-6:
        return np.zeros(K)

    return np.abs(fourier_result[:K]) / np.abs(fourier_result[0])


#--------------

def interpolate_contour(contour, num_points):
    """
    Interpole un contour pour obtenir un nombre de points suffisant.
    """
    if len(contour) < 3:
        return contour

    contour = contour.astype(np.float32)
    x, y = contour[:, 0], contour[:, 1]

    tck, u = splprep([x, y], s=0, per=False)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    return np.stack((x_new, y_new), axis=-1)


#----------------------------------------------------------------------------------------------------

def extract_canny_contours(image, low_threshold=100, high_threshold=200):
    """
    Date : 13/04/2025
    Membre : Lyna Bennani
    
    Détecte les contours d'une image à l'aide de l'algorithme de Canny.

    Paramètres :
    ------------
    image : ndarray
        Image d'entrée (en niveaux de gris ou couleur).
    low_threshold : int
        Seuil inférieur pour l'hystérésis.
    high_threshold : int
        Seuil supérieur pour l'hystérésis.

    Retour :
    --------
    contours : ndarray (uint8)
        Image binaire contenant les bords détectés.
    """
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    contours = cv2.Canny(image_gray, low_threshold, high_threshold)
    return contours

#----------------------------------------------------------------------------------------------------

def compute_hough(image_contours, rho=1, theta=np.pi / 180, threshold=100):
    """
    Date : 13/04/2025
    Membre : Lyna Bennani
    
    Applique la transformée de Hough pour détecter des lignes dans une image de bords (Canny).

    Paramètres :
    ------------
    image_contours : ndarray
        Image binaire de contours (typiquement sortie de Canny).
    rho : float
        Résolution du paramètre ρ en pixels.
    theta : float
        Résolution du paramètre θ en radians.
    threshold : int
        Seuil minimum d'intersections pour détecter une ligne.

    Retour :
    --------
    lines : list
        Liste des lignes détectées au format (rho, theta).
    """
    lines = cv2.HoughLines(image_contours, rho, theta, threshold)
    if lines is not None:
        return lines[:, 0, :]  # on enlève la structure [[[]]]
    else:
        return []

#----------------------------------------------------------------------------------------------------
def extract_all_features(image):
    """
    Extrait tous les descripteurs nécessaires à la comparaison d'images

    Paramètre :
        image (ndarray) : Image couleur au format RGB

    Retour :
        dict : Dictionnaire contenant tous les descripteurs ('rgb', 'hsv', etc.)
    """
    try:
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        rgb_hist, hsv_hist = compute_histograms(image, image_hsv)
        rgb = np.concatenate([cv2.normalize(h, None, 0, 1, cv2.NORM_MINMAX).flatten() for h in rgb_hist])
        hsv = np.concatenate([cv2.normalize(h, None, 0, 1, cv2.NORM_MINMAX).flatten() for h in hsv_hist])

        lbp = compute_lbp(image_gray)
        if lbp is None or not isinstance(lbp, np.ndarray):
            lbp = np.zeros(59)

        glcm = compute_glcm(image_gray)
        if glcm is None or not isinstance(glcm, np.ndarray):
            glcm = np.zeros((8, 8))

        gfd = compute_gfd(image, K=10)
        if gfd is None or not isinstance(gfd, np.ndarray) or len(gfd) != 10:
            gfd = np.zeros(10)

        contours = extract_canny_contours(image)
        hough = compute_hough(contours)
        if hough is None or not isinstance(hough, np.ndarray):
            hough = np.zeros((50, 2))
        elif hough.shape[0] < 50:
            padding = np.zeros((50 - hough.shape[0], 2))
            hough = np.vstack((hough, padding))
        else:
            hough = hough[:50]  # Tronqué à 50

        return {
            "rgb": rgb,
            "hsv": hsv,
            "lbp": lbp,
            "glcm": glcm.flatten(),
            "gfd": gfd,
            "hough": hough.flatten()
        }

    except Exception as e:
        print(f"[Erreur - extraction] : {e}")
        return None

#----------------------------------------------------------------------------------------------------

def resize_vector(vector, target_length):
    """
    Redimensionne un vecteur à une taille fixe et si il est trop long, tronque.
    Si il est trop court, complète avec des zéros

    Entrée :
        vector (array): vecteur original
        target_length (int): taille cible

    Sortie :
        numpy.array de taille `target_length`
    """
    vector = np.ravel(vector)
    if len(vector) >= target_length:
        return vector[:target_length]
    else:
        return np.pad(vector, (0, target_length - len(vector)), mode='constant')