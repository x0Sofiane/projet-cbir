"""
histogram_viewer.py — Fenêtre d'affichage des histogrammes de caractéristiques

Ce module définit une fenêtre PyQt6 indépendante qui s'ouvre à la demande,
et affiche sous forme de graphiques matplotlib les histogrammes associés
aux caractéristiques cochées (RGB, HSV, LBP, etc.) pour l’image requête.

Il est appelé depuis cbir_gui.py lors du clic sur "Afficher les graphiques"
"""
#----------------------------------------------------------------------------------------------------

import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from backend.extraction_caracteristiques import (
    compute_histograms, compute_lbp, compute_glcm
)

#----------------------------------------------------------------------------------------------------

class HistogramWindow(QWidget):
    """Fenêtre contenant les histogrammes matplotlib pour l’image requête"""
    def __init__(self, image_path, descripteurs_selectionnes):
        """Initialise la fenêtre et affiche les histogrammes pour les descripteurs cochés"""
        super().__init__()
        self.setWindowTitle("Histogrammes - Image requête")
        self.setMinimumSize(900, 600)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Taille : 3 de hauteur par histogramme
        hauteur_fig = max(3 * len(descripteurs_selectionnes), 6)
        self.canvas = FigureCanvas(Figure(figsize=(10, hauteur_fig)))
        self.layout.addWidget(self.canvas)

        self.axes = self.canvas.figure.subplots(len(descripteurs_selectionnes), 1)
        if len(descripteurs_selectionnes) == 1:
                self.axes = [self.axes]

        # Espacement entre les graphiques
        self.canvas.figure.subplots_adjust(hspace=2.8)

        self.image = cv2.imread(image_path)
        if self.image is None:
            return

        self.rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        self.plot_histograms(descripteurs_selectionnes)


    def plot_histograms(self, types):
        """
        Trace les histogrammes correspondant aux descripteurs demandés

        Entrée:
            types (list): Liste des descripteurs cochés.
        """
        for i, desc in enumerate(types):
            ax = self.axes[i]
            ax.clear()
            ax.set_title(f"Histogramme {desc.upper()}", fontsize=12)

            if desc.lower() == "rgb":
                rgb_hist, _ = compute_histograms(self.rgb, self.hsv)
                for idx, color in enumerate(['R', 'G', 'B']):
                    ax.plot(rgb_hist[idx], label=color)
                ax.legend()

            elif desc.lower() == "hsv":
                _, hsv_hist = compute_histograms(self.rgb, self.hsv)
                for idx, color in enumerate(['H', 'S', 'V']):
                    ax.plot(hsv_hist[idx], label=color)
                ax.legend()

            elif desc.lower() == "lbp":
                lbp = compute_lbp(self.gray)
                ax.plot(lbp)
                ax.set_xlabel("Patrons")
                ax.set_ylabel("Fréquence")

            elif desc.lower() == "glcm":
                glcm = compute_glcm(self.gray)
                ax.imshow(glcm, cmap="gray")
                ax.set_title("Matrice GLCM")

            else:
                ax.text(0.5, 0.5, f"{desc.upper()} non affichable", ha='center', va='center')
            self.canvas.draw()


