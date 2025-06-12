"""
cbir_gui.py — Interface graphique principale du projet CBIR

Ce fichier définit la fenêtre principale de l’application CBIR (Content-Based Image Retrieval)
construite avec PyQt6. Il permet d’importer une image requête, importer une base d'images, sélectionner le nombre de résultats attendus,
sélectionner descripteurs/métriques, entraîner un modèle pca avant chaque recherche et afficher les résultats ainsi que les graphiques relatifs 
à chaque caractéristique sélectionnée.

Est affiché :
- Contrôle de thème (clair/sombre)
- Boutons d'importation d'image requête et base d'images, entraînement de pca, lancement de recherche, affichage de graphiques et contrôle sur le mode clair/sombre
- Grille d'affichage des résultats
- Sélection dynamique de caractéristiques/métriques et du nombre de résultats attendus(avec un slider)
- Affichage de l'image requête 
- Fenêtre d'affichage des histogrammes
"""
#----------------------------------------------------------------------------------------------------

import sys
import os
import cv2
import joblib
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QFrame, QSlider,
    QScrollArea, QGridLayout, QCheckBox, QMessageBox
)
from PyQt6.QtGui import QPixmap, QPalette, QColor
from PyQt6.QtCore import Qt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from modules_additionnels.cbir_controller import rechercher
from backend.database import charger_base_descripteurs
from histogram_viewer import HistogramWindow

#----------------------------------------------------------------------------------------------------

class CBIRApp(QMainWindow):
    """
    Fenêtre principale de l'application CBIR
    """
    def __init__(self):
        """Initialise l'application et charge l'interface"""
        super().__init__()
        self.setWindowTitle("CBIR - Sujet L2D1 2025")
        self.setGeometry(100, 100, 1200, 800)
        self.theme_dark = True
        self.metric_selectors = {}
        self.active_descriptors = {}
        self.image_path = None
        self.nb_images = 5
        self.base = charger_base_descripteurs("database.json")
        self.dossier_base_images = "base_images" 
        self.setup_ui()
        

    def setup_ui(self):
        """
        Configure la fenêtre principale affichage général divisés en deux parties distinctes : 
        - la barre latérale/marge
        - la partie affichage de résultats 
        """
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout_principal = QHBoxLayout()
        self.central_widget.setLayout(self.layout_principal)
        self.init_sidebar()
        self.init_affichage()
        self.toggle_theme()

    def init_sidebar(self):
        """Crée la marge avec les boutons/sélecteurs et affichage de l'image requête """
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(260)
        self.sidebar_layout = QVBoxLayout()
        self.sidebar.setLayout(self.sidebar_layout)
        self.layout_principal.addWidget(self.sidebar)

        # Boutons de la partie supérieure de la barre latérale 
        self.import_btn = QPushButton("🖼 Image requête")
        self.import_btn.clicked.connect(self.import_image)
        self.sidebar_layout.addWidget(self.import_btn)

        self.search_btn = QPushButton("🔍 Lancer la recherche")
        self.search_btn.clicked.connect(self.lancer_recherche)
        self.sidebar_layout.addWidget(self.search_btn)

        self.import_base_btn = QPushButton("📁 Importer base")
        self.import_base_btn.clicked.connect(self.importer_base)
        self.sidebar_layout.addWidget(self.import_base_btn)

        self.pca_btn = QPushButton("⚙️ Entraîner PCA")
        self.pca_btn.clicked.connect(self.lancer_pca)
        self.sidebar_layout.addWidget(self.pca_btn)
        
        # Slider top_k pour choisir le nombre d'image résultats
        self.sidebar_layout.addWidget(QLabel("Nb d'images à afficher :"))
        self.slider_k = QSlider(Qt.Orientation.Horizontal)
        self.slider_k.setMinimum(1)
        self.slider_k.setMaximum(20)
        self.slider_k.setValue(self.nb_images)
        self.slider_k.valueChanged.connect(self.update_top_k)
        self.sidebar_layout.addWidget(self.slider_k)
        self.label_k_val = QLabel(str(self.nb_images))
        self.sidebar_layout.addWidget(self.label_k_val)

        # Cases à cocher pour les descripteurs
        for desc in ['rgb', 'hsv', 'lbp', 'glcm', 'gfd', 'hough']:
            check = QCheckBox(desc.upper())
            check.setChecked(True)
            self.active_descriptors[desc] = check
            self.sidebar_layout.addWidget(check)

            combo = QComboBox()
            combo.addItems(['euclidean', 'chi2', 'cosine', 'manhattan', 'emd', 'bhattacharyya', 'intersection', 'hellinger'])
            self.metric_selectors[desc] = combo
            self.sidebar_layout.addWidget(combo)

        self.sidebar_layout.addStretch()

        # Espace d’aperçu de l’image requête
        self.image_apercu = QLabel("Aperçu image requête")
        self.image_apercu.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_apercu.setFixedSize(220, 220)
        self.image_apercu.setStyleSheet("border: 1px solid gray; margin-top: 10px;")
        self.sidebar_layout.addWidget(self.image_apercu)


        self.graph_btn = QPushButton("📊 Afficher les graphiques")
        self.graph_btn.clicked.connect(self.afficher_histogrammes)
        self.sidebar_layout.addWidget(self.graph_btn)

        self.theme_btn = QPushButton("🌞")
        self.theme_btn.clicked.connect(self.toggle_theme)
        self.sidebar_layout.addWidget(self.theme_btn)

    def update_top_k(self):
        """Met à jour le nombre d’images affichées (= valeur de top_k)"""
        self.nb_images = self.slider_k.value()
        self.label_k_val.setText(str(self.nb_images))

    def init_affichage(self):
        """Configuration de la zone d'affichage des résultats"""
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.zone_affichage = QWidget()
        self.scroll.setWidget(self.zone_affichage)
        self.grille = QGridLayout()
        self.zone_affichage.setLayout(self.grille)
        self.layout_principal.addWidget(self.scroll)

    def toggle_theme(self):
        """Permet de changer le thème entre sombre et clair"""
        self.theme_dark = not self.theme_dark
        palette = QPalette()
        if self.theme_dark:
            palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
            palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
            self.theme_btn.setText("🌞")
            sidebar_color, content_color, border_color = "#1E1E1E", "#2C2C2C", "#444"
        else:
            palette.setColor(QPalette.ColorRole.Window, QColor(245, 245, 245))
            palette.setColor(QPalette.ColorRole.WindowText, QColor(20, 20, 20))
            self.theme_btn.setText("🌙")
            sidebar_color, content_color, border_color = "#F5F5F5", "#F7F7F7", "#999"

        self.setPalette(palette)
        self.sidebar.setStyleSheet(f"background-color: {sidebar_color}; border-right: 2px solid {border_color}; border-radius: 10px;")
        self.zone_affichage.setStyleSheet(f"background-color: {content_color}; border-radius: 12px;")
        for widget in self.sidebar.findChildren((QLabel, QPushButton, QCheckBox, QComboBox)):
            widget.setStyleSheet("color: black;" if not self.theme_dark else "color: white;")

    def import_image(self):
        """Importe une image requête via une boîte de dialogue, vérifie sa validité, et l'affiche dans la barre latérale"""
        fichier, _ = QFileDialog.getOpenFileName(self, "Choisir une image", "", "Images (*.jpg *.jpeg *.png)")
        if fichier and os.path.exists(fichier) and cv2.imread(fichier) is not None:
            self.image_path = fichier

            # Affichage de l'image dans l'espace prévu à cet effet
            if hasattr(self, "image_apercu"):
                pixmap = QPixmap(self.image_path).scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio)
                self.image_apercu.setPixmap(pixmap)
        else:
            QMessageBox.critical(self, "Erreur", "Image invalide ou introuvable.")


    def importer_base(self):
        """Importe une base d'images depuis un dossier et vérifie la validité de chacune des images"""
        dossier = QFileDialog.getExistingDirectory(self, "Choisir un dossier d'images")
        if dossier:
            try:
                from modules_additionnels.image_dataset_loader import generer_base_depuis_dossier
                self.base = generer_base_depuis_dossier(dossier, pca_model_path="pca_model.pkl")
                self.dossier_base_images = dossier
                if not self.base:
                    QMessageBox.warning(self, "Attention", "Aucune image valide trouvée.")
                else:
                    QMessageBox.information(self, "Succès", f"{len(self.base)} images chargées.")
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Erreur lors de l'importation : {str(e)}")

    def lancer_recherche(self):
        """Lance la recherche et affiche les résultats"""
        if not self.image_path:
            QMessageBox.critical(self, "Erreur", "Veuillez d'abord sélectionner une image requête.")
            return

        if not os.path.exists("pca_model.pkl"):
            QMessageBox.critical(self, "Erreur", "Le fichier du modèle PCA est manquant.")
            return

        if not self.base or len(self.base) < self.nb_images:
            QMessageBox.critical(self, "Erreur", "La base est vide ou insuffisante.")
            return

        mesures = {desc: box.currentText() for desc, box in self.metric_selectors.items() if self.active_descriptors[desc].isChecked()}
        if not mesures:
            QMessageBox.critical(self, "Erreur", "Veuillez cocher au moins une caractéristique.")
            return

        try:
            image_rgb = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
            resultats = rechercher(image_rgb, self.base, "pca_model.pkl", mesures, self.nb_images)
            self.afficher_resultats(resultats)
            QMessageBox.information(self, "Succès", f"{len(resultats)} résultats affichés.")
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            QMessageBox.critical(self, "Erreur interne", f"Une erreur est survenue :\n{str(e)}")

    def afficher_resultats(self, resultats):
        """
        Affiche les images résultant de la recherche triées par similarité(ordre croissant) dans la grille avec leurs 
        scores de smiliraité figurant juste en dessous de celles-ci 
        """
        # Nettoyage de la grille en premier lieu 
        for i in reversed(range(self.grille.count())):
            widget = self.grille.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        print(f"[INFO] {len(resultats)} résultats à afficher...")

        images_affichées = 0

        for idx, (nom, score) in enumerate(resultats):
            # Recherche de l'image dans les dossiers possibles
            chemins_possibles = [
                os.path.join(self.dossier_base_images, nom),
                os.path.join("base_images", nom)
            ]
            chemin_complet = next((p for p in chemins_possibles if os.path.exists(p)), None)

            if not chemin_complet:
                print(f"[ERREUR] Fichier non trouvé : {nom}")
                continue

            try:
                # Chargement et mise à l'échelle
                pixmap = QPixmap(chemin_complet).scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio)

                # Création du widget image + score
                container = QWidget()
                layout = QVBoxLayout()
                layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

                image_label = QLabel()
                image_label.setPixmap(pixmap)
                score_label = QLabel(f"Score : {score:.4f}")
                score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

                layout.addWidget(image_label)
                layout.addWidget(score_label)
                container.setLayout(layout)

                # Ajout dans la grille
                self.grille.addWidget(container, images_affichées // 5, images_affichées % 5)
                images_affichées += 1

            except Exception as e:
                print(f"[ERREUR Affichage] {nom} : {e}")
                continue

        print(f"[INFO] {images_affichées} images affichées avec scores.")

    def afficher_histogrammes(self):
        """Affiche les graphiques des descripteurs sélectionnés dans une nouvelle fenêtre"""
        if not self.image_path:
            QMessageBox.critical(self, "Erreur", "Veuillez d'abord sélectionner une image requête.")
            return
        descripteurs = [d.upper() for d in self.active_descriptors if self.active_descriptors[d].isChecked()]
        if not descripteurs:
            QMessageBox.warning(self, "Avertissement", "Aucun descripteur sélectionné.")
            return
        fenetre = HistogramWindow(self.image_path, descripteurs)
        fenetre.show()
        self.histogram_fenetre = fenetre

    def get_descripteurs_actifs(self):
        """
        Retourne la liste des descripteurs cochés
        """
        return [desc for desc, box in self.active_descriptors.items() if box.isChecked()]

    def lancer_pca(self):
        """
        Entraîne le modèle PCA sur les vecteurs correspondant aux descripteurs cochés,
        les normalise, les transforme, puis sauvegarde le modèle PCA, le scaler
        et les descripteurs utilisés dans un fichier texte
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        import joblib
        import numpy as np

        if not self.base:
            QMessageBox.warning(self, "Erreur", "Aucune base n'a été chargée.")
            return

        # Récupère les descripteurs cochés
        descripteurs_utilisés = [
            desc for desc, check in self.active_descriptors.items() if check.isChecked()
        ]
        if not descripteurs_utilisés:
            QMessageBox.warning(self, "Erreur", "Aucun descripteur sélectionné.")
            return

        vecteurs = []
        for _, descripteurs in self.base.items():
            try:
                vecteur = np.hstack([
                    np.ravel(descripteurs[d]) for d in descripteurs_utilisés
                ])
                vecteurs.append(vecteur)
            except Exception as e:
                print(f"[Ignoré] Image avec vecteurs non valides : {e}")

        if len(vecteurs) < 5:
            QMessageBox.critical(self, "Erreur", "Pas assez de vecteurs valides pour entraîner le PCA.")
            return

        try:
            X = np.array(vecteurs)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA(n_components=0.95)  # Garde 95% des informations(variance)
            pca.fit(X_scaled)
            n_components = pca.n_components_ 


            joblib.dump(pca, "pca_model.pkl")
            joblib.dump(scaler, "pca_scaler.pkl")

            # Enregistre aussi les descripteurs utilisés dans un fichier texte
            with open("pca_used_descriptors.txt", "w") as f:
                for desc in descripteurs_utilisés:
                    f.write(desc + "\n")

            QMessageBox.information(self, "Succès", f"PCA entraîné et sauvegardé avec {n_components} composantes.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur PCA", f"Erreur pendant l'entraînement PCA : {str(e)}")


#----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    fenetre = CBIRApp()
    fenetre.show()
    sys.exit(app.exec())
