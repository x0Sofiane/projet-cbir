from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMenuBar, QMenu, QFileDialog, QListWidget, QStackedWidget, QGroupBox, QGraphicsView, QGraphicsScene
from PyQt6.QtGui import QPixmap
import sys

def import_image(self):
        """Ouvre un dialogue pour importer une image"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Importer une Image", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            self.imported_image = file_path
            self.initial_image_label.setPixmap(QPixmap(file_path).scaled(400, 400))
            self.initial_image_label.setText("")  # Effacer le texte initial
            self.similar_images_label.setText("Images similaires chargées.")  # Afficher un message
            self.graph_label.setText("Graphes générés.")  # Afficher un message si un graphe existe