from PyQt6.QtWidgets import QLabel

class Viewer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("Résultats")