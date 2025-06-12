from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget

class HistoryWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Historique des Recherches")
        self.setGeometry(300, 200, 400, 600)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.title_label = QLabel("Historique des dernières recherches :")
        self.layout.addWidget(self.title_label)

        self.history_list = QListWidget()
        self.layout.addWidget(self.history_list)

    def add_entry(self, image_name, similarity_score):
        """Ajoute une entrée à l'historique"""
        entry = f"{image_name} - Score de similarité: {similarity_score:.2f}"
        self.history_list.addItem(entry)
