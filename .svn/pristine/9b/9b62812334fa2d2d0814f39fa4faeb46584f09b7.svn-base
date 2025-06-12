from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMenuBar, QMenu, QFileDialog, QStackedWidget, QFrame
from PyQt6.QtGui import QPixmap, QIcon, QPalette, QColor
from PyQt6.QtCore import Qt
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CBIR - Système de Recherche d'Images")
        self.setGeometry(100, 100, 1200, 800)
        
        self.theme_dark = False  # Mode clair par défaut

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.create_sidebar()  # Crée la barre latérale
        self.create_main_area()
        
        self.toggle_theme()  # Appliquer le thème après la création des éléments
        
        self.imported_image = None

    def create_sidebar(self):
        """Création de la barre latérale minimaliste"""
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(80)
        self.sidebar.setStyleSheet("background-color: #1E1E1E;")
        self.layout.addWidget(self.sidebar)

        sidebar_layout = QVBoxLayout()
        self.sidebar.setLayout(sidebar_layout)

        self.button_image = QPushButton("📷")
        self.button_database = QPushButton("📂")
        self.button_graphs = QPushButton("📊")
        self.button_settings = QPushButton("⚙️")
        self.button_theme = QPushButton("🌞")  # Bouton thème initial soleil
        
        for button in [self.button_image, self.button_database, self.button_graphs, self.button_settings, self.button_theme]:
            button.setStyleSheet("background: none; font-size: 20px; padding: 10px; color: white;")
            sidebar_layout.addWidget(button)
        
        self.button_image.clicked.connect(self.show_image_page)
        self.button_database.clicked.connect(self.show_database_page)
        self.button_graphs.clicked.connect(self.show_graph_page)
        self.button_theme.clicked.connect(self.toggle_theme)
        
        self.extra_button = QPushButton("🔧")
        self.extra_button.setStyleSheet("background: none; font-size: 20px; padding: 10px; color: white;")
        self.extra_button.setVisible(False)  # Caché par défaut
        sidebar_layout.addWidget(self.extra_button)
        
        self.button_settings.clicked.connect(self.toggle_extra_button)
        
        sidebar_layout.addStretch()
    
    def toggle_extra_button(self):
        """Affiche ou masque le bouton supplémentaire"""
        self.extra_button.setVisible(not self.extra_button.isVisible())
    
    def toggle_theme(self):
        """Change le thème entre mode sombre et mode clair"""
        self.theme_dark = not self.theme_dark
        palette = QPalette()
        if self.theme_dark:
            palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
            palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
            palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 50))
            palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
            self.button_theme.setText("🌙")
        else:
            palette.setColor(QPalette.ColorRole.Window, QColor(245, 245, 235))
            palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
            palette.setColor(QPalette.ColorRole.Button, QColor(220, 220, 220))
            palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
            self.button_theme.setText("🌞")
        self.setPalette(palette)

    def create_main_area(self):
        """Création de la zone principale épurée"""
        self.main_area = QStackedWidget()
        self.layout.addWidget(self.main_area)

        self.image_page = QWidget()
        image_layout = QVBoxLayout()
        self.image_page.setLayout(image_layout)

        self.image_label = QLabel("Aucune image chargée.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("font-size: 18px;")
        image_layout.addWidget(self.image_label)

        self.upload_button = QPushButton("Importer une Image")
        self.upload_button.setStyleSheet("background-color: #03DAC6; color: black; padding: 10px; font-size: 16px;")
        self.upload_button.clicked.connect(self.import_image)
        image_layout.addWidget(self.upload_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.main_area.addWidget(self.image_page)

    def import_image(self):
        """Ouvre un dialogue pour importer une image sans faire planter l'application"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Importer une Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
                self.image_label.setText("")
            else:
                self.image_label.setText("Erreur : Impossible de charger l'image.")

    def show_image_page(self):
        self.main_area.setCurrentWidget(self.image_page)

    def show_database_page(self):
        pass  # À implémenter

    def show_graph_page(self):
        pass  # À implémenter

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
