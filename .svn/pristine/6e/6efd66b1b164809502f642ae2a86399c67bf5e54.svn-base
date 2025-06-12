from PyQt6.QtWidgets import QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget

class Ui_MainWindow:
    def setupUi(self, MainWindow):
        MainWindow.setWindowTitle("CBIR Project")
        self.central_widget = QWidget(MainWindow)
        MainWindow.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.label = QLabel("Bienvenue dans CBIR Project")
        self.button = QPushButton("Charger une image")

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.button)