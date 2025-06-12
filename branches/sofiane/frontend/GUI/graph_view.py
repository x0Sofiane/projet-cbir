from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

class GraphView(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analyse Graphique")
        self.setGeometry(350, 250, 600, 400)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.canvas = FigureCanvas(plt.figure(figsize=(5, 4)))
        self.layout.addWidget(self.canvas)

        self.draw_sample_graph()

    def draw_sample_graph(self):
        """Affiche un graphe exemple pour tester l'interface"""
        ax = self.canvas.figure.add_subplot(111)
        x = np.arange(10)
        y = np.random.rand(10)
        ax.plot(x, y, marker="o", linestyle="-", color="b", label="Performances")
        ax.set_title("Analyse des Similarités")
        ax.set_xlabel("Comparaisons")
        ax.set_ylabel("Score")
        ax.legend()
        self.canvas.draw()
