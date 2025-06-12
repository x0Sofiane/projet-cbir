from PyQt6.QtGui import QPalette, QColor

theme_dark = False

def apply_theme(window, toggle=False):
    global theme_dark
    if toggle:
        theme_dark = not theme_dark

    palette = QPalette()
    if theme_dark:
        palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))  # Fond sombre
        palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))  # Texte clair
        palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 50))  # Boutons sombres
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))  # Texte boutons clair
    else:
        palette.setColor(QPalette.ColorRole.Window, QColor(245, 245, 245))  # Blanc doux pour fond
        palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))  # Texte noir
        palette.setColor(QPalette.ColorRole.Button, QColor(220, 220, 220))  # Boutons clairs
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))  # Texte boutons noir

    window.setPalette(palette)
