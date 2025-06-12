import csv
import random
import os

def stocker_histogramme_csv(chemin_fichier_csv, chemin_image, histogramme):
    """Stocke l'histogramme d'une image dans un fichier CSV."""
    mode = 'a' if os.path.exists(chemin_fichier_csv) else 'w'
    with open(chemin_fichier_csv, mode, newline='') as fichier_csv:
        writer = csv.writer(fichier_csv)
        if mode == 'w':
            writer.writerow(['chemin_image', 'histogramme'])
        writer.writerow([chemin_image, str(histogramme)])

def stocker_descripteur_texture_csv(chemin_fichier_csv, nom_fichier, descripteurs):
    """Stocke le descripteur de texture d'une image dans un fichier CSV."""
    mode = 'a' if os.path.exists(chemin_fichier_csv) else 'w'
    with open(chemin_fichier_csv, mode, newline='') as fichier_csv:
        writer = csv.writer(fichier_csv)
        if mode == 'w':
            writer.writerow(['nom_fichier', 'descripteurs'])
        writer.writerow([nom_fichier, str(descripteurs)])

def generer_histogramme_fictif():
    """Génère un histogramme fictif sous forme de chaîne."""
    histogramme = [random.random() for _ in range(256)]  # Exemple : 256 bins
    return str(histogramme)

def generer_descripteurs_fictifs():
    """Génère des descripteurs de texture fictifs sous forme de chaîne."""
    descripteurs = [random.random() for _ in range(18)]  # Exemple : 18 descripteurs
    return str(descripteurs)

if __name__ == "__main__":
    # Exemple d'utilisation (stockage uniquement en CSV)
    fichier_histogrammes_csv = "histogrammes.csv"
    fichier_textures_csv = "textures.csv"

    # Exemple : Stocker un histogramme et un descripteur de texture fictifs
    chemin_image_fictif = "chemin/vers/image_fictive.jpg"
    nom_fichier_fictif = "image_fictive.jpg"
    histogramme_fictif = generer_histogramme_fictif()
    descripteurs_fictifs = generer_descripteurs_fictifs()

    stocker_histogramme_csv(fichier_histogrammes_csv, chemin_image_fictif, histogramme_fictif)
    stocker_descripteur_texture_csv(fichier_textures_csv, nom_fichier_fictif, descripteurs_fictifs)

    print(f"Les histogrammes fictifs ont été stockés dans '{fichier_histogrammes_csv}'.")
    print(f"Les descripteurs de texture fictifs ont été stockés dans '{fichier_textures_csv}'.")