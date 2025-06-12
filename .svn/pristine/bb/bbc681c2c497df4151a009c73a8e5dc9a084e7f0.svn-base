
import sqlite3
import random

def creer_connexion(fichier_bdd):
    """Crée une connexion à la base de données SQLite spécifiée par fichier_bdd."""
    connexion = None
    try:
        connexion = sqlite3.connect(fichier_bdd)
        return connexion
    except sqlite3.Error as erreur:
        print(erreur)
    return connexion

def creer_table(connexion, instruction_sql_creation_table):
    """Crée une table à partir de l'instruction instruction_sql_creation_table."""
    try:
        curseur = connexion.cursor()
        curseur.execute(instruction_sql_creation_table)
    except sqlite3.Error as erreur:
        print(erreur)

def stocker_histogramme(connexion, chemin_image, histogramme):
    """Stocke l'histogramme d'une image dans la base de données."""
    instruction_sql = ''' INSERT INTO histogrammes(chemin_image, histogramme) VALUES(?, ?) '''
    curseur = connexion.cursor()
    curseur.execute(instruction_sql, (chemin_image, str(histogramme)))
    connexion.commit()
    return curseur.lastrowid

def stocker_descripteur_texture(connexion, nom_fichier, descripteurs):
    """Stocke le descripteur de texture d'une image dans la base de données."""
    instruction_sql = ''' INSERT INTO textures(nom_fichier, descripteurs) VALUES(?, ?) '''
    curseur = connexion.cursor()
    curseur.execute(instruction_sql, (nom_fichier, str(descripteurs)))
    connexion.commit()
    return curseur.lastrowid

def generer_histogramme_fictif():
    """Génère un histogramme fictif sous forme de chaîne."""
    histogramme = [random.random() for _ in range(256)]  # Exemple : 256 bins
    return str(histogramme)

def generer_descripteurs_fictifs():
    """Génère des descripteurs de texture fictifs sous forme de chaîne."""
    descripteurs = [random.random() for _ in range(18)]  # Exemple : 18 descripteurs
    return str(descripteurs)

if __name__ == "__main__":
    # Exemple d'utilisation (stockage uniquement)
    base_de_donnees = r"descripteurs_images.db"
    connexion = creer_connexion(base_de_donnees)

    if connexion is not None:
        # Créer les tables si elles n'existent pas
        instruction_sql_creation_histogrammes = """ CREATE TABLE IF NOT EXISTS histogrammes (
                                                            chemin_image TEXT PRIMARY KEY,
                                                            histogramme TEXT
                                                        ); """
        instruction_sql_creation_textures = """CREATE TABLE IF NOT EXISTS textures (
                                                            nom_fichier TEXT PRIMARY KEY,
                                                            descripteurs TEXT
                                                        );"""
        creer_table(connexion, instruction_sql_creation_histogrammes)
        creer_table(connexion, instruction_sql_creation_textures)

        # Exemple : Stocker un histogramme et un descripteur de texture fictifs
        chemin_image_fictif = "chemin/vers/image_fictive.jpg"
        nom_fichier_fictif = "image_fictive.jpg"
        histogramme_fictif = generer_histogramme_fictif()
        descripteurs_fictifs = generer_descripteurs_fictifs()

        stocker_histogramme(connexion, chemin_image_fictif, histogramme_fictif)
        stocker_descripteur_texture(connexion, nom_fichier_fictif, descripteurs_fictifs)

        connexion.close()
        print(f"Les tables 'histogrammes' et 'textures' ont été créées (si elles n'existaient pas) et des données fictives ont été stockées dans '{base_de_donnees}'.")
    else:
        print("Erreur ! impossible de créer la connexion à la base de données.")