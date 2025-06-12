import json

def save_features(image_path, features):
    with open("database.json", "w") as f:
        json.dump({image_path: features}, f)

def load_features(image_path):
    try:
        with open("database.json", "r") as f:
            data = json.load(f)
            return data.get(image_path, [])
    except FileNotFoundError:
        return []

def charger_base_descripteurs(chemin):
    try:
        with open(chemin, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Erreur lors du chargement de la base : {e}")
        return {}
    
def sauvegarder_base_descripteurs(base, chemin):
    """
    Sauvegarde toute la base CBIR dans un fichier JSON.
    """
    with open(chemin, "w") as f:
        json.dump(base, f, indent=2)