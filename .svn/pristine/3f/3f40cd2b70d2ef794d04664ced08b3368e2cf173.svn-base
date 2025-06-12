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