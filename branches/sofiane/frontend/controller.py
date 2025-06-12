### cbir_project/trunk/frontend/controller.py
from backend.extraction.features import extract_features
from backend.comparaison.metrics import compute_similarity

class Controller:
    def process_image(self, image_path):
        features = extract_features(image_path)
        return features

    def compare_images(self, img1, img2):
        feat1 = extract_features(img1)
        feat2 = extract_features(img2)
        return compute_similarity(feat1, feat2)