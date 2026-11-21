import os

# Variables pour l'extraction des données
DOSSIER_SRC = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(DOSSIER_SRC)
DOSSIER_IMAGES_TRAIN= os.path.join(RACINE, "asl_processed", "train")
DOSSIER_IMAGES_TEST = os.path.join(RACINE, "asl_processed", "test")
CHEMIN_TASK = os.path.join(DOSSIER_SRC, "hand_landmarker.task")
CHEMIN_SORTIE = os.path.join(RACINE, "donnees", "asl_landmarks.csv")

# Variables pour l'entrainement du modèle et traitement des données
PERSONNES_TRAIN = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
PERSONNES_TEST = ["P9", "P10"]
GRAINE = 4
COLONNE_CIBLE = "classe"
COLONNE_PERSONNE = "personne"
CHEMIN_CSV   = os.path.join("donnees", "asl_landmarks.csv")
CHEMIN_MODELE_FORET = os.path.join("modeles", "foret", "modele.joblib")
CHEMIN_META_FORET = os.path.join("modeles", "foret", "metadonnees.json")
CHEMIN_MODELE_HGBOOST = os.path.join("modeles", "hgboost", "modele.joblib")
CHEMIN_META_HGBOOST = os.path.join("modeles", "hgboost", "metadonnees.json")
CHEMIN_MODELE_REGRESSION = os.path.join("modeles", "regressionlogistique", "modele.joblib")
CHEMIN_META_REGRESSION = os.path.join("modeles", "regressionlogistique", "metadonnees.json")
