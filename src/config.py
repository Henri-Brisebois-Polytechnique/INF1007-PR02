import os

# Variables pour l'extraction des données
DOSSIER_SRC = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(DOSSIER_SRC)
DOSSIER_IMAGES = os.path.join(RACINE, "asl_processed", "train")
CHEMIN_TASK = os.path.join(DOSSIER_SRC, "hand_landmarker.task")
CHEMIN_SORTIE = os.path.join(RACINE, "donnees", "asl_landmarks.csv")

# Variables pour l'entrainement du modèle et traitement des données
PERSONNES_A_TESTER = ["P1", "P2", "P3", "P4", "P5"]
GRAINE = 4
COLONNE_CIBLE = "classe"
COLONNE_PERSONNE = "personne"
CHEMIN_CSV   = os.path.join("donnees", "asl_landmarks.csv")
CHEMIN_MODELE = os.path.join("modeles", "modele.joblib")
CHEMIN_META   = os.path.join("modeles", "metadonnees.json")
