from config import GRAINE, COLONNE_CIBLE, PERSONNES_TRAIN, CHEMIN_CSV, CHEMIN_MODELE, CHEMIN_META
from donnees import JeuDonnees, charger_csv
from tools import chronometrer
from exceptions import EntrainementImpossibleError

import os

import json
 
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class Modele:
    """Modele qui peut être entraîné sur un jeu de données traîté"""

    def __init__(self) -> None:
        self._foret = self.construire_modele()

    @staticmethod
    def construire_modele() -> RandomForestClassifier:
        """Construit la forêt aléatoire (non entraînée).
    
        Renvoie : un RandomForestClassifier de 200 arbres, avec une graine fixe.
        """
        return RandomForestClassifier(n_estimators=200, random_state=GRAINE, n_jobs=-1)

    @chronometrer
    def entrainer(self, X: pd.DataFrame, y: pd.Series) -> "Modele":
        """Entraîne le modèle sur les données d'entraînement.
    
        Reçoit : le modèle, les caractéristiques X et la cible y.
        Renvoie : le même modèle, entraîné.

        """
        if len(X) == 0:
            raise EntrainementImpossibleError("Aucune donnée d'entraînement")
        if len(X) != len(y):
            raise EntrainementImpossibleError(f"X contient {len(X)} lignes mais y en contient {len(y)}")
         
        self._foret.fit(X, y) #entraine le modèle
        return self

    def creer_metadonnees(self, X: pd.DataFrame) -> dict:
        """Construit le dictionnaire de métadonnées pour l'interface.
    
        Reçoit : les caractéristiques d'entraînement et le modèle entraîné.
        Renvoie : un dictionnaire sérialisable en JSON.
        """
        return {
            "colonnes_entree": list(X.columns), # nom des colonnes
            "variables_numeriques": {
                c: {"min": float(X[c].min()), "max": float(X[c].max())}
                for c in X.columns
            }, # l'intervalle pour chaque variable numérique
            "classes": [str(c) for c in self._foret.classes_], # nom de chaque signe identifiable
            "graine": GRAINE,
            "metrique": None,  # *** jsp ce quil faut mettre dans métrique mais ct dans les consignes
        }

    def sauvegarder(self, X: pd.DataFrame, chemin_modele: str, chemin_meta: str) -> None:
        """Sauvegarde le modèle (joblib) et les métadonnées (JSON).
    
        Reçoit : le modèle entraîné, X, et les deux chemins de sortie.
        Échoue : OSError si l'écriture est impossible.
        """
        location = os.getcwd()
        chemin_modele = os.path.join(location, chemin_modele)
        chemin_meta = os.path.join(location, chemin_meta)

        for chemin in (chemin_modele, chemin_meta):
            os.makedirs(os.path.dirname(chemin), exist_ok=True)  # crée le dossier parent au besoin, dirname extrait la portion "directory"
    
        joblib.dump(self._foret, chemin_modele) # enregistre le modèle dans un fichier pour le recharger plus tard
        with open(chemin_meta, "w", encoding="utf-8") as fichier:
            json.dump(self.creer_metadonnees(X), fichier, indent=2, ensure_ascii=False)

def main() -> Modele:
    """Enchaîne tout l'entraînement : chargement, filtrage, entraînement, sauvegarde.

    Renvoie : le modèle entraîné.
    Échoue : FileNotFoundError si le CSV est introuvable,
             OSError si la sauvegarde est impossible.
    """
    jeu = JeuDonnees(charger_csv(CHEMIN_CSV), COLONNE_CIBLE)
    jeu.filtrer_personnes(PERSONNES_TRAIN)
    entrees_X, entrees_y = jeu.preparer_entrees()
    foret = Modele()
    foret.entrainer(entrees_X, entrees_y)
    foret.sauvegarder(entrees_X, CHEMIN_MODELE, CHEMIN_META)
    return foret

if __name__ == "__main__":
    main()
