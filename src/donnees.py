from config import COLONNE_PERSONNE

import os
import pandas as pd

class JeuDonnees:
    """Jeu de données des points de repère, avec sa colonne cible."""

    def __init__(self, df: pd.DataFrame, cible: str) -> None:
        self._df = df
        self._cible = cible

    def __len__(self) -> int:
        """Renvoie le nombre de lignes du jeu de données."""
        return len(self._df)

    def filtrer_personnes(self, personnes: list[str]) -> None:
        """Garde uniquement les lignes appartenant aux personnes données.

        Reçoit : la liste des personnes à garder.
        Modifie : le DataFrame interne (remplacé par une copie filtrée).
        """
        self._df = self._df[self._df[COLONNE_PERSONNE].isin(personnes)].copy() #regarde si la valeur dans "personne" est dans la liste des personnes qu'on veut traiter

    def preparer_entrees(self) -> tuple[pd.DataFrame, pd.Series]:
        """Sépare les caractéristiques (X) de la cible (y).

        Retire la cible et la colonne « personne » (ce n'est pas une caractéristique).

        Renvoie : un tuple (X, y).
        """
        a_retirer = [self._cible, COLONNE_PERSONNE]
        X = self._df.drop(columns=[c for c in a_retirer if c in self._df.columns]) #conserve toutes les colonnes qu'on ne souhaite pas retirer
        y = self._df[self._cible] # conserve uniquement la colonne cible
        return X, y

def charger_csv(chemin: str) -> pd.DataFrame:
    """Charge le fichier CSV des points de repère.

    Reçoit : le chemin du fichier CSV.
    Renvoie : un DataFrame pandas contenant toutes les lignes du fichier.
    """
    location = os.path.join(os.getcwd(), chemin) #creation du chemin qu'on va utiliser pour lire le csv
        #IMPORTANT! démarrer le fichier python du dossier INF007-PR02 (celui qui contient donnees/asl_landmarks.csv)

    with open(location, "r", encoding="utf-8") as fichier:
        df = pd.read_csv(fichier)

    return df
