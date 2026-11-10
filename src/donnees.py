from config import COLONNE_PERSONNE

import os
import pandas as pd

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

def filtrer_personnes(df: pd.DataFrame, personnes: list[str]) -> pd.DataFrame:
    """Garde uniquement les lignes appartenant aux personnes données.

    Reçoit : le DataFrame complet et la liste des personnes à garder.
    Renvoie : une copie du DataFrame filtré.
    """
    return df[df["personne"].isin(personnes)].copy() #regarde si la valeur dans "personne" est dans la liste des personnes qu'on veut traiter

def preparer_entrees(df: pd.DataFrame, cible: str) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare les caractéristiques (X) de la cible (y).
 
    Retire la cible et la colonne « personne » (ce n'est pas une caractéristique). 
 
    Reçoit : le DataFrame d'entraînement (déjà filtré sur p1 à p5).
    Renvoie : un tuple (X, y).
    """
 
    a_retirer = [cible, COLONNE_PERSONNE]
    X = df.drop(columns=[c for c in a_retirer if c in df.columns]) #conserve toutes les colonnes qu'on ne souhaite pas retirer
    y = df[cible] # conserve uniquement la colonne cible
    return X, y
