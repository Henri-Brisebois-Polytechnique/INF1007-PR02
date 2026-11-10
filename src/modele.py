from config import GRAINE

import os

import json
import time
from collections.abc import Callable
 
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def chronometrer(fonction: Callable) -> Callable:
    """Décorateur qui affiche le temps d'exécution d'une fonction."""
 
    def enveloppe(*args, **kwargs): # *args récupère les arguments positionnels ; **kwargs récupère les arguments nommés.
        """Chronomètre l'appel à la fonction et renvoie son résultat."""
        debut = time.perf_counter() # Enregistre l'heure de début
        resultat = fonction(*args, **kwargs) # Appelle la fonction avec ses arguments
        duree = time.perf_counter() - debut # Prends en note l'heure de fin et calcule le temps écoulé
        print(f"[{fonction.__name__}] terminé en {duree:.2f} s") # Affiche la durée
        return resultat # Renvoie le résultat de la fonction
 
    return enveloppe # Renvoie la fonction enveloppe qui remplace la fonction originale
 
def construire_modele() -> RandomForestClassifier:
    """Construit la forêt aléatoire (non entraînée).
 
    Renvoie : un RandomForestClassifier de 200 arbres, avec une graine fixe.
    """
    return RandomForestClassifier(n_estimators=200, random_state=GRAINE, n_jobs=-1)

@chronometrer
def entrainer(modele: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """Entraîne le modèle sur les données d'entraînement.
 
    Reçoit : le modèle, les caractéristiques X et la cible y.
    Renvoie : le même modèle, entraîné.

    """
    return modele.fit(X, y) #entraine le modèle

def creer_metadonnees(X: pd.DataFrame, modele: RandomForestClassifier) -> dict:
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
        "classes": [str(c) for c in modele.classes_], # nom de chaque signe identifiable
        "graine": GRAINE,
        "metrique": None,  # *** jsp ce quil faut mettre dans métrique mais ct dans les consignes
    }

def sauvegarder(modele: RandomForestClassifier, X: pd.DataFrame, chemin_modele: str, chemin_meta: str) -> None:
    """Sauvegarde le modèle (joblib) et les métadonnées (JSON).
 
    Reçoit : le modèle entraîné, X, et les deux chemins de sortie.
    Échoue : OSError si l'écriture est impossible.
    """
    location = os.getcwd()
    chemin_modele = os.path.join(location, chemin_modele)
    chemin_meta = os.path.join(location, chemin_meta)

    for chemin in (chemin_modele, chemin_meta):
        if not os.path.exists(chemin): # vérifie que le dossier n'existe pas déjà
            os.mkdir(chemin) # création du dossier à l'endroit voulu
 
    joblib.dump(modele, chemin_modele) # enregistre le modèle dans un fichier pour le recharger plus tard
    with open(chemin_meta, "w", encoding="utf-8") as fichier:
        json.dump(creer_metadonnees(X, modele), fichier, indent=2, ensure_ascii=False)


def charger_modele(chemin_modele: str) -> RandomForestClassifier:
    """Recharge un modèle sauvegardé.
 
    Échoue : FileNotFoundError si le fichier n'existe pas.
    """
    location = os.path.join(os.getcwd(), chemin_modele) #creation du chemin qu'on va utiliser pour retourner le modèle
    #IMPORTANT! démarrer le fichier python du dossier INF007-PR02 (celui qui contient donnees/modele)

    return joblib.load(location)  # recharge l’objet enregistré dans le fichier


# DEVRAIS-JE METTRE ÇA DANS UN MAIN() OR SOMETHING... 

# df = charger_csv(CHEMIN_CSV)
# df_train = filtrer_personnes(df, PERSONNES_A_TESTER)
# entrees_X, entrees_y = preparer_entrees(df_train, COLONNE_CIBLE)
# foret_vierge = construire_modele()
# foret_entrainee = entrainer(foret_vierge, entrees_X, entrees_y)
# sauvegarder(foret_entrainee, entrees_X, CHEMIN_MODELE, CHEMIN_META)