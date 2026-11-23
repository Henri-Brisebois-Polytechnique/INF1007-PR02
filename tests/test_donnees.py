"""Tests du CSV de points de repère et des fonctions de donnees.py."""

import math
import os
import string

import numpy as np
import pytest

from donnees import JeuDonnees, charger_csv
from exceptions import ColonneManquanteError, FichierDonneesIntrouvableError

CHEMIN_CSV = os.path.join(os.path.dirname(__file__), "..", "donnees", "asl_landmarks.csv")
COLONNES_POINTS = [f"{axe}{i}" for i in range(21) for axe in "xyz"]
ENTETE = ["personne"] + COLONNES_POINTS + ["classe"]
CLASSES = set(string.digits + string.ascii_uppercase)
MIN_PAR_CLASSE = 500
TOLERANCE = 1e-5  # coordonnées arrondies à 6 décimales

DF = charger_csv(CHEMIN_CSV)  # chargé une seule fois pour tous les tests

# Liste de listes : une liste par main, avec ses 63 coordonnées [x0, y0, z0, x1, y1, z1, ...]
MAINS = DF[COLONNES_POINTS].values

# Tests sur le dataset

def test_entete():
    assert list(DF.columns) == ENTETE

def test_csv_non_vide():
    assert len(DF) > 0

def test_aucune_valeur_vide():
    for colonne in DF.columns:
        assert DF[colonne].isna().sum() == 0 # nombre de cases vides dans la colonne

def test_format_personne():
    for personne in DF["personne"]:
        assert personne[0] == "P"
        assert personne[1:].isdigit()

def test_plusieurs_personnes():
    assert len(set(DF["personne"])) >= 2

def test_toutes_les_classes_presentes():
    assert set(DF["classe"]) == CLASSES

def test_assez_d_exemples_par_classe():
    classes = list(DF["classe"])
    for classe in CLASSES:
        assert classes.count(classe) >= MIN_PAR_CLASSE

def test_coordonnees_finies():
    for main in MAINS:
        for valeur in main:
            assert math.isfinite(valeur)   # ni infini, ni NaN

def test_poignet_a_l_origine():
    for main in MAINS:
        assert main[0] == 0 # x0
        assert main[1] == 0 # y0
        assert main[2] == 0 # z0

def test_distance_max_au_poignet_vaut_un():
    for main in MAINS:
        points = main.reshape(21, 3)
        distances = np.linalg.norm(points, axis=1)
        assert max(distances) == pytest.approx(1, abs=TOLERANCE)


# Fonctions de donnees.py

def test_charger_csv_fichier_absent():
    with pytest.raises(FichierDonneesIntrouvableError):
        charger_csv("fichier_inexistant.csv")

def test_filtrer_personnes():
    jeu = JeuDonnees(DF, "classe")
    jeu.filtrer_personnes(["P1", "P2"])
    assert set(jeu._df["personne"]) == {"P1", "P2"}

def test_filtrer_personne_inexistante():
    jeu = JeuDonnees(DF, "classe")
    jeu.filtrer_personnes(["P99"])
    assert len(jeu) == 0

def test_preparer_entrees():
    X, y = JeuDonnees(DF, "classe").preparer_entrees()
    assert list(X.columns) == COLONNES_POINTS # ni "classe" ni "personne"
    assert len(X) == len(y) == len(DF)

def test_preparer_entrees_cible_absente():
    with pytest.raises(ColonneManquanteError):
        JeuDonnees(DF, "colonne_inexistante").preparer_entrees()
