"""Validation du CSV de points de repère produit par extraction.py."""

import csv
import os
import string

import numpy as np

CHEMIN_CSV = os.path.join(os.path.dirname(__file__), "..", "donnees", "asl_landmarks.csv")
ENTETE = ["personne"] + [f"{axe}{i}" for i in range(21) for axe in "xyz"] + ["classe"]
CLASSES = set(string.digits + string.ascii_uppercase)
MIN_PAR_CLASSE = 500
TOLERANCE = 1e-5  # coordonnées arrondies à 6 décimales


def lire_csv():
    """Retourne l'entête et la liste des lignes du CSV."""
    with open(CHEMIN_CSV, newline="", encoding="utf-8") as fichier:
        lignes = list(csv.reader(fichier))
    return lignes[0], lignes[1:]


def lire_points():
    """Retourne les coordonnées sous forme de tableau (n, 21, 3)."""
    _, lignes = lire_csv()
    return np.array([ligne[1:-1] for ligne in lignes], dtype=float).reshape(-1, 21, 3)


def test_csv_existe():
    assert os.path.isfile(CHEMIN_CSV)


def test_entete():
    entete, _ = lire_csv()
    assert entete == ENTETE


def test_csv_non_vide():
    _, lignes = lire_csv()
    assert len(lignes) > 0


def test_nombre_de_colonnes():
    _, lignes = lire_csv()
    for ligne in lignes:
        assert len(ligne) == len(ENTETE)


def test_aucune_valeur_vide():
    _, lignes = lire_csv()
    for ligne in lignes:
        assert "" not in ligne


def test_format_personne():
    _, lignes = lire_csv()
    for ligne in lignes:
        assert ligne[0].startswith("P") and ligne[0][1:].isdigit()


def test_plusieurs_personnes():
    _, lignes = lire_csv()
    personnes = {ligne[0] for ligne in lignes}
    assert len(personnes) >= 2


def test_toutes_les_classes_presentes():
    _, lignes = lire_csv()
    classes = {ligne[-1] for ligne in lignes}
    assert classes == CLASSES


def test_assez_d_exemples_par_classe():
    _, lignes = lire_csv()
    classes = [ligne[-1] for ligne in lignes]
    for classe in CLASSES:
        assert classes.count(classe) >= MIN_PAR_CLASSE


def test_coordonnees_finies():
    points = lire_points()
    assert np.isfinite(points).all()


def test_poignet_a_l_origine():
    points = lire_points()
    assert np.allclose(points[:, 0], 0)


def test_distance_max_au_poignet_vaut_un():
    points = lire_points()
    distances_max = np.linalg.norm(points, axis=2).max(axis=1)
    assert np.allclose(distances_max, 1, atol=TOLERANCE)
