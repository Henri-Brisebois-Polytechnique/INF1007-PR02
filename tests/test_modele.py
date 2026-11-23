"""Validation des modèles produit par modele.py."""

import os
import json
import joblib
import pytest

CHEMIN_MODELE_FORET = os.path.join(os.path.dirname(__file__), "..", "modeles", "foret", "modele.joblib")
CHEMIN_META_FORET = os.path.join(os.path.dirname(__file__), "..", "modeles", "foret", "metadonnees.json")
CHEMIN_MODELE_HGBOOST = os.path.join(os.path.dirname(__file__), "..", "modeles", "hgboost", "modele.joblib")
CHEMIN_META_HGBOOST = os.path.join(os.path.dirname(__file__), "..", "modeles", "hgboost", "metadonnees.json")
CHEMIN_MODELE_REGRESSION = os.path.join(os.path.dirname(__file__), "..", "modeles", "regressionlogistique", "modele.joblib")
CHEMIN_META_REGRESSION = os.path.join(os.path.dirname(__file__), "..", "modeles", "regressionlogistique", "metadonnees.json")

CHEMINS_MODELES = [CHEMIN_MODELE_FORET, CHEMIN_MODELE_HGBOOST, CHEMIN_MODELE_REGRESSION]
CHEMINS_METAS = [CHEMIN_META_FORET, CHEMIN_META_HGBOOST, CHEMIN_META_REGRESSION]
META_NOM_CLES = ["type_de_modele", "colonnes_entree", "variables_numeriques", "classes", "graine", "metrique"]


def lire_modele(chemin_modele):
    """Retourne le modèle prédéfini."""
    modele = joblib.load(chemin_modele)
    return modele

def lire_json(chemin_json):
    """Retourne le contenu du fichier JSON."""
    with open(chemin_json, "r", encoding="utf-8") as f:
        contenu = json.load(f)
    return contenu

@pytest.mark.parametrize("chemin_modele", CHEMINS_MODELES)
def test_modele_existe(chemin_modele):
    assert os.path.isfile(chemin_modele)

@pytest.mark.parametrize("chemin_meta", CHEMINS_METAS)
def test_metadonnes_existe(chemin_meta):
    assert os.path.isfile(chemin_meta)

@pytest.mark.parametrize("chemin_meta", CHEMINS_METAS)
def test_meta_non_vide(chemin_meta):
    contenu = lire_json(chemin_meta)
    assert len(contenu) > 0

@pytest.mark.parametrize("chemin_meta", CHEMINS_METAS)
def test_meta_structure(chemin_meta):
    contenu = lire_json(chemin_meta)
    cles = []
    for i in contenu.keys():
        cles.append(i)
    assert cles == META_NOM_CLES