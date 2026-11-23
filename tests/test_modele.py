"""Validation des modèles produit par modele.py."""

import os
import json
import joblib

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

for c in CHEMINS_MODELES:

    def test_modele_existe():
        print("allo")
        assert os.path.isfile(c)


for c in CHEMINS_METAS:

    def test_metadonnes_existe():
        assert os.path.isfile(c)

    def test_meta_non_vide():
        contenu = lire_json(c)
        assert len(contenu) > 0

    def test_meta_structure():
        contenu = lire_json(c)
        cles = []
        for i in contenu.keys():
            cles.append(i)
        assert cles == META_NOM_CLES