"""Extraction des points de repère de la main (MediaPipe) vers un CSV d'entraînement.

Parcourt asl_processed/train/<classe>/*.jpg et écrit une ligne par image :
    personne, x0, y0, z0, ..., x20, y20, z20, classe
"""

import csv
import os

import numpy as np

from tools import detecteur, charger_image, extraire_vecteur, chronometrer
from config import DOSSIER_SRC, RACINE, DOSSIER_IMAGES, CHEMIN_SORTIE

def verifier_dossier(dossier: str) -> None:
    """Vérifie que le dossier d'images existe.

    :param dossier: dossier contenant un sous-dossier par classe.
    :raises FileNotFoundError: si le dossier est absent (ou n'est pas un dossier).
    """
    if not os.path.isdir(dossier):
        raise FileNotFoundError(
            "Assurez-vous d'avoir les dossiers par classe dans ce format "
            f"à la racine du projet : asl_processed/train/ (cherché : {dossier})"
        )


def iterer_images(dossier: str):
    """Génère les couples (chemin de l'image, classe) du dossier d'images.

    :param dossier: dossier contenant un sous-dossier par classe.
    :return: générateur de tuples (chemin, classe).
    """
    for classe in sorted(os.listdir(dossier)):
        for nom in sorted(os.listdir(os.path.join(dossier, classe))):
            yield os.path.join(dossier, classe, nom), classe

@chronometrer
def main() -> None:
    """Détection par MediaPipe, puis ajout d'une ligne au CSV si détection valide."""
    verifier_dossier(DOSSIER_IMAGES)
    os.makedirs(os.path.dirname(CHEMIN_SORTIE), exist_ok=True)
    entete = ["personne"]
    entete += [f"{axe}{i}" for i in range(21) for axe in "xyz"] + ["classe"]
    nb_ok = nb_echecs = 0

    with (detecteur() as d,
          open(CHEMIN_SORTIE, "w", newline="", encoding="utf-8") as fichier):
        w = csv.writer(fichier)
        w.writerow(entete)
        for chemin, classe in iterer_images(DOSSIER_IMAGES):
            vecteur = extraire_vecteur(d, charger_image(chemin, marge=True))
            if vecteur is None:
                nb_echecs += 1
                continue
            personne = os.path.basename(chemin).split("_")[0]
            w.writerow([personne, *np.round(vecteur,6).tolist(), classe])
            nb_ok += 1
            if nb_ok % 1000 == 0:
                print(f"{nb_ok} images traitées...")

    print(f"Terminé : {nb_ok} vecteurs, {nb_echecs} images sans main détectée.")


if __name__ == "__main__":
    main()
