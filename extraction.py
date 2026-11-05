"""Extraction des points de repère de la main (MediaPipe) vers un CSV d'entraînement.

Parcourt asl_processed/train/<classe>/*.jpg et écrit une ligne par image :
    personne, main, x0, y0, z0, ..., x20, y20, z20, classe
"""

import csv
import os

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

DOSSIER_IMAGES = os.path.join("asl_processed", "train")
CHEMIN_MODELE = "hand_landmarker.task"
CHEMIN_SORTIE = os.path.join("donnees", "asl_landmarks.csv")
MARGE = 0.25  # bordure ajoutée autour de l'image : aide la détection (92 % -> 99 %)


def iterer_images(dossier: str):
    """Génère les couples (chemin de l'image, classe) du dossier d'images.

    :param dossier: dossier contenant un sous-dossier par classe.
    :return: générateur de tuples (chemin, classe).
    """
    for classe in sorted(os.listdir(dossier)):
        for nom in sorted(os.listdir(os.path.join(dossier, classe))):
            yield os.path.join(dossier, classe, nom), classe


def charger_image(chemin: str) -> mp.Image:
    """Charge une image et l'entoure d'une bordure (répétition des bords).

    :param chemin: chemin de l'image.
    :return: image MediaPipe.
    """
    pixels = mp.Image.create_from_file(chemin).numpy_view()[..., :3]
    p = int(max(pixels.shape[:2]) * MARGE)
    pixels = np.pad(pixels, ((p, p), (p, p), (0, 0)), mode="edge")
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(pixels))


def normaliser(points: np.ndarray) -> np.ndarray:
    """Centre les 21 points sur le poignet et les met à l'échelle (calcul vectorisé).

    :param points: tableau (21, 3) des coordonnées MediaPipe.
    :return: tableau (63,) des coordonnées normalisées.
    """
    centres = points - points[0]
    return (centres / np.max(np.linalg.norm(centres, axis=1))).ravel()

# détection par MediaPipe, puis ajout d'une ligne au CSV si détection valide
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=CHEMIN_MODELE),
    num_hands=1,
    min_hand_detection_confidence=0.3,
)
os.makedirs(os.path.dirname(CHEMIN_SORTIE), exist_ok=True)
entete = ["personne", "main"]
entete += [f"{axe}{i}" for i in range(21) for axe in "xyz"] + ["classe"]
nb_ok = nb_echecs = 0

with (vision.HandLandmarker.create_from_options(options) as detecteur,
      open(CHEMIN_SORTIE, "w", newline="", encoding="utf-8") as fichier):
    w = csv.writer(fichier)
    w.writerow(entete)
    for chemin, classe in iterer_images(DOSSIER_IMAGES):
        resultat = detecteur.detect(charger_image(chemin))
        if not resultat.hand_landmarks:
            nb_echecs += 1
            continue
        points = np.array([[p.x, p.y, p.z] for p in resultat.hand_landmarks[0]])
        personne = os.path.basename(chemin).split("_")[0]
        main_detectee = resultat.handedness[0][0].category_name
        vecteur = np.round(normaliser(points), 6).tolist()
        w.writerow([personne, main_detectee, *vecteur, classe])
        nb_ok += 1
        if nb_ok % 1000 == 0:
            print(f"{nb_ok} images traitées...")

print(f"Terminé : {nb_ok} vecteurs, {nb_echecs} images sans main détectée.")
