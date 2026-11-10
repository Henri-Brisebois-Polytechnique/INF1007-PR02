"""Outils communs de traitement d'image, partagés par l'extraction et l'interface."""

import mediapipe as mp
import numpy as np

MARGE = 0.25  # bordure ajoutée autour de l'image : aide la détection (92 % -> 99 %)


def charger_image(chemin: str, marge: bool = False) -> mp.Image:
    """Charge une image, en l'entourant au besoin d'une bordure (répétition des bords).

    :param chemin: chemin de l'image.
    :param marge: True pour ajouter la bordure (extraction), False sinon (détection).
    :return: image MediaPipe.
    """
    pixels = mp.Image.create_from_file(chemin).numpy_view()[..., :3]
    if marge:
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
