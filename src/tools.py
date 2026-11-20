"""Outils communs de traitement d'image, partagés par l'extraction et l'interface."""

from exceptions import ModeleIntrouvableError

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerResult
import numpy as np

import time
import os
import joblib
from config import CHEMIN_TASK

from sklearn.ensemble import RandomForestClassifier
from collections.abc import Callable

MARGE = 0.25  # bordure ajoutée autour de l'image : aide la détection (92 % -> 99 %)

def detecteur() -> HandLandmarker:
    """Retourne le détecteur MediaPipe, construit à partir du fichier .task fournit par Google.
    :return: objet HandLandmarker, le modèle.
    """
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=CHEMIN_TASK),
        num_hands=1,
        min_hand_detection_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(options)


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

def extraire_vecteur(detecteur : HandLandmarker, image: mp.Image, normaliser: bool = True) -> np.ndarray | HandLandmarkerResult | None:
    """Récupère les vecteurs d'une image avec MediaPipe pour le traitement par le classificateur.
    
    :param detecteur: objet HandLandmarker représentant le modèle MediaPipe
    :param image: image MediaPipe.
    :param normaliser: précise s'il faut normaliser
    :return: vecteurs détectés dans l'image, en np.array si normalisé, en HandLandmarkerResult si non.

    """
    resultat = detecteur.detect(image)
    if not resultat.hand_landmarks:
        return None
    if(normaliser):
        points = np.array([[p.x, p.y, p.z] for p in resultat.hand_landmarks[0]])
        return normaliser(points)
    else:
        return resultat

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

def charger_modele(chemin_modele: str) -> RandomForestClassifier:
    """Recharge un modèle sauvegardé.
 
    Renvoie : le modèle entraîné au préalable.
    """
    
    location = os.path.join(os.getcwd(), chemin_modele) #creation du chemin qu'on va utiliser pour retourner le modèle
    #IMPORTANT! démarrer le fichier python du dossier INF007-PR02 (celui qui contient donnees/modele)

    if not os.path.exists(location):
        raise ModeleIntrouvableError("Le modèle est introuvable.")

    return joblib.load(location)  # recharge l’objet enregistré dans le fichier
