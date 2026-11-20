import numpy as np
from mediapipe import Image, ImageFormat
from tools import detecteur, extraire_vecteur, charger_modele
from sklearn.ensemble import RandomForestClassifier
import config
import pandas as pd
import gradio as gr

MODELE=charger_modele(config.CHEMIN_MODELE)


def gradio_video(img_input: Image) -> str:
    """Détecte une main et prédit la lettre qu'elle représente en ASL

    :param img_input: image contenant potentiellement une main.
    :return: lettre prédite.
    """
    if img_input is None:
        return ""

    image= Image(image_format= ImageFormat.SRGB, data= np.asarray(img_input))
    
    with detecteur() as d:
        vecteur = extraire_vecteur(d, image)
        if vecteur is None:
            return "Aucune main détectée."
        vecteurs_df = pd.DataFrame([vecteur],columns=[f"{axe}{i}" for i in range(21) for axe in "xyz"], )

    predite= prediction(MODELE, vecteurs_df)

    return predite
    ##image_annotee = dessiner_sur_image(image.numpy_view(), resultat_detection)


def prediction(modele: RandomForestClassifier, vecteurs: pd.DataFrame) -> str:
    """Prédit une lettre.
    
    Le tableau des vecteurs doit avoir en première ligne l'identifiant de chaque composante sous la forme x0, y0, z0, x1, y1, z1...

    :param modele: objet RandomForestClassifier représentant le modèle de prédiction.
    :param vecteurs: tableau DataFrame (2,63) des composantes x,y,z des 21 vecteurs représentant une main.
    :return: lettre prédite.
    """
    
    predite = modele.predict(vecteurs)
    
    return predite[0]


##def dessiner_sur_image(rgb_image, resultat_detection):
##    liste_points= resultat_detection.hand_landmarks
##    image_annotee= np.copy(rgb_image)
##
##    for point in liste_points:



def build_interface() -> gr.Blocks:
    """Construit l'interface Gradio.

    :return: Un objet Blocks.
    """
    with gr.Blocks() as demo:
        with gr.Tab("Live"):
            cam = gr.Image(sources=["webcam"],streaming=True)
            lettre= gr.Label()

            cam.stream(
                fn=gradio_video,
                inputs=cam,
                outputs=lettre,
                stream_every=0.5
            )

        with gr.Tab("Test avec fichier"):
            photo = gr.Image(sources="upload")
            lettre= gr.Label()

            photo.upload(
                fn=gradio_video,
                inputs=photo,
                outputs= lettre
            )
        
        
    return demo

