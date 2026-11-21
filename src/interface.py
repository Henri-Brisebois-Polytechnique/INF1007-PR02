import numpy as np

import mediapipe as mp
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles


from tools import detecteur, extraire_vecteur, charger_modele, normaliser
from sklearn.ensemble import RandomForestClassifier
import config
import pandas as pd
import gradio as gr

COLONNES = [f"{axe}{i}" for i in range(21) for axe in "xyz"]
MODELE=charger_modele(config.CHEMIN_MODELE_FORET)
d= detecteur()




def gradio_video(img_input: Image) -> str:

    """Détecte une main et prédit la lettre qu'elle représente en ASL

    :param img_input: image contenant potentiellement une main.
    :return: lettre prédite.
    """
    if img_input is None:
        return "", None

    image = Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(img_input))

    resultat = extraire_vecteur(d, image, norm=False)   # no `with d:`
    if resultat is None:
        return "Aucune main détectée.", img_input

    points = np.array([[p.x, p.y, p.z] for p in resultat.hand_landmarks[0]])
    vecteur = normaliser(points)          # or points.ravel(), see below
    df = pd.DataFrame([vecteur], columns=COLONNES)

    return prediction(MODELE, df), dessiner_sur_image(img_input, resultat)

def prediction(modele: RandomForestClassifier, vecteurs: pd.DataFrame) -> str:
    """Prédit une lettre.
    
    Le tableau des vecteurs doit avoir en première ligne l'identifiant de chaque composante sous la forme x0, y0, z0, x1, y1, z1...

    :param modele: objet RandomForestClassifier représentant le modèle de prédiction.
    :param vecteurs: tableau DataFrame (2,63) des composantes x,y,z des 21 vecteurs représentant une main.
    :return: lettre prédite.
    """
    
    return modele.predict(vecteurs)[0]


def dessiner_sur_image(rgb_image: Image, resultat_detection: HandLandmarkerResult) ->Image:
    """Dessine les landmarks sur une image

    :param rgb_image: l'image à annoter.
    :param resultat_detection: les vecteurs à dessiner.
    :return: l'image annotée
    """
    hand_landmarks= resultat_detection.hand_landmarks

    annotated = np.copy(rgb_image)
    for hand in hand_landmarks: 
        
        drawing_utils.draw_landmarks(
            annotated,
            hand,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style(),
        )
    return annotated

    



def build_interface() -> gr.Blocks:
    """Construit l'interface Gradio.

    :return: Un objet Blocks.
    """
    with gr.Blocks() as demo:
        with gr.Tab("Live"):
            out_img= gr.Image(label="Vidéo annotée", height=480)
            out_text= gr.Textbox(label= "Lettre prédite")

            cam = gr.Image(sources=["webcam"],streaming=True,
                           label="Webcam (input)", height=150)


            cam.stream(
                fn=gradio_video,
                inputs=cam,
                outputs=[out_text, out_img],
                stream_every=0.1
            )

        
        
        
    return demo

