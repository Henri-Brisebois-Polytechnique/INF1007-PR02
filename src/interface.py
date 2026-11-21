from gradio.components.image_editor import WebcamOptions
import numpy as np
import mediapipe as mp
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

from tools import detecteur, extraire_vecteur, charger_modele, vecteur_depuis_resultat
from sklearn.ensemble import RandomForestClassifier
from config import CHEMIN_MODELE_FORET
import pandas as pd
import gradio as gr

COLONNES = [f"{axe}{i}" for i in range(21) for axe in "xyz"]
MODELE=charger_modele(CHEMIN_MODELE_FORET)
MODELE.n_jobs = 1
d = detecteur()

def gradio_video(img_input: Image) -> tuple:

    """Détecte une main et prédit la lettre qu'elle représente en ASL

    :param img_input: image contenant potentiellement une main.
    :return: tuple regroupant les trois classes aux plus hautes probabilités et l'objet Image de l'image annotée.
    """
    if img_input is None:
        return "", None

    image = Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(img_input))

    resultat = extraire_vecteur(d, image, norm=False)   # no `with d:`
    if resultat is None:
        return "Aucune main détectée.", img_input

    vecteur = vecteur_depuis_resultat(resultat)  # même traitement que l'entraînement (inversion de main incluse)
    df = pd.DataFrame([vecteur], columns=COLONNES)

    pred = list(prediction(MODELE, df).items())
    top = f"{pred[0][0]} ({pred[0][1]*100:.2f} %)\n\nAutres possibilités : "
    for p in pred[1:3]:
        top += f"{p[0]} ({p[1]*100:.2f} %), "
    return top[:-2], dessiner_sur_image(img_input, resultat)

def prediction(modele: RandomForestClassifier, vecteurs: pd.DataFrame) -> dict:
    """Prédit une lettre.
    
    Le tableau des vecteurs doit avoir en première ligne l'identifiant de chaque composante sous la forme x0, y0, z0, x1, y1, z1...

    :param modele: objet RandomForestClassifier représentant le modèle de prédiction.
    :param vecteurs: tableau DataFrame (2,63) des composantes x,y,z des 21 vecteurs représentant une main.
    :return: dictionnaire trié des probabilités associées à chaque classe.
    """

    probas = modele.predict_proba(vecteurs)[0]
    dic = {str(c): float(p) for c,p in zip(modele.classes_, probas)} # zip(a,b) retourne sous la forme [(a1,b1),(a2,b2),...,(an,bn)] 
    return dict(sorted(dic.items(), key=lambda p: p[1], reverse=True))


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
            out_img= gr.Image(label="Vidéo annotée", height=480, format="jpeg")
            out_text= gr.Textbox(label= "Lettres prédites")

            cam = gr.Image(sources=["webcam"],streaming=True,
                           label="Webcam (input)", height=150, webcam_options=gr.WebcamOptions(constraints={"video":{"width": 640, "height":480}}))
            cam.stream(
                fn=gradio_video,
                inputs=cam,
                outputs=[out_text, out_img],
                stream_every=0.05
            )
    return demo
