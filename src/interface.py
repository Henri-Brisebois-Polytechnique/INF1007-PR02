
import numpy as np
import mediapipe as mp
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

from tools import detecteur, extraire_vecteur, charger_modele, vecteur_depuis_resultat
from sklearn.linear_model import LogisticRegression
from config import CHEMIN_MODELE_FORET, CHEMIN_ALPHABET
import pandas as pd
import gradio as gr

COLONNES = [f"{axe}{i}" for i in range(21) for axe in "xyz"]
MODELE=charger_modele(CHEMIN_MODELE_FORET)
MODELE.n_jobs = 1
LETTRES = [str(c) for c in MODELE.classes_]
d = detecteur()

def gradio_video(img_input: Image) -> tuple:

    """Détecte une main et prédit la lettre qu'elle représente en ASL

    :param img_input: image contenant potentiellement une main.
    :return: tuple regroupant les trois classes aux plus hautes probabilités et l'objet Image de l'image annotée.
    """
    if img_input is None:
        return "", None,None, gr.update(height=1)

    image = Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(img_input))

    resultat = extraire_vecteur(d, image, norm=False)   # no `with d:`
    if resultat is None:
        return "Aucune main détectée.", img_input,gr.skip(), gr.update(height=1)

    vecteur = vecteur_depuis_resultat(resultat)  # même traitement que l'entraînement (inversion de main incluse)
    df = pd.DataFrame([vecteur], columns=COLONNES)

    probas= prediction(MODELE, df)
    probas_decroissant= sorted(probas, key=lambda p: p[1], reverse=True)

    top = f"{probas_decroissant[0][0]} ({probas_decroissant[0][1]*100:.2f} %)\n\nAutres possibilités :\n"
    for p in probas_decroissant[1:3]:
        top += f"   {p[0]} ({p[1]*100:.2f} %),\n"
    return top[:-2], dessiner_sur_image(img_input, resultat), donnees_graphique(probas),gr.update(height=1)

def prediction(modele: LogisticRegression, vecteurs: pd.DataFrame)->list:
    """Prédit une lettre.
    
    Le tableau des vecteurs doit avoir en première ligne l'identifiant de chaque composante sous la forme x0, y0, z0, x1, y1, z1...

    :param modele: objet LogisticRegression représentant le modèle de prédiction.
    :param vecteurs: tableau DataFrame (2,63) des composantes x,y,z des 21 vecteurs représentant une main.
    :return: dictionnaire trié des probabilités associées à chaque classe.
    """

    probas = modele.predict_proba(vecteurs)[0]
    
    return list(zip(modele.classes_, probas)) # zip(a,b) retourne sous la forme [(a1,b1),(a2,b2),...,(an,bn)]


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

def donnees_graphique(probas: list[tuple[str, float]] | None = None) -> pd.DataFrame:
    if probas is None:
        valeurs = [0.0] * len(LETTRES)
    else:
        valeurs = [p * 100 for _, p in probas]
    return pd.DataFrame({"Lettre": LETTRES, "Probabilité (%)": valeurs})


def build_interface() -> gr.Blocks:
    """Construit l'interface Gradio.

    :return: Un objet Blocks.
    """
    with gr.Blocks(fill_height=True, fill_width=True) as demo:
    
        with gr.Row("", scale=1):

            #Webcam annotée avec vecteurs
            out_img= gr.Image(label="Vidéo annotée", format="jpeg", show_label=False, scale=3)
                
            #Colonne de droite
            with gr.Column(scale=1 ):

                #Input webcam
                cam = gr.Image(sources=["webcam"],streaming=True, show_label= False,
                               label="Webcam (input)", buttons=[])

                #Textbox prédiction
                out_text= gr.Textbox(value= "Pour débuter, activez la webcam en cliquant sur l'icône ci-dessus.",
                                     label= "Lettres prédites", scale=1, lines=5)

                #Guide Alphabet
                with gr.Tab("Alphabet"):
                    alphabet= gr.Image(value= CHEMIN_ALPHABET, format= "jpeg", show_label=False, scale=2, buttons=[])
                #Graphique
                with gr.Tab("Graphique"):
                    graphique = gr.BarPlot(
                                            value=donnees_graphique(),
                                            x="Lettre",
                                            y="Probabilité (%)",
                                            y_lim=[0, 100],
                                            sort=LETTRES,        
                                            show_label=False,
                                        )

                #Fonction par frame roule à chaque 0.01s
                cam.stream(
                    fn=gradio_video,
                    inputs=cam,
                    outputs=[out_text, out_img,graphique, cam],
                    stream_every=0.1,
                    
                )

    return demo
