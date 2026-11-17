import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions, vision
from tools import detecteur, extraire_vecteur, charger_modele
from sklearn.ensemble import RandomForestClassifier
import config
import pandas as pd

import gradio as gr



def gradio_video(video_input):
    

    image= Image(image_format= ImageFormat.SRGB, data= np.asarray(video_input))
    
    with detecteur() as d:
        vecteurs_df = pd.DataFrame([extraire_vecteur(d, image)],
                                   columns=[f"{axe}{i}" for i in range(21) for axe in "xyz"], )


    
    modele=charger_modele(config.CHEMIN_MODELE)
    

    predite= prediction(modele, vecteurs_df)

    return predite
    ##image_annotee = dessiner_sur_image(image.numpy_view(), resultat_detection)

    

def prediction(modele: RandomForestClassifier, vecteurs) -> str:
    
    predite = modele.predict(vecteurs)
    
    
    return predite[0]



##def dessiner_sur_image(rgb_image, resultat_detection):
##    liste_points= resultat_detection.hand_landmarks
##    image_annotee= np.copy(rgb_image)
##
##    for point in liste_points:



def build_interface():

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

