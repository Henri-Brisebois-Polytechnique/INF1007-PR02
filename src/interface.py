import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions, vision
from tools import detecteur, extraire_vecteur, charger_modele
from sklearn.ensemble import RandomForestClassifier

import gradio as gr

first_time=True

def gradio_video(video_input):
    

    image= Image(image_format= ImageFormat.SRGB, data= np.asarray(video_input))
    
    with detecteur() as d:
        vecteurs = extraire_vecteur(d, image)

    if(first_time):
        modele=charger_modele
        first_time= False

    predite= prediction(modele, vecteurs)

    return predite
    ##image_annotee = dessiner_sur_image(image.numpy_view(), resultat_detection)

    

def prediction(modele: RandomForestClassifier, vecteurs) -> str:
    
    predite = modele.predict(vecteurs)

    return predite



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
        
        
    return demo

