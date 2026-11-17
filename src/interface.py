import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions, vision
import tools

import gradio as gr



def gradio_video(video_input):
    

    image= Image(image_format= ImageFormat.SRGB, data= np.asarray(video_input))

    with tools.detecteur as detecteur:
        resultat_detection = detecteur.detect(image)

    image_annotee = dessiner_sur_image(image.numpy_view(), resultat_detection)

    return image_annotee



def dessiner_sur_image(rgb_image, resultat_detection):
    liste_points= resultat_detection.hand_landmarks
    image_annotee= np.copy(rgb_image)

    for point in liste_points:


def build_interface():

    with gr.Blocks() as demo:
        photo= gr.Image(label="Photo d'une lettre en langage des signe")
        output= gr.Textbox(label="Lettre prédite")
        

    
    return demo

