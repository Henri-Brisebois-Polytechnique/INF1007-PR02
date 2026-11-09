import gradio as gr


def prediction(image):
    return f"Wow! Belle main!"

def build_interface():

    with gr.Blocks() as demo:
        photo= gr.Image(label="Photo d'une lettre en langage des signe")
        green_btn = gr.Button("Prédire")
        output= gr.Textbox(label="Lettre prédite")
        green_btn.click(fn=prediction, inputs=photo, outputs=output, api_name= "predire")
        

    
    return demo

