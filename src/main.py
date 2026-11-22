import interface
import gradio.themes

theme= gradio.themes.Monochrome(text_size="lg")
if __name__ == "__main__":
    demo = interface.build_interface()
    demo.launch(theme=theme)