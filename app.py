"""HF Spaces entry point — launches the Gradio benchmark dashboard."""
from ui.app import create_app
from ui.theme import CUSTOM_CSS

app = create_app()
app.launch(server_name="0.0.0.0", server_port=7860, css=CUSTOM_CSS)
