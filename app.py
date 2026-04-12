"""HF Spaces entry point — launches the Gradio benchmark dashboard."""
from ui.app import create_app
from ui.theme import CUSTOM_CSS

app = create_app()
app.launch(css=CUSTOM_CSS)
