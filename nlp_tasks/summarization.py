# summarization.py
from .model_manager import ModelManager

def summarize_text(text):
    manager = ModelManager()
    return manager.summarize(text)

