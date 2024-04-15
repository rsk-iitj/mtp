# keyword_extraction.py
from .model_manager import ModelManager

def extract_keywords(text):
    manager = ModelManager()
    return manager.extract_keywords(text)