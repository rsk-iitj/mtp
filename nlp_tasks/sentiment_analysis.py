# sentiment_analysis.py
from .model_manager import ModelManager

def analyze_sentiment(text):
    manager = ModelManager()
    sentiment_label, sentiment_score = manager.analyze_sentiment(text)
    return sentiment_label, sentiment_score
