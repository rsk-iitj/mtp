from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, BertForTokenClassification
import torch


class ModelManager:
    def __init__(self):
        # Initialize the device to use GPU if available, otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load models and move them to the specified device
        self.summarizer = pipeline('summarization', model='t5-small', device=0 if torch.cuda.is_available() else -1)
        self.sentiment_analyzer = pipeline('sentiment-analysis',
                                           model='distilbert-base-uncased-finetuned-sst-2-english',
                                           device=0 if torch.cuda.is_available() else -1)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.keyword_model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        self.keyword_model.to(self.device)  # Move the keyword model to the specified device

    def summarize(self, text):
        return self.summarizer(text, max_length=512, min_length=100, do_sample=False)[0]['summary_text']

    def analyze_sentiment(self, text):
        # Split the text into manageable parts if too long
        parts = [text[i: i + 512] for i in range(0, len(text), 512)]
        sentiments = []
        for part in parts:
            if len(part) > 0:
                results = self.sentiment_analyzer(part)
                sentiments.append((results[0]['label'], results[0]['score']))
        # Aggregate or choose sentiment here as needed
        return sentiments[0] if sentiments else ("Neutral", 1.0)

    def extract_keywords(self, text):
        # Ensure that the text is within the model's capacity
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        inputs = inputs.to(self.device)  # Move inputs to the same device as the model
        outputs = self.keyword_model(**inputs)
        token_labels = outputs.logits.argmax(dim=-1).squeeze().tolist()

        # Decode the tokens back to words and filter out non-keyword tokens (assuming '0' is non-keyword)
        keywords = [self.tokenizer.decode([tok]) for tok, label in
                    zip(inputs.input_ids.squeeze().tolist(), token_labels) if label != 0]
        return keywords
